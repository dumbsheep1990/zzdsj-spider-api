from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query, Path, Form, UploadFile, File
from typing import Dict, Any, List, Optional
from datetime import datetime
from bson.objectid import ObjectId
import asyncio
from pydantic import BaseModel
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from db.connection import get_async_db
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawler.models.article import GovArticle
from config.logging_config import setup_logging

logger = setup_logging("AIExtraction")

router = APIRouter(prefix="/ai", tags=["ai"])


class ExtractionRequest(BaseModel):
    url: Optional[str] = None
    html_content: Optional[str] = None
    ai_model: str = "gpt-3.5-turbo"
    custom_prompt: Optional[str] = None
    extraction_schema: Optional[Dict[str, Any]] = None


async def perform_extraction(
    extraction_id: ObjectId,
    source_type: str,
    source: str,
    ai_model: str,
    custom_prompt: Optional[str],
    extraction_schema: Optional[Dict[str, Any]],
    db
):
    """Background task to perform content extraction using crawl4ai"""
    try:
        # Get LLM configuration from database
        llm_config = await db.llm_configs.find_one({"is_active": True})
        
        if not llm_config:
            # Use default configuration if none is found
            llm_config = {
                "provider": "openai",
                "openai_config": {
                    "api_key": "YOUR_API_KEY",  # Should be set by the user
                    "model": ai_model
                }
            }
        
        # Create extraction strategy
        instruction = custom_prompt or "从网页中提取标题、发布日期、发布部门和正文内容。如果附件存在，也提取附件信息。"
        schema = extraction_schema or GovArticle.schema()
        
        # Configure LLM based on provider
        provider = llm_config.get("provider")
        extraction_strategy = None
        
        if provider == "openai":
            config = llm_config.get("openai_config", {})
            api_key = config.get("api_key")
            model = ai_model or config.get("model", "gpt-3.5-turbo")
            
            if not api_key:
                raise ValueError("OpenAI API key not provided, cannot perform extraction")
            
            extraction_strategy = LLMExtractionStrategy(
                provider=f"openai/{model}",
                api_token=api_key,
                schema=schema,
                extraction_type="schema",
                instruction=instruction
            )
        
        elif provider == "ollama":
            config = llm_config.get("ollama_config", {})
            base_url = config.get("base_url", "http://localhost:11434")
            model = ai_model or config.get("model", "llama2")
            
            extraction_strategy = LLMExtractionStrategy(
                provider=f"ollama/{model}",
                base_url=base_url,
                schema=schema,
                extraction_type="schema",
                instruction=instruction
            )
        
        elif provider == "custom":
            config = llm_config.get("custom_model_config", {})
            url = config.get("url")
            api_key = config.get("api_key")
            headers = config.get("headers", {})
            
            if not url:
                raise ValueError("Custom model URL not provided, cannot perform extraction")
            
            extraction_strategy = LLMExtractionStrategy(
                provider="custom",
                api_token=api_key,
                schema=schema,
                extraction_type="schema",
                instruction=instruction,
                base_url=url,
                headers=headers
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        browser_config = BrowserConfig(verbose=True)
        run_config = CrawlerRunConfig(
            extraction_strategy=extraction_strategy,
            cache_mode=CacheMode.BYPASS
        )
        
        # Extract content using crawl4ai
        async with AsyncWebCrawler(config=browser_config) as crawler:
            # Process differently based on source type
            if source_type == "url":
                result = await crawler.arun(url=source, config=run_config)
                
                # Update extraction record with result
                await db.ai_extractions.update_one(
                    {"_id": extraction_id},
                    {"$set": {
                        "status": "completed",
                        "result": result.extracted_content if result and result.extracted_content else {},
                        "completed_at": datetime.now(),
                        "html": result.html if result else ""
                    }}
                )
                
            elif source_type == "html":
                # For HTML content we need to use a different approach
                # Create a temporary HTML file for crawl4ai to load
                import tempfile
                
                with tempfile.NamedTemporaryFile(suffix=".html", mode="w+", delete=False) as tmp:
                    tmp.write(source)
                    tmp_path = tmp.name
                
                try:
                    # Use file:// protocol to load the temp HTML file
                    file_url = f"file://{tmp_path}"
                    result = await crawler.arun(url=file_url, config=run_config)
                    
                    # Update extraction record with result
                    await db.ai_extractions.update_one(
                        {"_id": extraction_id},
                        {"$set": {
                            "status": "completed",
                            "result": result.extracted_content if result and result.extracted_content else {},
                            "completed_at": datetime.now()
                        }}
                    )
                    
                finally:
                    # Cleanup the temporary file
                    import os
                    try:
                        os.unlink(tmp_path)
                    except Exception as e:
                        logger.error(f"Error deleting temporary file: {str(e)}")
            
            else:
                raise ValueError(f"Unsupported source type: {source_type}")
        
    except Exception as e:
        logger.error(f"Extraction error: {str(e)}")
        # Update record with error
        await db.ai_extractions.update_one(
            {"_id": extraction_id},
            {"$set": {
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.now()
            }}
        )


@router.post("/extract-url",
             summary="Extract content from URL",
             description="Extracts structured content from the given URL using AI.",
             response_description="Extraction result"
             )
async def extract_from_url(
        background_tasks: BackgroundTasks,
        url: str = Form(..., description="URL to extract content from"),
        ai_model: str = Form("gpt-3.5-turbo", description="AI model to use for extraction"),
        custom_prompt: Optional[str] = Form(None, description="Custom extraction prompt"),
        db=Depends(get_async_db)
):
    """Extract content from URL using AI"""
    try:
        # Validate URL
        try:
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError("Invalid URL format")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid URL format")
        
        # Create extraction record
        extraction_id = ObjectId()
        extraction_record = {
            "_id": extraction_id,
            "type": "url",
            "source": url,
            "model": ai_model,
            "status": "processing",
            "created_at": datetime.now()
        }
        
        if custom_prompt:
            extraction_record["custom_prompt"] = custom_prompt
        
        await db.ai_extractions.insert_one(extraction_record)
        
        # Schedule extraction in background
        background_tasks.add_task(
            perform_extraction,
            extraction_id,
            "url",
            url,
            ai_model,
            custom_prompt,
            None,
            db
        )
        
        return {
            "message": "Extraction started",
            "extraction_id": str(extraction_id),
            "status": "processing"
        }
    except HTTPException:
        raise
    except Exception as e:
        # Update record with error if it was created
        if 'extraction_id' in locals():
            await db.ai_extractions.update_one(
                {"_id": extraction_id},
                {"$set": {
                    "status": "failed",
                    "error": str(e),
                    "completed_at": datetime.now()
                }}
            )
        
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@router.post("/extract-html",
             summary="Extract content from HTML",
             description="Extracts structured content from the provided HTML content using AI.",
             response_description="Extraction result"
             )
async def extract_from_html(
        background_tasks: BackgroundTasks,
        html_content: str = Form(..., description="HTML content to extract from"),
        ai_model: str = Form("gpt-3.5-turbo", description="AI model to use for extraction"),
        custom_prompt: Optional[str] = Form(None, description="Custom extraction prompt"),
        db=Depends(get_async_db)
):
    """Extract content from HTML using AI"""
    try:
        # Create extraction record
        extraction_id = ObjectId()
        extraction_record = {
            "_id": extraction_id,
            "type": "html",
            "model": ai_model,
            "status": "processing",
            "created_at": datetime.now()
        }
        
        if custom_prompt:
            extraction_record["custom_prompt"] = custom_prompt
        
        # Save truncated HTML preview
        html_preview = html_content[:1000] + ("..." if len(html_content) > 1000 else "")
        extraction_record["html_preview"] = html_preview
        
        await db.ai_extractions.insert_one(extraction_record)
        
        # Schedule extraction in background
        background_tasks.add_task(
            perform_extraction,
            extraction_id,
            "html",
            html_content,
            ai_model,
            custom_prompt,
            None,
            db
        )
        
        return {
            "message": "Extraction started",
            "extraction_id": str(extraction_id),
            "status": "processing"
        }
    except Exception as e:
        # Update record with error if it was created
        if 'extraction_id' in locals():
            await db.ai_extractions.update_one(
                {"_id": extraction_id},
                {"$set": {
                    "status": "failed",
                    "error": str(e),
                    "completed_at": datetime.now()
                }}
            )
        
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@router.get("/history",
            summary="Get extraction history",
            description="Returns a list of AI extraction history with pagination.",
            response_description="List of extractions"
            )
async def get_extract_history(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(20, ge=1, le=100, description="Items per page"),
        type_filter: Optional[str] = Query(None, description="Filter by extraction type (url, html)"),
        status: Optional[str] = Query(None, description="Filter by status (processing, completed, failed)"),
        db=Depends(get_async_db)
):
    """Get extraction history with pagination"""
    try:
        # Calculate skip value for pagination
        skip = (page - 1) * page_size
        
        # Build filter
        filter_query = {}
        if type_filter:
            filter_query["type"] = type_filter
        if status:
            filter_query["status"] = status
        
        # Get total count
        total_count = await db.ai_extractions.count_documents(filter_query)
        
        # Get paginated extractions
        cursor = db.ai_extractions.find(filter_query).sort("created_at", -1).skip(skip).limit(page_size)
        extractions = await cursor.to_list(length=page_size)
        
        # Format response
        formatted_extractions = []
        for extraction in extractions:
            formatted_extraction = {
                "id": str(extraction["_id"]),
                "type": extraction.get("type"),
                "model": extraction.get("model"),
                "status": extraction.get("status"),
                "created_at": extraction.get("created_at").isoformat() if extraction.get("created_at") else None,
                "completed_at": extraction.get("completed_at").isoformat() if extraction.get("completed_at") else None
            }
            
            # Add source info
            if "source" in extraction:
                formatted_extraction["source"] = extraction["source"]
            if "html_preview" in extraction:
                formatted_extraction["html_preview"] = extraction["html_preview"]
            
            # Add error info if failed
            if extraction.get("status") == "failed" and "error" in extraction:
                formatted_extraction["error"] = extraction["error"]
            
            # Add result preview
            if "result" in extraction:
                # Just include title for preview
                if extraction["result"] and "title" in extraction["result"]:
                    formatted_extraction["title"] = extraction["result"]["title"]
            
            formatted_extractions.append(formatted_extraction)
        
        # Build pagination metadata
        total_pages = (total_count + page_size - 1) // page_size
        
        return {
            "data": formatted_extractions,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_count": total_count,
                "total_pages": total_pages
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve extraction history: {str(e)}")


@router.get("/history/{extraction_id}",
            summary="Get extraction result",
            description="Returns the result of a specific extraction.",
            response_description="Extraction result"
            )
async def get_extract_result(
        extraction_id: str = Path(..., description="Extraction ID"),
        db=Depends(get_async_db)
):
    """Get detailed result for a specific extraction"""
    try:
        # Find extraction record
        extraction = await db.ai_extractions.find_one({"_id": ObjectId(extraction_id)})
        
        if not extraction:
            raise HTTPException(status_code=404, detail="Extraction not found")
        
        # Format response
        response = {
            "id": str(extraction["_id"]),
            "type": extraction.get("type"),
            "model": extraction.get("model"),
            "status": extraction.get("status"),
            "created_at": extraction.get("created_at").isoformat() if extraction.get("created_at") else None,
            "completed_at": extraction.get("completed_at").isoformat() if extraction.get("completed_at") else None
        }
        
        # Add source info
        if "source" in extraction:
            response["source"] = extraction["source"]
        if "html_preview" in extraction:
            response["html_preview"] = extraction["html_preview"]
        if "html" in extraction:
            response["html"] = extraction["html"]
        
        # Add custom prompt if available
        if "custom_prompt" in extraction:
            response["custom_prompt"] = extraction["custom_prompt"]
        
        # Add result and error info
        if "result" in extraction:
            response["result"] = extraction["result"]
        
        if extraction.get("status") == "failed" and "error" in extraction:
            response["error"] = extraction["error"]
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve extraction result: {str(e)}")


@router.delete("/history/{extraction_id}",
               summary="Delete extraction",
               description="Deletes a specific extraction record.",
               response_description="Deletion result"
               )
async def delete_extract_history(
        extraction_id: str = Path(..., description="Extraction ID"),
        db=Depends(get_async_db)
):
    """Delete a specific extraction record"""
    try:
        # Delete extraction record
        result = await db.ai_extractions.delete_one({"_id": ObjectId(extraction_id)})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Extraction not found")
        
        return {
            "message": "Extraction deleted successfully",
            "id": extraction_id
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete extraction: {str(e)}")


@router.delete("/history",
               summary="Batch delete extractions",
               description="Deletes multiple extraction records.",
               response_description="Deletion result"
               )
async def batch_delete_history(
        ids: List[str] = Query(..., description="List of extraction IDs to delete"),
        db=Depends(get_async_db)
):
    """Delete multiple extraction records"""
    try:
        # Convert IDs to ObjectIds
        object_ids = [ObjectId(id_) for id_ in ids]
        
        # Delete extraction records
        result = await db.ai_extractions.delete_many({"_id": {"$in": object_ids}})
        
        return {
            "message": f"Deleted {result.deleted_count} extraction records",
            "deleted_count": result.deleted_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete extractions: {str(e)}")


@router.post("/extract-batch",
             summary="Batch extract content",
             description="Extracts content from multiple URLs in a batch.",
             response_description="Batch extraction result"
             )
async def extract_batch(
        background_tasks: BackgroundTasks,
        urls: List[str] = Form(..., description="List of URLs to extract from"),
        ai_model: str = Form("gpt-3.5-turbo", description="AI model to use for extraction"),
        custom_prompt: Optional[str] = Form(None, description="Custom extraction prompt"),
        db=Depends(get_async_db)
):
    """Extract content from multiple URLs"""
    try:
        batch_id = str(ObjectId())
        extraction_ids = []
        
        for url in urls:
            # Validate URL
            try:
                parsed_url = urlparse(url)
                if not parsed_url.scheme or not parsed_url.netloc:
                    continue  # Skip invalid URLs
            except Exception:
                continue
            
            # Create extraction record
            extraction_id = ObjectId()
            extraction_record = {
                "_id": extraction_id,
                "type": "url",
                "source": url,
                "model": ai_model,
                "status": "processing",
                "created_at": datetime.now(),
                "batch_id": batch_id
            }
            
            if custom_prompt:
                extraction_record["custom_prompt"] = custom_prompt
            
            await db.ai_extractions.insert_one(extraction_record)
            extraction_ids.append(str(extraction_id))
            
            # Schedule extraction in background
            background_tasks.add_task(
                perform_extraction,
                extraction_id,
                "url",
                url,
                ai_model,
                custom_prompt,
                None,
                db
            )
        
        return {
            "message": f"Batch extraction started for {len(extraction_ids)} URLs",
            "batch_id": batch_id,
            "extraction_ids": extraction_ids,
            "status": "processing"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch extraction failed: {str(e)}")


@router.get("/batch/{batch_id}",
            summary="Get batch extraction status",
            description="Returns the status of a batch extraction.",
            response_description="Batch extraction status"
            )
async def get_batch_status(
        batch_id: str = Path(..., description="Batch ID"),
        db=Depends(get_async_db)
):
    """Get status of a batch extraction"""
    try:
        # Get all extraction records for this batch
        cursor = db.ai_extractions.find({"batch_id": batch_id})
        extractions = await cursor.to_list(length=100)
        
        if not extractions:
            raise HTTPException(status_code=404, detail="Batch not found")
        
        # Count status
        total = len(extractions)
        completed = sum(1 for e in extractions if e.get("status") == "completed")
        failed = sum(1 for e in extractions if e.get("status") == "failed")
        processing = sum(1 for e in extractions if e.get("status") == "processing")
        
        # Calculate overall status
        overall_status = "processing"
        if processing == 0:
            if failed == total:
                overall_status = "failed"
            else:
                overall_status = "completed"
        
        # Format extraction previews
        extraction_previews = []
        for extraction in extractions:
            preview = {
                "id": str(extraction["_id"]),
                "status": extraction.get("status"),
                "source": extraction.get("source", "")
            }
            
            if extraction.get("status") == "completed" and "result" in extraction:
                if extraction["result"] and "title" in extraction["result"]:
                    preview["title"] = extraction["result"]["title"]
            
            extraction_previews.append(preview)
        
        return {
            "batch_id": batch_id,
            "status": overall_status,
            "total": total,
            "completed": completed,
            "failed": failed,
            "processing": processing,
            "extractions": extraction_previews
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve batch status: {str(e)}")
