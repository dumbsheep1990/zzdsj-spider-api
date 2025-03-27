from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query, Path, Body, File, Form, UploadFile
from app.api.models.schemas import CrawlerConfig, CrawlerStatus, LLMModelInfo, CustomCrawlResult, AdvancedCrawlerConfig
from app.services.crawler_service import CrawlerService
from db.connection import get_async_db
from typing import List, Dict, Any, Optional
from datetime import datetime
from urllib.parse import urlparse
from bson.objectid import ObjectId
import uuid
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy

router = APIRouter(prefix="/crawler", tags=["crawler"])

@router.get("/heartbeat",
           summary="Get crawler service status",
           description="Returns a heartbeat status to indicate the crawler service is running.",
           response_description="Crawler service heartbeat status"
           )
async def get_heartbeat():
    """Get crawler service heartbeat status"""
    try:
        return {
            "status": "online",
            "timestamp": datetime.now().isoformat(),
            "service": "crawler"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve crawler heartbeat: {str(e)}")


# ... (rest of the code remains the same)

@router.post("/custom",
             response_model=CustomCrawlResult,
             summary="Perform a custom crawl on a single URL",
             description="Crawls a single URL with custom settings and extracts content.",
             response_description="Result of the custom crawl"
             )
async def custom_crawl(background_tasks: BackgroundTasks, 
                       url: str = Form(..., description="URL to crawl"),
                       depth: int = Form(1, ge=1, le=5, description="Crawl depth"),
                       use_browser: bool = Form(True, description="Whether to use browser for crawling"),
                       follow_links: bool = Form(False, description="Whether to follow links on the page"),
                       extract_content: bool = Form(True, description="Whether to extract content"),
                       use_llm: bool = Form(False, description="Whether to use LLM for extraction"),
                       ai_model: Optional[str] = Form(None, description="AI model to use for extraction"),
                       db=Depends(get_async_db)):
    """Perform a custom crawl on a single URL"""
    try:
        # Validate URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise HTTPException(status_code=400, detail="Invalid URL")
        
        # Create crawl record
        crawl_id = str(ObjectId())
        now = datetime.now()
        
        # Create record in database
        await db.custom_crawls.insert_one({
            "_id": ObjectId(crawl_id),
            "url": url,
            "depth": depth,
            "use_browser": use_browser,
            "follow_links": follow_links,
            "extract_content": extract_content,
            "use_llm": use_llm,
            "ai_model": ai_model,
            "status": "processing",
            "created_at": now
        })
        
        # Prepare for crawl using crawl4ai
        browser_config = BrowserConfig(
            headless=True,  # Run in headless mode
            viewport_width=1280,
            viewport_height=800,
            timeout=30000,  # 30 seconds timeout
            js_enabled=True,
            verbose=True
        )
        
        # Setup extraction strategy
        extraction_strategy = LLMExtractionStrategy()
        
        if use_llm and extract_content:
            # Get LLM configuration if use_llm is enabled
            llm_config = await db.llm_configs.find_one({"is_active": True})
            
            if not llm_config:
                # Use default configuration if none is found
                raise HTTPException(status_code=400, detail="No active LLM configuration found")
            
            # Create extraction strategy based on provider
            provider = llm_config.get("provider")
            
            if provider == "openai":
                config = llm_config.get("openai_config", {})
                api_key = config.get("api_key")
                model = ai_model or config.get("model", "gpt-3.5-turbo")
                
                if not api_key:
                    raise HTTPException(status_code=400, detail="OpenAI API key not provided")
                
                extraction_strategy = LLMExtractionStrategy(
                    provider=f"openai/{model}",
                    api_token=api_key,
                    extraction_type="structured",
                    instruction="提取网页中的关键信息，包括标题、发布日期、正文内容等。"
                )
                
            elif provider == "ollama":
                config = llm_config.get("ollama_config", {})
                base_url = config.get("base_url", "http://localhost:11434")
                model = ai_model or config.get("model", "llama2")
                
                extraction_strategy = LLMExtractionStrategy(
                    provider=f"ollama/{model}",
                    base_url=base_url,
                    extraction_type="structured",
                    instruction="提取网页中的关键信息，包括标题、发布日期、正文内容等。"
                )
            
            elif provider == "custom":
                config = llm_config.get("custom_model_config", {})
                url = config.get("url")
                api_key = config.get("api_key")
                headers = config.get("headers", {})
                
                if not url:
                    raise HTTPException(status_code=400, detail="Custom model URL not provided")
                
                extraction_strategy = LLMExtractionStrategy(
                    provider="custom",
                    api_token=api_key,
                    extraction_type="structured",
                    instruction="提取网页中的关键信息，包括标题、发布日期、正文内容等。",
                    base_url=url,
                    headers=headers
                )
        
        # Run config
        run_config = CrawlerRunConfig(
            follow_links=follow_links,
            max_depth=depth,
            extraction_strategy=extraction_strategy if extract_content else None,
            cache_mode=CacheMode.BYPASS  # Don't use cache
        )
        
        # Perform crawl in background
        background_tasks.add_task(
            perform_crawl, 
            crawl_id=crawl_id, 
            url=url, 
            use_browser=use_browser, 
            run_config=run_config,
            db=db
        )
        
        # Return initial status
        return {
            "crawl_id": crawl_id,
            "url": url,
            "status": "processing",
            "created_at": now.isoformat(),
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start custom crawl: {str(e)}")


async def perform_crawl(crawl_id: str, url: str, use_browser: bool, run_config: Any, db):
    """Background task to perform the actual crawl"""
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig
        
        # Setup browser config
        browser_config = BrowserConfig(
            headless=True,
            viewport_width=1280,
            viewport_height=800,
            timeout=30000,  # 30 seconds timeout
            js_enabled=True,
            verbose=True
        )
        
        # Perform the crawl
        async with AsyncWebCrawler(config=browser_config if use_browser else None) as crawler:
            result = await crawler.arun(url=url, config=run_config)
            
            # Update database with results
            update_data = {
                "status": "completed",
                "completed_at": datetime.now(),
            }
            
            if result:
                # Extract links if available
                links = result.links if hasattr(result, "links") else []
                
                # Add extracted data
                update_data.update({
                    "content": result.extracted_content if hasattr(result, "extracted_content") else {},
                    "html": result.html if hasattr(result, "html") else "",
                    "links": links
                })
            
            # Update the database record
            await db.custom_crawls.update_one(
                {"_id": ObjectId(crawl_id)},
                {"$set": update_data}
            )
    
    except Exception as e:
        # Update database with error
        try:
            await db.custom_crawls.update_one(
                {"_id": ObjectId(crawl_id)},
                {"$set": {
                    "status": "failed",
                    "error": str(e),
                    "completed_at": datetime.now()
                }}
            )
        except Exception as update_error:
            # Log if we can't update database
            print(f"Failed to update custom crawl status: {str(update_error)}")


@router.get("/custom/{crawl_id}",
            response_model=CustomCrawlResult,
            summary="Get result of a custom crawl",
            description="Returns the result of a previously executed custom crawl.",
            response_description="Custom crawl result"
            )
async def get_custom_crawl_result(crawl_id: str = Path(..., description="Crawl ID"), 
                                 db=Depends(get_async_db)):
    """Get the result of a custom crawl"""
    try:
        from bson.objectid import ObjectId
        
        # Convert string ID to ObjectId
        try:
            obj_id = ObjectId(crawl_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid crawl ID format")
        
        # Fetch crawl from database
        crawl = await db.custom_crawls.find_one({"_id": obj_id})
        
        if not crawl:
            raise HTTPException(status_code=404, detail="Custom crawl not found")
        
        # Format response
        result = {
            "crawl_id": str(crawl["_id"]),
            "url": crawl["url"],
            "status": crawl["status"],
            "created_at": crawl["created_at"].isoformat() if isinstance(crawl["created_at"], datetime) else crawl["created_at"],
        }
        
        # Add optional fields if they exist
        if "content" in crawl:
            result["content"] = crawl["content"]
        
        if "html" in crawl:
            result["html"] = crawl["html"]
        
        if "links" in crawl:
            result["links"] = crawl["links"]
        
        if "error" in crawl:
            result["error"] = crawl["error"]
        
        if "completed_at" in crawl:
            result["completed_at"] = crawl["completed_at"].isoformat() if isinstance(crawl["completed_at"], datetime) else crawl["completed_at"]
        
        return result
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get custom crawl result: {str(e)}")


@router.post("/start-with-file",
             response_model=CrawlerStatus,
             summary="Start crawler with URLs from file",
             description="Starts a crawler job using URLs from an uploaded file.",
             response_description="Current status of the crawler after starting"
             )
async def start_crawler_with_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="File containing URLs to crawl"),
    max_pages: Optional[int] = Form(None, description="Maximum number of pages to crawl"),
    max_depth: Optional[int] = Form(None, description="Maximum crawl depth"),
    include_subdomains: bool = Form(True, description="Whether to crawl subdomains"),
    crawl_interval: float = Form(1.0, description="Interval between requests in seconds"),
    use_llm: bool = Form(False, description="Whether to use LLM for content extraction"),
    max_concurrent_tasks: int = Form(10, ge=1, le=100, description="Maximum number of concurrent crawling tasks"),
    llm_provider: Optional[str] = Form(None, description="LLM provider"),
    llm_api_key: Optional[str] = Form(None, description="LLM API key"),
    llm_model: Optional[str] = Form(None, description="LLM model"),
    llm_base_url: Optional[str] = Form(None, description="LLM base URL for custom providers")
):
    """Start the crawler with URLs from a file"""
    try:
        # Read URLs from the uploaded file
        contents = await file.read()
        urls = [line.strip() for line in contents.decode('utf-8').splitlines() if line.strip()]
        
        if not urls:
            raise HTTPException(status_code=400, detail="No valid URLs found in the file")
        
        # Use the first URL as the base URL
        base_url = urls[0]
        
        # Create crawler config
        config = {
            "base_url": base_url,
            "max_pages": max_pages,
            "max_depth": max_depth,
            "include_subdomains": include_subdomains,
            "crawl_interval": crawl_interval,
            "use_llm": use_llm,
            "max_concurrent_tasks": max_concurrent_tasks,
            "additional_urls": urls[1:] if len(urls) > 1 else []  # Store additional URLs
        }
        
        # Add LLM config if enabled
        if use_llm:
            if not llm_provider:
                raise HTTPException(status_code=400, detail="LLM provider must be specified when use_llm is enabled")
            
            llm_config = {
                "provider": llm_provider
            }
            
            if llm_provider == "openai":
                if not llm_api_key:
                    raise HTTPException(status_code=400, detail="API key is required for OpenAI")
                
                llm_config["openai_config"] = {
                    "api_key": llm_api_key,
                    "model": llm_model or "gpt-3.5-turbo"
                }
                
            elif llm_provider == "ollama":
                llm_config["ollama_config"] = {
                    "base_url": llm_base_url or "http://localhost:11434",
                    "model": llm_model or "llama2"
                }
                
            elif llm_provider == "custom":
                if not llm_base_url:
                    raise HTTPException(status_code=400, detail="Base URL is required for custom LLM provider")
                
                llm_config["custom_model_config"] = {
                    "url": llm_base_url,
                    "api_key": llm_api_key
                }
            
            config["llm_config"] = llm_config
        
        # Convert to CrawlerConfig object and start crawler
        from app.api.models.schemas import CrawlerConfig
        crawler_config = CrawlerConfig(**config)
        
        service = CrawlerService.get_instance()
        status = await service.start_crawler(crawler_config.dict())
        return status
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start crawler with file: {str(e)}")


@router.get("/advanced-config",
            response_model=AdvancedCrawlerConfig,
            summary="Get crawler advanced configuration",
            description="Returns the current advanced crawler configuration.",
            response_description="Current advanced crawler configuration"
            )
async def get_advanced_config(db=Depends(get_async_db)):
    """Get advanced crawler configuration"""
    try:
        config = await db.system_config.find_one({"type": "advanced_crawler_config"})
        
        if not config:
            # Return default config
            return AdvancedCrawlerConfig()
        
        # Remove internal fields
        if "_id" in config:
            del config["_id"]
        if "type" in config:
            del config["type"]
        
        return config
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get advanced configuration: {str(e)}")


@router.post("/advanced-config",
             response_model=AdvancedCrawlerConfig,
             summary="Save crawler advanced configuration",
             description="Saves advanced crawler configuration.",
             response_description="Saved advanced crawler configuration"
             )
async def save_advanced_config(config: AdvancedCrawlerConfig, db=Depends(get_async_db)):
    """Save advanced crawler configuration"""
    try:
        # Add metadata
        config_dict = config.dict()
        config_dict["type"] = "advanced_crawler_config"
        config_dict["updated_at"] = datetime.now()
        
        # Upsert config
        await db.system_config.update_one(
            {"type": "advanced_crawler_config"},
            {"$set": config_dict},
            upsert=True
        )
        
        return config
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save advanced configuration: {str(e)}")