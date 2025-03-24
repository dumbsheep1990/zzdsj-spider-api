from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query, Path, Body
from app.api.models.schemas import CrawlerConfig, CrawlerStatus, LLMModelInfo
from app.services.crawler_service import CrawlerService
from db.connection import get_async_db
from typing import List, Dict, Any, Optional
from datetime import datetime

router = APIRouter(prefix="/crawler", tags=["crawler"])


@router.post("/start",
             response_model=CrawlerStatus,
             summary="Start a new crawler job",
             description="""
    Starts a new crawler job with the specified configuration.

    The crawler will begin by discovering subdomains, then crawl the main site and any discovered subdomains.
    Content will be extracted and stored in the database.

    If LLM extraction is enabled, the specified LLM provider will be used to extract structured content.
    """,
             response_description="Current status of the crawler after starting"
             )
async def start_crawler(config: CrawlerConfig, background_tasks: BackgroundTasks):
    """Start the crawler with the given configuration"""
    try:
        service = CrawlerService.get_instance()
        status = await service.start_crawler(config.dict())
        return status
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start crawler: {str(e)}")


@router.get("/status",
            response_model=CrawlerStatus,
            summary="Get crawler status",
            description="Returns the current status of the crawler, including progress statistics.",
            response_description="Current status of the crawler"
            )
async def get_crawler_status():
    """Get current crawler status"""
    service = CrawlerService.get_instance()
    return service.get_status()


@router.post("/stop",
             response_model=CrawlerStatus,
             summary="Stop the running crawler",
             description="Stops the currently running crawler job.",
             response_description="Final status of the crawler after stopping"
             )
async def stop_crawler():
    """Stop the running crawler"""
    try:
        service = CrawlerService.get_instance()
        status = service.stop_crawler()
        return status
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop crawler: {str(e)}")


@router.get("/stats",
            summary="Get crawler statistics",
            description="Returns statistics from the most recent crawler job, including concurrency information.",
            response_description="Statistics from the most recent crawler job"
            )
async def get_crawler_stats():
    """Get crawler statistics"""
    service = CrawlerService.get_instance()
    stats = await service.get_stats()

    if not stats:
        return {"message": "No crawler statistics available"}

    # Add current concurrency information if crawler is running
    if service.crawler:
        stats["current_concurrency"] = {
            "max_concurrent_tasks": service.crawler.max_concurrent_tasks,
            "active_tasks": service.crawler.status.get("active_tasks", 0)
        }

    return stats


@router.get("/llm-models",
            response_model=LLMModelInfo,
            summary="Get available LLM models",
            description="Returns information about available LLM models for content extraction.",
            response_description="Available LLM providers and models"
            )
async def get_llm_models():
    """Get available LLM models for configuration"""
    return {
        "providers": [
            {
                "name": "openai",
                "display_name": "OpenAI",
                "models": [
                    {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo"},
                    {"id": "gpt-4", "name": "GPT-4"},
                    {"id": "gpt-4-turbo", "name": "GPT-4 Turbo"}
                ],
                "requires_api_key": True
            },
            {
                "name": "ollama",
                "display_name": "Ollama",
                "models": [
                    {"id": "llama2", "name": "Llama 2"},
                    {"id": "mistral", "name": "Mistral"},
                    {"id": "mistral-openorca", "name": "Mistral OpenOrca"},
                    {"id": "phi", "name": "Phi"},
                    {"id": "vicuna", "name": "Vicuna"},
                    {"id": "mixtral", "name": "Mixtral 8x7B"}
                ],
                "requires_api_key": False,
                "requires_base_url": True,
                "default_base_url": "http://localhost:11434"
            },
            {
                "name": "custom",
                "display_name": "Custom Model",
                "requires_url": True,
                "requires_api_key": False,
                "supports_headers": True
            }
        ]
    }


@router.get("/configs",
            summary="Get previous crawler configurations",
            description="Returns a list of previously used crawler configurations.",
            response_description="List of previous crawler configurations"
            )
async def get_crawler_configs():
    """Get previous crawler configurations"""
    service = CrawlerService.get_instance()
    configs = await service.get_crawl_configs()
    return configs


@router.post("/restart/{config_id}",
             response_model=CrawlerStatus,
             summary="Restart crawler with saved configuration",
             description="Restarts the crawler using a previously saved configuration.",
             response_description="Current status of the crawler after restarting"
             )
async def restart_crawler(
        config_id: str = Path(..., description="Configuration ID"),
        background_tasks: BackgroundTasks = None
):
    """Restart crawler with a saved configuration"""
    try:
        service = CrawlerService.get_instance()
        status = await service.restart_crawler(config_id)
        return status
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restart crawler: {str(e)}")


@router.get("/subdomains",
            summary="Get discovered subdomains",
            description="Returns a list of subdomains discovered during crawling.",
            response_description="List of discovered subdomains"
            )
async def get_subdomains(
        crawled: Optional[bool] = Query(None, description="Filter by crawled status"),
        db=Depends(get_async_db)
):
    """Get all discovered subdomains"""
    query = {"type": "subdomain"}

    if crawled is not None:
        query["crawled"] = crawled

    cursor = db.urls.find(query)
    subdomains = await cursor.to_list(length=100)

    # Convert ObjectId to string and format dates
    for subdomain in subdomains:
        subdomain["_id"] = str(subdomain["_id"])
        if "discovered_at" in subdomain:
            subdomain["discovered_at"] = subdomain["discovered_at"].isoformat()
        if "crawled_at" in subdomain:
            subdomain["crawled_at"] = subdomain["crawled_at"].isoformat()

    return subdomains


@router.post("/subdomain/{subdomain_id}/crawl",
             summary="Crawl specific subdomain",
             description="Initiates crawling for a specific subdomain.",
             response_description="Crawling task status"
             )
async def crawl_subdomain(
        subdomain_id: str = Path(..., description="Subdomain ID"),
        background_tasks: BackgroundTasks = None
):
    """Crawl a specific subdomain"""
    try:
        service = CrawlerService.get_instance()
        result = await service.crawl_subdomain(subdomain_id)
        return {"message": "Subdomain crawling started", "subdomain_id": subdomain_id, "status": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start subdomain crawling: {str(e)}")


@router.post("/url/{url_id}/recrawl",
             summary="Recrawl specific URL",
             description="Recrawls a specific URL that was previously crawled.",
             response_description="Recrawling task status"
             )
async def recrawl_url(
        url_id: str = Path(..., description="URL ID"),
        background_tasks: BackgroundTasks = None
):
    """Recrawl a specific URL"""
    try:
        service = CrawlerService.get_instance()
        result = await service.recrawl_url(url_id)
        return {"message": "URL recrawling started", "url_id": url_id, "status": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start URL recrawling: {str(e)}")


@router.get("/history",
            summary="Get crawler history",
            description="Returns the history of crawler runs.",
            response_description="Crawler run history"
            )
async def get_crawler_history(
        limit: int = Query(10, ge=1, le=100, description="Maximum number of records to return"),
        db=Depends(get_async_db)
):
    """Get history of crawler runs"""
    try:
        cursor = db.stats.find({}).sort("start_time", -1).limit(limit)
        history = await cursor.to_list(length=limit)

        # Process results
        result = []
        for item in history:
            item["_id"] = str(item["_id"])
            item["start_time"] = item["start_time"].isoformat()
            item["end_time"] = item["end_time"].isoformat()
            result.append(item)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get crawler history: {str(e)}")


@router.get("/urls",
            summary="Get crawled URLs",
            description="Returns a list of URLs that have been crawled.",
            response_description="List of crawled URLs"
            )
async def get_crawled_urls(
        crawled: bool = Query(True, description="Filter by crawled status"),
        limit: int = Query(100, ge=1, le=1000, description="Maximum number of URLs to return"),
        domain: Optional[str] = Query(None, description="Filter by domain"),
        db=Depends(get_async_db)
):
    """Get list of crawled URLs"""
    try:
        query = {"crawled": crawled}

        if domain:
            query["url"] = {"$regex": domain, "$options": "i"}

        cursor = db.urls.find(query).sort("crawled_at" if crawled else "discovered_at", -1).limit(limit)
        urls = await cursor.to_list(length=limit)

        # Process results
        result = []
        for url in urls:
            url["_id"] = str(url["_id"])
            if "discovered_at" in url:
                url["discovered_at"] = url["discovered_at"].isoformat()
            if "crawled_at" in url:
                url["crawled_at"] = url["crawled_at"].isoformat()
            result.append(url)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get URLs: {str(e)}")


@router.delete("/url/{url_id}",
               summary="Delete URL",
               description="Deletes a URL from the database.",
               response_description="Deletion confirmation"
               )
async def delete_url(
        url_id: str = Path(..., description="URL ID"),
        db=Depends(get_async_db)
):
    """Delete a URL from the database"""
    try:
        from bson.objectid import ObjectId

        result = await db.urls.delete_one({"_id": ObjectId(url_id)})

        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="URL not found")

        return {"message": "URL deleted successfully", "url_id": url_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete URL: {str(e)}")


@router.post("/test-llm",
             summary="Test LLM extraction",
             description="Tests LLM extraction with the provided configuration on a sample URL.",
             response_description="LLM extraction test result"
             )
async def test_llm_extraction(
        config: Dict[str, Any] = Body(
            ...,
            example={
                "provider": "openai",
                "openai_config": {
                    "api_key": "sk-1234567890abcdef",
                    "model": "gpt-3.5-turbo"
                },
                "url": "https://www.example.gov.cn/news/sample.html"
            },
            description="LLM configuration and test URL"
        )
):
    """Test LLM extraction with the provided configuration"""
    try:
        service = CrawlerService.get_instance()
        result = await service.test_llm_extraction(config)
        return {"message": "LLM extraction test completed", "result": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to test LLM extraction: {str(e)}")


@router.post("/schedule",
             summary="Schedule crawler",
             description="Schedules a crawler job to run at specified intervals.",
             response_description="Crawler scheduling result"
             )
async def schedule_crawler(
        schedule: Dict[str, Any] = Body(
            ...,
            example={
                "cron": "0 0 * * *",  # Daily at midnight
                "config_id": "6072f1b12c723a8c9d89a123"  # ID of a saved configuration
            },
            description="Crawler schedule configuration"
        )
):
    """Schedule a crawler job"""
    try:
        service = CrawlerService.get_instance()
        result = await service.schedule_crawler(schedule)
        return {"message": "Crawler scheduled successfully", "schedule_id": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to schedule crawler: {str(e)}")


@router.get("/schedules",
            summary="Get crawler schedules",
            description="Returns a list of scheduled crawler jobs.",
            response_description="List of crawler schedules"
            )
async def get_crawler_schedules():
    """Get list of scheduled crawler jobs"""
    try:
        service = CrawlerService.get_instance()
        schedules = await service.get_crawler_schedules()
        return schedules
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get crawler schedules: {str(e)}")


@router.delete("/schedule/{schedule_id}",
               summary="Delete crawler schedule",
               description="Deletes a scheduled crawler job.",
               response_description="Deletion confirmation"
               )
async def delete_crawler_schedule(
        schedule_id: str = Path(..., description="Schedule ID")
):
    """Delete a scheduled crawler job"""
    try:
        service = CrawlerService.get_instance()
        result = await service.delete_crawler_schedule(schedule_id)
        return {"message": "Crawler schedule deleted successfully", "schedule_id": schedule_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete crawler schedule: {str(e)}")


@router.post("/concurrency",
             summary="Update crawler concurrency",
             description="Updates the maximum number of concurrent tasks for the running crawler.",
             response_description="Updated crawler status"
             )
async def update_concurrency(
        max_tasks: int = Body(..., ge=1, le=100, description="Maximum number of concurrent tasks", example=20)
):
    """Update the maximum number of concurrent tasks for the running crawler"""
    try:
        service = CrawlerService.get_instance()

        if not service.crawler:
            raise HTTPException(status_code=400, detail="No crawler instance found")

        # Update concurrency setting
        service.crawler.max_concurrent_tasks = max_tasks
        status = service.get_status()

        return {
            "message": f"Crawler concurrency updated to {max_tasks} tasks",
            "status": status
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update crawler concurrency: {str(e)}")

@router.get("/concurrency/presets",
    summary="Get concurrency presets",
    description="Returns recommended concurrency presets for different scenarios.",
    response_description="List of concurrency presets"
)
async def get_concurrency_presets():
    """Get recommended concurrency presets"""
    return {
        "presets": [
            {
                "name": "Low Impact",
                "max_concurrent_tasks": 5,
                "description": "Gentle crawling with minimal impact on target servers"
            },
            {
                "name": "Balanced",
                "max_concurrent_tasks": 10,
                "description": "Balanced performance and server impact"
            },
            {
                "name": "Performance",
                "max_concurrent_tasks": 25,
                "description": "High performance crawling for robust servers"
            },
            {
                "name": "Maximum",
                "max_concurrent_tasks": 50,
                "description": "Maximum crawling speed for very robust servers (use with caution)"
            }
        ]
    }