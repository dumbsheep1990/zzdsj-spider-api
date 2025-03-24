from fastapi import APIRouter, HTTPException, BackgroundTasks, Path, Body
from app.api.models.schemas import CleaningConfig
from app.services.cleaning_service import CleaningService
from app.services.crawler_service import CrawlerService
from typing import Dict, List, Any

router = APIRouter(prefix="/cleaning", tags=["cleaning"])


@router.post("/config",
             summary="Set cleaning configuration",
             description="Saves data cleaning configuration for future cleaning operations.",
             response_description="Saved cleaning configuration"
             )
async def set_cleaning_config(config: CleaningConfig):
    """Set data cleaning configuration"""
    try:
        saved_config = await CleaningService.set_config(config.dict())
        return {"message": "Data cleaning configuration saved", "config": saved_config}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save configuration: {str(e)}")


@router.get("/config",
            summary="Get cleaning configuration",
            description="Returns the current data cleaning configuration.",
            response_description="Current cleaning configuration"
            )
async def get_cleaning_config():
    """Get current data cleaning configuration"""
    try:
        config = await CleaningService.get_config()
        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get configuration: {str(e)}")


@router.post("/run",
             summary="Run data cleaning",
             description="""
    Starts a data cleaning process based on the current configuration.

    This is a background task that processes all articles in the database.
    Cleaning operations include:
    - Removing HTML tags
    - Extracting tables
    - Applying custom rules
    """,
             response_description="Data cleaning task status"
             )
async def run_data_cleaning(background_tasks: BackgroundTasks):
    """Start data cleaning process"""
    # Check if crawler is running
    crawler_service = CrawlerService.get_instance()
    if crawler_service.crawler and crawler_service.crawler.get_status()["status"] == "running":
        raise HTTPException(
            status_code=400,
            detail="Crawler is currently running. Please wait for crawler to complete before running data cleaning."
        )

    # Start cleaning task in background
    background_tasks.add_task(CleaningService.run_cleaning)
    return {"message": "Data cleaning task started"}


@router.post("/article/{article_id}",
             summary="Clean single article",
             description="Applies the current cleaning configuration to a specific article.",
             response_description="Cleaning result"
             )
async def clean_single_article(
        article_id: str = Path(..., description="Article ID"),
        background_tasks: BackgroundTasks = None
):
    """Clean a single article"""
    try:
        result = await CleaningService.clean_single_article(article_id)
        return {"message": "Article cleaning completed", "article_id": article_id, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clean article: {str(e)}")


@router.post("/rules",
             summary="Add custom cleaning rule",
             description="Adds a new custom regex cleaning rule.",
             response_description="Updated cleaning rules"
             )
async def add_cleaning_rule(
        rule: Dict[str, str] = Body(..., example={r"\s+": " "}, description="Regex pattern and replacement")
):
    """Add a custom cleaning rule"""
    try:
        result = await CleaningService.add_cleaning_rule(rule)
        return {"message": "Cleaning rule added successfully", "rules": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add cleaning rule: {str(e)}")


@router.delete("/rules/{pattern}",
               summary="Delete cleaning rule",
               description="Deletes a custom regex cleaning rule by pattern.",
               response_description="Updated cleaning rules"
               )
async def delete_cleaning_rule(
        pattern: str = Path(..., description="Regex pattern to delete")
):
    """Delete a custom cleaning rule"""
    try:
        result = await CleaningService.delete_cleaning_rule(pattern)
        return {"message": "Cleaning rule deleted successfully", "rules": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete cleaning rule: {str(e)}")


@router.get("/status",
            summary="Get cleaning status",
            description="Returns the status of the current cleaning process.",
            response_description="Current cleaning status"
            )
async def get_cleaning_status():
    """Get current cleaning status"""
    try:
        status = await CleaningService.get_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cleaning status: {str(e)}")