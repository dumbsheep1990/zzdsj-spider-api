from fastapi import APIRouter, HTTPException, Query, Depends, Response, Path
from typing import Optional, Dict, List, Any
from bson.objectid import ObjectId
from datetime import datetime
import csv
import io
import json
from app.api.models.schemas import ArticleFilter, PaginatedResponse
from db.connection import get_async_db

router = APIRouter(prefix="/articles", tags=["articles"])


@router.get("/",
            response_model=PaginatedResponse,
            summary="Get articles with pagination",
            description="""
    Returns a paginated list of articles with optional filtering.

    - **page**: Page number (starting from 1)
    - **page_size**: Number of items per page
    - **domain**: Filter by domain
    - **start_date**: Filter by publish date (start)
    - **end_date**: Filter by publish date (end)
    - **keyword**: Filter by keyword in title or content
    """,
            response_description="Paginated list of articles"
            )
async def get_articles(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(20, ge=1, le=100, description="Number of items per page"),
        domain: Optional[str] = Query(None, description="Filter by domain"),
        start_date: Optional[str] = Query(None, description="Filter by publish date (start)"),
        end_date: Optional[str] = Query(None, description="Filter by publish date (end)"),
        keyword: Optional[str] = Query(None, description="Filter by keyword in title or content"),
        db=Depends(get_async_db)
):
    """Get articles with pagination and filtering"""
    # Build query conditions
    query = {}

    if domain:
        query["domain"] = domain

    date_query = {}
    if start_date:
        date_query["$gte"] = datetime.fromisoformat(start_date)
    if end_date:
        date_query["$lte"] = datetime.fromisoformat(end_date)

    if date_query:
        query["publish_date"] = date_query

    if keyword:
        query["$or"] = [
            {"title": {"$regex": keyword, "$options": "i"}},
            {"content": {"$regex": keyword, "$options": "i"}}
        ]

    # Pagination
    total = await db.articles.count_documents(query)
    skip = (page - 1) * page_size

    cursor = db.articles.find(query).sort("crawled_at", -1).skip(skip).limit(page_size)
    articles = await cursor.to_list(length=page_size)

    # Process results
    result = []
    for article in articles:
        # Convert ObjectId to string
        article["_id"] = str(article["_id"])

        # Format dates
        if "crawled_at" in article:
            article["crawled_at"] = article["crawled_at"].isoformat()
        if "publish_date" in article and isinstance(article["publish_date"], datetime):
            article["publish_date"] = article["publish_date"].isoformat()

        # Remove large fields for list view
        if "raw_html" in article:
            del article["raw_html"]
        if "content" in article and len(article["content"]) > 500:
            article["content"] = article["content"][:500] + "..."

        result.append(article)

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "data": result
    }


@router.get("/{article_id}",
            summary="Get article details",
            description="Returns detailed information for a specific article by ID.",
            response_description="Article details"
            )
async def get_article_detail(
        article_id: str = Path(..., description="Article ID"),
        db=Depends(get_async_db)
):
    """Get detailed information for a specific article"""
    try:
        article = await db.articles.find_one({"_id": ObjectId(article_id)})

        if not article:
            raise HTTPException(status_code=404, detail="Article not found")

        # Convert ObjectId to string
        article["_id"] = str(article["_id"])

        # Format dates
        if "crawled_at" in article:
            article["crawled_at"] = article["crawled_at"].isoformat()
        if "publish_date" in article and isinstance(article["publish_date"], datetime):
            article["publish_date"] = article["publish_date"].isoformat()
        if "cleaned_at" in article and isinstance(article["cleaned_at"], datetime):
            article["cleaned_at"] = article["cleaned_at"].isoformat()

        return article

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve article: {str(e)}")


@router.get("/export/{format}",
            summary="Export articles",
            description="""
    Exports articles in various formats (json, csv, excel).

    - **format**: Export format (json, csv, excel)
    - **domain**: Filter by domain
    - **start_date**: Filter by publish date (start)
    - **end_date**: Filter by publish date (end)
    - **keyword**: Filter by keyword in title or content
    """,
            response_description="Exported articles file"
            )
async def export_articles(
        format: str = Path(..., regex="^(json|csv|excel)$", description="Export format (json, csv, excel)"),
        domain: Optional[str] = Query(None, description="Filter by domain"),
        start_date: Optional[str] = Query(None, description="Filter by publish date (start)"),
        end_date: Optional[str] = Query(None, description="Filter by publish date (end)"),
        keyword: Optional[str] = Query(None, description="Filter by keyword in title or content"),
        db=Depends(get_async_db)
):
    """Export articles in various formats"""
    if format not in ["json", "csv", "excel"]:
        raise HTTPException(status_code=400, detail="Unsupported export format")

    # Build query
    query = {}
    if domain:
        query["domain"] = domain

    date_query = {}
    if start_date:
        date_query["$gte"] = datetime.fromisoformat(start_date)
    if end_date:
        date_query["$lte"] = datetime.fromisoformat(end_date)

    if date_query:
        query["publish_date"] = date_query

    if keyword:
        query["$or"] = [
            {"title": {"$regex": keyword, "$options": "i"}},
            {"content": {"$regex": keyword, "$options": "i"}}
        ]

    # Get articles
    cursor = db.articles.find(query).sort("crawled_at", -1)
    articles = await cursor.to_list(length=1000)  # Limit to 1000 for export

    # Process articles
    for article in articles:
        article["_id"] = str(article["_id"])
        if "crawled_at" in article:
            article["crawled_at"] = article["crawled_at"].isoformat()
        if "publish_date" in article and isinstance(article["publish_date"], datetime):
            article["publish_date"] = article["publish_date"].isoformat()
        if "raw_html" in article:
            del article["raw_html"]  # Remove raw HTML for export

    # Export based on format
    if format == "json":
        content = json.dumps(articles, ensure_ascii=False, indent=2)
        media_type = "application/json"
        filename = f"articles_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    elif format == "csv":
        output = io.StringIO()
        if articles:
            fieldnames = ["title", "publish_date", "department", "content", "url", "domain"]
            writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for article in articles:
                writer.writerow(article)

        content = output.getvalue()
        media_type = "text/csv"
        filename = f"articles_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    elif format == "excel":
        import pandas as pd
        from io import BytesIO

        df = pd.DataFrame(articles)
        output = BytesIO()
        df.to_excel(output, index=False)
        content = output.getvalue()
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        filename = f"articles_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

    # Return response with appropriate content type
    headers = {
        "Content-Disposition": f"attachment; filename={filename}"
    }
    return Response(content=content, media_type=media_type, headers=headers)


@router.delete("/{article_id}",
               summary="Delete article",
               description="Deletes a specific article by ID.",
               response_description="Deletion confirmation"
               )
async def delete_article(
        article_id: str = Path(..., description="Article ID"),
        db=Depends(get_async_db)
):
    """Delete an article by ID"""
    try:
        result = await db.articles.delete_one({"_id": ObjectId(article_id)})

        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Article not found")

        return {"message": "Article deleted successfully", "article_id": article_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete article: {str(e)}")


@router.get("/stats/summary",
            summary="Get article statistics",
            description="Returns summary statistics about the articles in the database.",
            response_description="Article statistics summary"
            )
async def get_article_stats(db=Depends(get_async_db)):
    """Get summary statistics about articles"""
    try:
        # Count total articles
        total_articles = await db.articles.count_documents({})

        # Count articles by domain
        pipeline = [
            {"$group": {"_id": "$domain", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        cursor = db.articles.aggregate(pipeline)
        domains = await cursor.to_list(length=100)

        # Count articles by publish date (monthly)
        pipeline = [
            {"$match": {"publish_date": {"$exists": True}}},
            {"$project": {
                "year_month": {
                    "$dateToString": {
                        "format": "%Y-%m",
                        "date": {"$toDate": "$publish_date"}
                    }
                }
            }},
            {"$group": {"_id": "$year_month", "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}}
        ]
        cursor = db.articles.aggregate(pipeline)
        timeline = await cursor.to_list(length=100)

        return {
            "total_articles": total_articles,
            "domains": domains,
            "timeline": timeline
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get article statistics: {str(e)}")