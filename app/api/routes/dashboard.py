from fastapi import APIRouter, HTTPException, Depends
from db.connection import get_async_db
from datetime import datetime, timedelta
from typing import List, Dict, Any

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/summary",
            summary="Get dashboard summary",
            description="Returns summary statistics for the dashboard.",
            response_description="Dashboard summary data"
            )
async def get_dashboard_summary(db=Depends(get_async_db)):
    """Get summary statistics for the dashboard"""
    try:
        # Get total counts
        total_articles = await db.articles.count_documents({})
        total_subdomains = await db.urls.count_documents({"type": "subdomain"})
        total_crawled_urls = await db.urls.count_documents({"crawled": True})

        # Get recent articles
        recent_articles_cursor = db.articles.find({}).sort("crawled_at", -1).limit(5)
        recent_articles = await recent_articles_cursor.to_list(length=5)

        # Process recent articles
        for article in recent_articles:
            article["_id"] = str(article["_id"])
            if "crawled_at" in article:
                article["crawled_at"] = article["crawled_at"].isoformat()
            if "publish_date" in article and isinstance(article["publish_date"], datetime):
                article["publish_date"] = article["publish_date"].isoformat()
            if "raw_html" in article:
                del article["raw_html"]
            if "content" in article and len(article["content"]) > 200:
                article["content"] = article["content"][:200] + "..."

        # Get latest crawler run
        latest_run = await db.stats.find_one({}, sort=[("start_time", -1)])

        if latest_run:
            latest_run["_id"] = str(latest_run["_id"])
            latest_run["start_time"] = latest_run["start_time"].isoformat()
            latest_run["end_time"] = latest_run["end_time"].isoformat()

        return {
            "total_articles": total_articles,
            "total_subdomains": total_subdomains,
            "total_crawled_urls": total_crawled_urls,
            "recent_articles": recent_articles,
            "latest_run": latest_run
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard summary: {str(e)}")


@router.get("/stats/articles",
            summary="Get article statistics",
            description="Returns articles statistics grouped by domain, date, or other criteria.",
            response_description="Article statistics"
            )
async def get_article_stats(
        group_by: str = "domain",
        db=Depends(get_async_db)
):
    """Get article statistics grouped by various criteria"""
    try:
        if group_by == "domain":
            # Group by domain
            pipeline = [
                {"$group": {"_id": "$domain", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
        elif group_by == "date":
            # Group by month
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
        elif group_by == "department":
            # Group by department
            pipeline = [
                {"$match": {"department": {"$exists": True}}},
                {"$group": {"_id": "$department", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
        else:
            raise HTTPException(status_code=400, detail=f"Invalid group_by parameter: {group_by}")

        cursor = db.articles.aggregate(pipeline)
        stats = await cursor.to_list(length=100)

        return {"group_by": group_by, "data": stats}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get article statistics: {str(e)}")


@router.get("/timeline",
            summary="Get crawler timeline",
            description="Returns crawler activity timeline.",
            response_description="Crawler timeline data"
            )
async def get_crawler_timeline(
        days: int = 30,
        db=Depends(get_async_db)
):
    """Get crawler activity timeline"""
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Get crawler runs in date range
        cursor = db.stats.find(
            {"start_time": {"$gte": start_date, "$lte": end_date}}
        ).sort("start_time", 1)

        runs = await cursor.to_list(length=100)

        # Process runs
        timeline = []
        for run in runs:
            timeline.append({
                "id": str(run["_id"]),
                "start_time": run["start_time"].isoformat(),
                "end_time": run["end_time"].isoformat(),
                "duration_seconds": run["duration_seconds"],
                "visited_urls": run["visited_urls"],
                "articles_found": run["articles_found"],
                "subdomains_found": run["subdomains_found"]
            })

        return {"days": days, "timeline": timeline}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get crawler timeline: {str(e)}")


@router.get("/subdomains",
            summary="Get subdomain statistics",
            description="Returns statistics about discovered subdomains.",
            response_description="Subdomain statistics"
            )
async def get_subdomain_stats(db=Depends(get_async_db)):
    """Get statistics about discovered subdomains"""
    try:
        # Get subdomain counts
        total_subdomains = await db.urls.count_documents({"type": "subdomain"})
        crawled_subdomains = await db.urls.count_documents({"type": "subdomain", "crawled": True})

        # Get top subdomains by articles
        pipeline = [
            {"$group": {"_id": "$domain", "count": {"$sum": 1}}},
            {"$match": {"_id": {"$ne": None}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]

        cursor = db.articles.aggregate(pipeline)
        top_subdomains = await cursor.to_list(length=10)

        return {
            "total_subdomains": total_subdomains,
            "crawled_subdomains": crawled_subdomains,
            "crawled_percentage": (crawled_subdomains / total_subdomains * 100) if total_subdomains > 0 else 0,
            "top_subdomains": top_subdomains
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get subdomain statistics: {str(e)}")