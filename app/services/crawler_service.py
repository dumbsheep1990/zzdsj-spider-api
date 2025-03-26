import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional

from crawl4ai import BrowserConfig, AsyncWebCrawler, CrawlerRunConfig, CacheMode

from crawler.core.crawler import GovWebsiteCrawler
from crawler.core.extraction import ContentExtractor
from db.connection import get_async_db
from config.logging_config import setup_logging

logger = setup_logging("CrawlerService")


class CrawlerService:
    _instance = None
    _crawler = None
    _task = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = CrawlerService()
        return cls._instance

    @property
    def crawler(self):
        return self._crawler

    @property
    def task(self):
        return self._task

    async def start_crawler(self, config: Dict[str, Any]):
        """Start the crawler with the given configuration"""
        if self._crawler and self._crawler.get_status()["status"] == "running":
            raise ValueError("Crawler is already running")

        # Extract LLM configuration
        use_llm = config.get("use_llm", False)
        llm_config = config.get("llm_config") if use_llm else None

        # Extract concurrency configuration
        max_concurrent_tasks = config.get("max_concurrent_tasks", 10)

        # Create new crawler instance
        self._crawler = GovWebsiteCrawler(
            base_url=config["base_url"],
            use_llm=use_llm,
            llm_config=llm_config,
            max_concurrent_tasks=max_concurrent_tasks
        )

        # Start crawler in background
        async def run_crawler():
            try:
                await self._crawler.start_crawling()
            except Exception as e:
                logger.error(f"Crawler error: {str(e)}")

        self._task = asyncio.create_task(run_crawler())
        return self._crawler.get_status()

    def stop_crawler(self):
        """Stop the running crawler"""
        if not self._crawler or self._crawler.get_status()["status"] != "running":
            raise ValueError("No crawler is running")

        # Set status to stopped
        self._crawler.status["status"] = "stopped"
        self._crawler.status["end_time"] = datetime.now().isoformat()

        # Cancel the task if possible
        if self._task and not self._task.done():
            self._task.cancel()

        return self._crawler.get_status()

    def get_status(self) -> Dict[str, Any]:
        """Get current crawler status"""
        if not self._crawler:
            return {
                "status": "not_started",
                "visited_urls": 0,
                "articles_found": 0,
                "subdomains_found": 0
            }

        return self._crawler.get_status()

    async def get_stats(self) -> Optional[Dict[str, Any]]:
        """Get crawler statistics"""
        db = await get_async_db()
        stats = await db.stats.find_one({}, sort=[("start_time", -1)])

        if not stats:
            return None

        # Convert ObjectId to string and format dates
        stats["_id"] = str(stats["_id"])
        stats["start_time"] = stats["start_time"].isoformat()
        stats["end_time"] = stats["end_time"].isoformat()

        return stats

    async def get_crawl_configs(self) -> List[Dict[str, Any]]:
        """Get previous crawler configurations"""
        db = await get_async_db()
        cursor = db.crawler_configs.find().sort("start_time", -1).limit(10)
        configs = await cursor.to_list(length=10)

        # Process results
        result = []
        for config in configs:
            # Convert ObjectId to string
            config["_id"] = str(config["_id"])

            # Format dates
            if "start_time" in config:
                config["start_time"] = config["start_time"].isoformat()

            result.append(config)

        return result


async def restart_crawler(self, config_id: str) -> Dict[str, Any]:
    """Restart crawler with a saved configuration"""
    try:
        from bson.objectid import ObjectId

        db = await get_async_db()

        # Get the saved configuration
        config = await db.crawler_configs.find_one({"_id": ObjectId(config_id)})

        if not config:
            raise ValueError(f"Configuration not found with ID: {config_id}")

        # Remove _id from config for reuse
        config.pop("_id", None)

        # Start crawler with the loaded configuration
        status = await self.start_crawler(config)

        return status

    except Exception as e:
        logger.error(f"Error restarting crawler with config {config_id}: {str(e)}")
        raise


async def crawl_subdomain(self, subdomain_id: str) -> Dict[str, Any]:
    """Crawl a specific subdomain"""
    try:
        from bson.objectid import ObjectId

        db = await get_async_db()

        # Get the subdomain
        subdomain = await db.urls.find_one({"_id": ObjectId(subdomain_id)})

        if not subdomain:
            raise ValueError(f"Subdomain not found with ID: {subdomain_id}")

        # Check if crawler is already running
        if self._crawler and self._crawler.get_status()["status"] == "running":
            raise ValueError("A crawler is already running")

        # Create a new crawler instance for just this subdomain
        self._crawler = GovWebsiteCrawler(
            base_url=subdomain["url"],
            use_llm=False
        )

        # Start crawler in background
        async def run_crawler():
            try:
                await self._crawler.crawl_site(subdomain["url"])
                self._crawler.status["status"] = "completed"
                self._crawler.status["end_time"] = datetime.now().isoformat()
            except Exception as e:
                logger.error(f"Error crawling subdomain: {str(e)}")
                self._crawler.status["status"] = "error"
                self._crawler.status["error"] = str(e)

        self._task = asyncio.create_task(run_crawler())

        await db.urls.update_one(
            {"_id": ObjectId(subdomain_id)},
            {"$set": {"crawl_status": "in_progress", "crawl_started_at": datetime.now()}}
        )

        return self._crawler.get_status()

    except Exception as e:
        logger.error(f"Error starting subdomain crawl: {str(e)}")
        raise


async def recrawl_url(self, url_id: str) -> Dict[str, Any]:
    """Recrawl a specific URL"""
    try:
        from bson.objectid import ObjectId

        db = await get_async_db()

        # Get the URL
        url_doc = await db.urls.find_one({"_id": ObjectId(url_id)})

        if not url_doc:
            raise ValueError(f"URL not found with ID: {url_id}")

        # Create a temporary crawler instance just for this URL
        crawler = GovWebsiteCrawler(
            base_url=url_doc["url"],
            use_llm=self._crawler.use_llm if self._crawler else False,
            llm_config=self._crawler.llm_config if self._crawler else None
        )

        # Crawl the URL
        result = await crawler.crawl_site(url_doc["url"])

        # Update URL status
        await db.urls.update_one(
            {"_id": ObjectId(url_id)},
            {"$set": {"crawled": True, "crawled_at": datetime.now()}}
        )

        return {"url": url_doc["url"], "crawled_pages": result}

    except Exception as e:
        logger.error(f"Error recrawling URL: {str(e)}")
        raise


async def test_llm_extraction(self, config: Dict[str, Any]) -> Dict[str, Any]:
    """Test LLM extraction with the provided configuration"""
    try:
        url = config.pop("url", None)

        if not url:
            raise ValueError("URL is required for testing LLM extraction")

        # Create LLM strategy from config
        llm_strategy = self._create_llm_strategy(config)

        if not llm_strategy:
            raise ValueError("Failed to create LLM strategy from configuration")

        # Create a temporary extractor
        extractor = ContentExtractor(llm_strategy)

        # Create a temporary crawler to fetch the URL
        browser_config = BrowserConfig(verbose=True)
        async with AsyncWebCrawler(config=browser_config) as crawler:
            run_config = CrawlerRunConfig(
                extract_metadata=True,
                extract_links=True,
                extract_images=True,
                cache_mode=CacheMode.BYPASS
            )

            result = await crawler.arun(
                url=url,
                config=run_config
            )

            # Extract article data
            article_data = await extractor.extract_article_data(url, result)

            # Return extracted data
            return {
                "url": url,
                "extracted": {
                    "title": article_data.get("title", ""),
                    "publish_date": article_data.get("publish_date", ""),
                    "department": article_data.get("department", ""),
                    "content_preview": article_data.get("content", "")[:500] + "..." if len(
                        article_data.get("content", "")) > 500 else article_data.get("content", ""),
                    "attachments": article_data.get("attachments", [])
                },
                "is_llm_extracted": article_data.get("is_llm_extracted", False)
            }

    except Exception as e:
        logger.error(f"Error testing LLM extraction: {str(e)}")
        raise


async def schedule_crawler(self, schedule: Dict[str, Any]) -> str:
    """Schedule a crawler job"""
    try:
        db = await get_async_db()

        # Validate cron expression
        cron = schedule.get("cron")
        if not cron:
            raise ValueError("Cron expression is required")

        # Get config_id and validate
        config_id = schedule.get("config_id")
        if not config_id:
            raise ValueError("Configuration ID is required")

        # Validate that the configuration exists
        from bson.objectid import ObjectId
        config = await db.crawler_configs.find_one({"_id": ObjectId(config_id)})

        if not config:
            raise ValueError(f"Configuration not found with ID: {config_id}")

        # Create schedule document
        schedule_doc = {
            "cron": cron,
            "config_id": config_id,
            "name": schedule.get("name"),
            "enabled": schedule.get("enabled", True),
            "created_at": datetime.now(),
            "next_run": None,  # Will be calculated by the scheduler
            "last_run": None
        }

        # Insert schedule
        result = await db.crawler_schedules.insert_one(schedule_doc)

        return str(result.inserted_id)

    except Exception as e:
        logger.error(f"Error scheduling crawler: {str(e)}")
        raise


async def get_crawler_schedules(self) -> List[Dict[str, Any]]:
    """Get list of scheduled crawler jobs"""
    try:
        db = await get_async_db()

        # Get all schedules
        cursor = db.crawler_schedules.find({})
        schedules = await cursor.to_list(length=100)

        # Process results
        result = []
        for schedule in schedules:
            # Convert ObjectId to string
            schedule["id"] = str(schedule.pop("_id"))

            # Format dates
            if "created_at" in schedule:
                schedule["created_at"] = schedule["created_at"].isoformat()
            if "next_run" in schedule and schedule["next_run"]:
                schedule["next_run"] = schedule["next_run"].isoformat()
            if "last_run" in schedule and schedule["last_run"]:
                schedule["last_run"] = schedule["last_run"].isoformat()

            result.append(schedule)

        return result

    except Exception as e:
        logger.error(f"Error getting crawler schedules: {str(e)}")
        raise


async def delete_crawler_schedule(self, schedule_id: str) -> bool:
    """Delete a scheduled crawler job"""
    try:
        from bson.objectid import ObjectId

        db = await get_async_db()

        # Delete the schedule
        result = await db.crawler_schedules.delete_one({"_id": ObjectId(schedule_id)})

        if result.deleted_count == 0:
            raise ValueError(f"Schedule not found with ID: {schedule_id}")

        return True

    except Exception as e:
        logger.error(f"Error deleting crawler schedule: {str(e)}")
        raise