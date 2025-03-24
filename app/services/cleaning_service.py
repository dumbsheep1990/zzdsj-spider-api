import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any
from crawler.core.extraction import DataCleaner
from db.connection import get_async_db, get_sync_db
from config.logging_config import setup_logging

logger = setup_logging("CleaningService")


class CleaningService:
    @staticmethod
    async def set_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Save cleaning configuration"""
        # Save config to file
        with open("config/cleaning_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        return config

    @staticmethod
    async def get_config() -> Dict[str, Any]:
        """Get current cleaning configuration"""
        try:
            # Read config from file
            with open("config/cleaning_config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            # Default config
            default_config = {
                "remove_html_tags": True,
                "extract_tables": True,
                "extract_attachments": True,
                "extract_images": True,
                "custom_rules": {}
            }
            return default_config

    @staticmethod
    async def run_cleaning():
        """Run data cleaning process in background"""
        try:
            # Get cleaning config
            config = await CleaningService.get_config()

            # Connect to database
            db = await get_async_db()

            # Get all articles
            total_articles = await db.articles.count_documents({})

            logger.info(f"Starting data cleaning for {total_articles} articles")

            # Process articles in batches
            batch_size = 50
            processed = 0

            cursor = db.articles.find({})

            async for article in cursor:
                try:
                    # Clean article based on config
                    cleaned_data = await DataCleaner.clean_article(article, config)

                    # Update article
                    await db.articles.update_one(
                        {"_id": article["_id"]},
                        {"$set": cleaned_data}
                    )

                    processed += 1
                    if processed % 100 == 0:
                        logger.info(f"Processed {processed}/{total_articles} articles")

                except Exception as e:
                    logger.error(f"Error cleaning article {article['_id']}: {str(e)}")

            logger.info(f"Data cleaning completed, processed {processed}/{total_articles} articles")
            return {"success": True, "processed": processed, "total": total_articles}

        except Exception as e:
            logger.error(f"Data cleaning task error: {str(e)}")
            return {"success": False, "error": str(e)}


async def clean_single_article(article_id: str) -> Dict[str, Any]:
    """Clean a single article"""
    try:
        from bson.objectid import ObjectId

        # Connect to database
        db = await get_async_db()

        # Get article
        article = await db.articles.find_one({"_id": ObjectId(article_id)})

        if not article:
            raise ValueError(f"Article not found with ID: {article_id}")

        # Get cleaning config
        config = await CleaningService.get_config()

        # Clean article
        cleaned_data = await DataCleaner.clean_article(article, config)

        # Update article
        result = await db.articles.update_one(
            {"_id": ObjectId(article_id)},
            {"$set": cleaned_data}
        )

        return {
            "article_id": article_id,
            "cleaning_fields": list(cleaned_data.keys())
        }

    except Exception as e:
        logger.error(f"Error cleaning article {article_id}: {str(e)}")
        raise


@staticmethod
async def add_cleaning_rule(rule: Dict[str, str]) -> Dict[str, str]:
    """Add a custom cleaning rule"""
    try:
        # Get current config
        config = await CleaningService.get_config()

        # Initialize custom_rules if not exists
        if "custom_rules" not in config:
            config["custom_rules"] = {}

        # Add or update the rule
        config["custom_rules"].update(rule)

        # Save config
        await CleaningService.set_config(config)

        return config["custom_rules"]

    except Exception as e:
        logger.error(f"Error adding cleaning rule: {str(e)}")
        raise


@staticmethod
async def delete_cleaning_rule(pattern: str) -> Dict[str, str]:
    """Delete a custom cleaning rule"""
    try:
        # Get current config
        config = await CleaningService.get_config()

        # Check if custom_rules exists
        if "custom_rules" not in config or not config["custom_rules"]:
            return {}

        # Remove the rule if exists
        if pattern in config["custom_rules"]:
            del config["custom_rules"][pattern]

        # Save config
        await CleaningService.set_config(config)

        return config["custom_rules"]

    except Exception as e:
        logger.error(f"Error deleting cleaning rule: {str(e)}")
        raise


@staticmethod
async def get_status() -> Dict[str, Any]:
    """Get current cleaning status"""
    try:
        db = await get_async_db()

        # Get the latest cleaning status
        status = await db.cleaning_status.find_one({}, sort=[("started_at", -1)])

        if not status:
            return {
                "status": "not_started",
                "total_articles": 0,
                "processed_articles": 0
            }

        # Format dates
        if "started_at" in status:
            status["started_at"] = status["started_at"].isoformat()
        if "completed_at" in status:
            status["completed_at"] = status["completed_at"].isoformat()

        # Convert ObjectId to string
        status["_id"] = str(status["_id"])

        return status

    except Exception as e:
        logger.error(f"Error getting cleaning status: {str(e)}")
        raise