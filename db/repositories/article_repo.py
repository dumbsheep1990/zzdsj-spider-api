from datetime import datetime
from bson.objectid import ObjectId
from typing import Dict, List, Optional, Any


class ArticleRepository:
    def __init__(self, db):
        self.collection = db.articles

    async def insert_article(self, article_data: Dict[str, Any]) -> str:
        """Insert a new article document"""
        result = await self.collection.insert_one(article_data)
        return str(result.inserted_id)

    async def find_articles(self, query: Dict[str, Any], skip: int = 0, limit: int = 20) -> List[Dict[str, Any]]:
        """Find articles with pagination"""
        cursor = self.collection.find(query).sort("crawled_at", -1).skip(skip).limit(limit)
        return await cursor.to_list(length=limit)

    async def count_articles(self, query: Dict[str, Any]) -> int:
        """Count articles matching query"""
        return await self.collection.count_documents(query)

    async def get_article_by_id(self, article_id: str) -> Optional[Dict[str, Any]]:
        """Get an article by ID"""
        return await self.collection.find_one({"_id": ObjectId(article_id)})

    async def update_article(self, article_id: str, update_data: Dict[str, Any]) -> bool:
        """Update an article document"""
        result = await self.collection.update_one(
            {"_id": ObjectId(article_id)},
            {"$set": update_data}
        )
        return result.modified_count > 0