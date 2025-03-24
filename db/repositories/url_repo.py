from datetime import datetime
from typing import Dict, List, Optional, Any


class UrlRepository:
    def __init__(self, db):
        self.collection = db.urls

    async def insert_url(self, url_data: Dict[str, Any]) -> str:
        """Insert a new URL document"""
        result = await self.collection.insert_one(url_data)
        return str(result.inserted_id)

    async def update_url(self, url: str, update_data: Dict[str, Any]) -> bool:
        """Update a URL document"""
        result = await self.collection.update_one(
            {"url": url},
            {"$set": update_data}
        )
        return result.modified_count > 0

    async def upsert_url(self, url: str, url_data: Dict[str, Any]) -> None:
        """Insert or update URL document"""
        await self.collection.update_one(
            {"url": url},
            {"$set": url_data},
            upsert=True
        )

    async def get_subdomains(self) -> List[Dict[str, Any]]:
        """Get all subdomains"""
        cursor = self.collection.find({"type": "subdomain"})
        return await cursor.to_list(length=100)

    async def get_uncrawled_urls(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get URLs that haven't been crawled yet"""
        cursor = self.collection.find({"crawled": False}).limit(limit)
        return await cursor.to_list(length=limit)