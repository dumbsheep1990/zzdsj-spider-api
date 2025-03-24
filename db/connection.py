from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from config.settings import MONGODB_URI, DB_NAME

# Async MongoDB client for FastAPI
async def get_async_db():
    client = AsyncIOMotorClient(MONGODB_URI)
    db = client[DB_NAME]
    return db

# Sync MongoDB client for crawler
def get_sync_db():
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    return db, client