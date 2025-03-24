from fastapi import APIRouter
from app.api.routes import crawler, articles, cleaning

api_router = APIRouter(prefix="/api")
api_router.include_router(crawler.router)
api_router.include_router(articles.router)
api_router.include_router(cleaning.router)