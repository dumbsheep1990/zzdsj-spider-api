from fastapi import APIRouter
from app.api.routes import crawler, articles, cleaning, ai, export, llm, vector, system, config, llm_service

api_router = APIRouter(prefix="/api")
api_router.include_router(crawler.router)
api_router.include_router(articles.router)
api_router.include_router(cleaning.router)
api_router.include_router(ai.router)
api_router.include_router(export.router)
api_router.include_router(llm.router)
api_router.include_router(vector.router)
api_router.include_router(system.router)
api_router.include_router(config.router)
api_router.include_router(llm_service.router)