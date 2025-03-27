from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from app.api.routes import api_router
from config.logging_config import setup_logging
from config.database import get_app_config
from db.connection import close_all_connections

# Set up logger
logger = setup_logging("API")

# 加载应用配置
app_config = get_app_config()

# Create FastAPI app with metadata for Swagger UI
app = FastAPI(
    title="政策文档智能爬虫框架",
    description=f"""
    智政智能爬虫框架

    ## Features

    * **Crawler爬虫控制接口**: Start--开始, stop--停止, 爬虫任务监控。
    * **Content Extraction文本元数据抽取接口**: 解析文档、固定字段元数据抽取。
    * **Data Cleaning数据清洗接口**: 支持自定义数据清洗。
    * **LLM Integration大模型爬虫接口**: 使用大模型提示词进行指定数据的爬取以及预清洗任务。
    * **Vector DB向量数据库集成**: 支持多种向量数据库，包括Pinecone、Qdrant、Weaviate等。
    * **配置管理**: 可配置数据库、中间件和LLM服务。
    
    ### 当前已配置的服务
    
    - 数据库: {', '.join([f"{name} ({config.type})" for name, config in app_config.databases.items() if config.enabled])}
    - 中间件: {', '.join([f"{name} ({config.type})" for name, config in app_config.middlewares.items() if config.enabled]) or '无'}
    - LLM服务: {', '.join([f"{name} ({config.type})" for name, config in app_config.llm_services.items() if config.enabled]) or '无'}
    """,
    version="1.0.0",
    docs_url=None,  # Disable default docs
    redoc_url=None,  # Disable default redoc
    contact={
        "name": "Development By zzdsj",
        "email": "",
    },
    license_info={
        "name": "智政科技",
        "url": "https://www.zzdsj.com.cn/",
    }
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Custom Swagger UI route with enhanced styling
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
        swagger_favicon_url="/static/favicon.png"
    )


@app.get("/")
async def root():
    return {"message": "Government Website Crawler System API", "docs": "/docs"}


# 应用启动事件处理
@app.on_event("startup")
async def startup_db_client():
    """应用启动时初始化数据库连接"""
    logger.info("正在初始化数据库和中间件连接...")
    # 预热连接，但不需要特别操作，因为get_async_db()是懒加载的


# 应用关闭事件处理
@app.on_event("shutdown")
async def shutdown_db_client():
    """应用关闭时清理数据库连接"""
    logger.info("正在关闭所有数据库和中间件连接...")
    await close_all_connections()


if __name__ == "__main__":
    import uvicorn
    from config.settings import API_HOST, API_PORT

    uvicorn.run(app, host=API_HOST, port=API_PORT)