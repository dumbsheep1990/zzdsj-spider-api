from fastapi import APIRouter, HTTPException, Depends, Body
from typing import Dict, Any, List, Optional
from config.database import (
    AppConfig, 
    DatabaseConfig, 
    MiddlewareConfig, 
    LLMServiceConfig,
    get_app_config,
    update_config,
    get_db_config,
    get_middleware_config,
    get_llm_service_config,
    DatabaseType,
    MiddlewareType,
    LLMServiceType
)
from db.connection import get_async_db, get_sync_db

router = APIRouter(prefix="/config", tags=["config"])


@router.get("/",
            summary="获取系统配置",
            description="返回当前系统的完整配置信息。",
            response_description="系统配置信息"
            )
async def get_system_config():
    """获取系统配置"""
    try:
        # 获取应用配置并转换为字典
        config = get_app_config()
        return config.dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取系统配置失败: {str(e)}")


@router.post("/",
             summary="更新系统配置",
             description="更新系统的配置信息。",
             response_description="更新后的系统配置信息"
             )
async def update_system_config(config: Dict[str, Any] = Body(...)):
    """更新系统配置"""
    try:
        # 将配置字典转换为AppConfig对象
        new_config = AppConfig.parse_obj(config)
        # 更新配置
        update_config(new_config)
        return new_config.dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新系统配置失败: {str(e)}")


@router.get("/databases",
            summary="获取数据库配置",
            description="返回所有数据库配置信息。",
            response_description="数据库配置列表"
            )
async def get_databases():
    """获取所有数据库配置"""
    try:
        config = get_app_config()
        result = {}
        for name, db_config in config.databases.items():
            # 过滤敏感信息
            db_dict = db_config.dict()
            if "password" in db_dict:
                db_dict["password"] = "*****" if db_dict["password"] else None
            result[name] = db_dict
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取数据库配置失败: {str(e)}")


@router.post("/databases/{name}",
             summary="更新数据库配置",
             description="更新指定名称的数据库配置。",
             response_description="更新后的数据库配置"
             )
async def update_database(name: str, config: Dict[str, Any] = Body(...)):
    """更新指定数据库配置"""
    try:
        app_config = get_app_config()
        db_config = DatabaseConfig.parse_obj(config)
        app_config.databases[name] = db_config
        update_config(app_config)
        
        # 返回更新后的配置（隐藏密码）
        result = db_config.dict()
        if "password" in result:
            result["password"] = "*****" if result["password"] else None
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新数据库配置失败: {str(e)}")


@router.get("/middlewares",
            summary="获取中间件服务配置",
            description="返回所有中间件服务配置信息。",
            response_description="中间件服务配置列表"
            )
async def get_middlewares():
    """获取所有中间件服务配置"""
    try:
        config = get_app_config()
        result = {}
        for name, middleware_config in config.middlewares.items():
            # 过滤敏感信息
            mw_dict = middleware_config.dict()
            if "password" in mw_dict:
                mw_dict["password"] = "*****" if mw_dict["password"] else None
            if "api_key" in mw_dict:
                mw_dict["api_key"] = "*****" if mw_dict["api_key"] else None
            result[name] = mw_dict
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取中间件服务配置失败: {str(e)}")


@router.post("/middlewares/{name}",
             summary="更新中间件服务配置",
             description="更新指定名称的中间件服务配置。",
             response_description="更新后的中间件服务配置"
             )
async def update_middleware(name: str, config: Dict[str, Any] = Body(...)):
    """更新指定中间件服务配置"""
    try:
        app_config = get_app_config()
        middleware_config = MiddlewareConfig.parse_obj(config)
        app_config.middlewares[name] = middleware_config
        update_config(app_config)
        
        # 返回更新后的配置（隐藏敏感信息）
        result = middleware_config.dict()
        if "password" in result:
            result["password"] = "*****" if result["password"] else None
        if "api_key" in result:
            result["api_key"] = "*****" if result["api_key"] else None
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新中间件服务配置失败: {str(e)}")


@router.get("/llm-services",
            summary="获取LLM服务配置",
            description="返回所有LLM服务配置信息。",
            response_description="LLM服务配置列表"
            )
async def get_llm_services():
    """获取所有LLM服务配置"""
    try:
        config = get_app_config()
        result = {}
        for name, llm_config in config.llm_services.items():
            # 过滤敏感信息
            llm_dict = llm_config.dict()
            if "api_key" in llm_dict:
                llm_dict["api_key"] = "*****" if llm_dict["api_key"] else None
            result[name] = llm_dict
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取LLM服务配置失败: {str(e)}")


@router.post("/llm-services/{name}",
             summary="更新LLM服务配置",
             description="更新指定名称的LLM服务配置。",
             response_description="更新后的LLM服务配置"
             )
async def update_llm_service(name: str, config: Dict[str, Any] = Body(...)):
    """更新指定LLM服务配置"""
    try:
        app_config = get_app_config()
        llm_config = LLMServiceConfig.parse_obj(config)
        app_config.llm_services[name] = llm_config
        update_config(app_config)
        
        # 返回更新后的配置（隐藏敏感信息）
        result = llm_config.dict()
        if "api_key" in result:
            result["api_key"] = "*****" if result["api_key"] else None
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新LLM服务配置失败: {str(e)}")


@router.get("/test-connection/{type}/{name}",
            summary="测试连接",
            description="测试指定类型和名称的服务连接。",
            response_description="连接测试结果"
            )
async def test_connection(type: str, name: str):
    """测试服务连接"""
    try:
        if type == "database":
            # 测试数据库连接
            db_config = get_db_config(name)
            if db_config.type == DatabaseType.MONGODB:
                # 测试MongoDB连接
                from db.connection import get_mongodb_client
                client = await get_mongodb_client(name)
                await client.admin.command("ping")
                return {"success": True, "message": f"Successfully connected to MongoDB: {name}"}
            elif db_config.type == DatabaseType.POSTGRESQL:
                # 测试PostgreSQL连接
                from db.connection import get_postgres_pool
                pool = await get_postgres_pool(name)
                async with pool.acquire() as conn:
                    version = await conn.fetchval("SELECT version()")
                return {"success": True, "message": f"Successfully connected to PostgreSQL: {name}", "version": version}
            else:
                return {"success": False, "message": f"Unsupported database type: {db_config.type}"}
        
        elif type == "middleware":
            # 测试中间件连接
            middleware_config = get_middleware_config(name)
            if not middleware_config:
                return {"success": False, "message": f"Middleware not found: {name}"}
            
            if middleware_config.type == MiddlewareType.REDIS:
                # 测试Redis连接
                from db.connection import get_redis_client
                redis = await get_redis_client(name)
                await redis.ping()
                return {"success": True, "message": f"Successfully connected to Redis: {name}"}
            elif middleware_config.type == MiddlewareType.ELASTICSEARCH:
                # 测试Elasticsearch连接
                from db.connection import get_elasticsearch_client
                es = get_elasticsearch_client(name)
                info = await es.info()
                return {"success": True, "message": f"Successfully connected to Elasticsearch: {name}", "version": info["version"]["number"]}
            elif middleware_config.type in [MiddlewareType.PINECONE, MiddlewareType.QDRANT, MiddlewareType.WEAVIATE]:
                # 测试向量数据库连接
                from db.connection import get_vector_db_client
                vector_db = get_vector_db_client(name)
                if vector_db:
                    return {"success": True, "message": f"Successfully connected to {middleware_config.type}: {name}"}
                else:
                    return {"success": False, "message": f"Failed to connect to {middleware_config.type}: {name}"}
            else:
                return {"success": False, "message": f"Unsupported middleware type: {middleware_config.type}"}
        
        elif type == "llm":
            # 测试LLM服务连接
            llm_config = get_llm_service_config(name)
            if not llm_config:
                return {"success": False, "message": f"LLM service not found: {name}"}
            
            if llm_config.type == LLMServiceType.OPENAI:
                # 测试OpenAI连接
                import openai
                openai.api_key = llm_config.api_key
                models = openai.Model.list()
                return {"success": True, "message": f"Successfully connected to OpenAI API", "available_models": [m.id for m in models.data[:5]]}
            elif llm_config.type == LLMServiceType.OLLAMA:
                # 测试Ollama连接
                import requests
                response = requests.get(f"{llm_config.uri}/api/tags")
                if response.status_code == 200:
                    return {"success": True, "message": f"Successfully connected to Ollama", "available_models": response.json()}
                else:
                    return {"success": False, "message": f"Failed to connect to Ollama: {response.text}"}
            else:
                return {"success": False, "message": f"Unsupported LLM service type: {llm_config.type}"}
        
        else:
            return {"success": False, "message": f"Unsupported service type: {type}"}
    
    except Exception as e:
        return {"success": False, "message": f"Connection test failed: {str(e)}"}
