from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
import asyncpg
import redis.asyncio
import elasticsearch
from typing import Tuple, Optional, Dict, Any, Union
from config.database import (
    get_db_config, 
    get_middleware_config,
    DatabaseType,
    MiddlewareType
)

# 缓存连接实例
_db_clients = {}
_middleware_clients = {}

# MongoDB 连接
async def get_mongodb_client(name: str = "primary") -> AsyncIOMotorClient:
    """获取 MongoDB 异步客户端"""
    global _db_clients
    if f"mongodb:{name}" not in _db_clients:
        db_config = get_db_config(name)
        if db_config.type != DatabaseType.MONGODB:
            raise ValueError(f"数据库 '{name}' 不是 MongoDB 类型")
        
        uri = db_config.uri
        client = AsyncIOMotorClient(uri)
        _db_clients[f"mongodb:{name}"] = client
    
    return _db_clients[f"mongodb:{name}"]

# PostgreSQL 连接
async def get_postgres_pool(name: str = "postgres") -> asyncpg.Pool:
    """获取 PostgreSQL 连接池"""
    global _db_clients
    if f"postgres:{name}" not in _db_clients:
        db_config = get_db_config(name)
        if db_config.type != DatabaseType.POSTGRESQL:
            raise ValueError(f"数据库 '{name}' 不是 PostgreSQL 类型")
        
        # 构建连接字符串或使用URI
        if db_config.uri:
            dsn = db_config.uri
        else:
            dsn = f"postgresql://{db_config.username}:{db_config.password}@{db_config.host}:{db_config.port}/{db_config.name}"
        
        # 创建连接池
        pool = await asyncpg.create_pool(dsn=dsn, **db_config.params)
        _db_clients[f"postgres:{name}"] = pool
    
    return _db_clients[f"postgres:{name}"]

# Redis 连接
async def get_redis_client(name: str = "cache") -> redis.asyncio.Redis:
    """获取 Redis 客户端"""
    global _middleware_clients
    if f"redis:{name}" not in _middleware_clients:
        middleware_config = get_middleware_config(name)
        if not middleware_config or middleware_config.type != MiddlewareType.REDIS:
            raise ValueError(f"中间件 '{name}' 不是 Redis 类型或不存在")
        
        # 如果提供了URI，直接使用
        if middleware_config.uri:
            redis_client = redis.asyncio.from_url(middleware_config.uri)
        else:
            # 否则使用主机、端口等参数构建连接
            redis_client = redis.asyncio.from_url(
                f"redis://{middleware_config.host}:{middleware_config.port}",
                password=middleware_config.password,
                **middleware_config.params
            )
        _middleware_clients[f"redis:{name}"] = redis_client
    
    return _middleware_clients[f"redis:{name}"]

# Elasticsearch 连接
def get_elasticsearch_client(name: str = "search") -> elasticsearch.AsyncElasticsearch:
    """获取 Elasticsearch 客户端"""
    global _middleware_clients
    if f"elasticsearch:{name}" not in _middleware_clients:
        middleware_config = get_middleware_config(name)
        if not middleware_config or middleware_config.type != MiddlewareType.ELASTICSEARCH:
            raise ValueError(f"中间件 '{name}' 不是 Elasticsearch 类型或不存在")
        
        # 构建连接
        if middleware_config.uri:
            es = elasticsearch.AsyncElasticsearch(middleware_config.uri)
        else:
            es = elasticsearch.AsyncElasticsearch(
                [f"{middleware_config.host}:{middleware_config.port}"],
                http_auth=(middleware_config.username, middleware_config.password) if middleware_config.username else None,
                **middleware_config.params
            )
        _middleware_clients[f"elasticsearch:{name}"] = es
    
    return _middleware_clients[f"elasticsearch:{name}"]

# 向量数据库连接工厂
def get_vector_db_client(name: str = "vector"):
    """获取向量数据库客户端"""
    middleware_config = get_middleware_config(name)
    if not middleware_config or middleware_config.type == MiddlewareType.NONE:
        return None
    
    # 根据向量数据库类型返回相应的客户端
    if middleware_config.type == MiddlewareType.PINECONE:
        try:
            import pinecone
            pinecone.init(
                api_key=middleware_config.api_key,
                environment=middleware_config.params.get("environment", "us-west1-gcp")
            )
            return pinecone
        except ImportError:
            raise ImportError("请安装 pinecone-client 包")
    
    elif middleware_config.type == MiddlewareType.QDRANT:
        try:
            from qdrant_client import QdrantClient
            if middleware_config.uri:
                return QdrantClient(url=middleware_config.uri, api_key=middleware_config.api_key)
            else:
                return QdrantClient(
                    host=middleware_config.host,
                    port=middleware_config.port,
                    api_key=middleware_config.api_key
                )
        except ImportError:
            raise ImportError("请安装 qdrant-client 包")
    
    elif middleware_config.type == MiddlewareType.WEAVIATE:
        try:
            import weaviate
            auth_config = None
            if middleware_config.api_key:
                auth_config = weaviate.auth.AuthApiKey(api_key=middleware_config.api_key)
            
            if middleware_config.uri:
                return weaviate.Client(url=middleware_config.uri, auth_client_secret=auth_config)
            else:
                return weaviate.Client(
                    url=f"http://{middleware_config.host}:{middleware_config.port}",
                    auth_client_secret=auth_config
                )
        except ImportError:
            raise ImportError("请安装 weaviate-client 包")
    
    # 可以根据需要添加更多的向量数据库支持
    
    raise ValueError(f"不支持的向量数据库类型: {middleware_config.type}")

# 原始的兼容API，保留向后兼容性
async def get_async_db():
    """获取默认异步MongoDB数据库连接(兼容现有代码)"""
    db_config = get_db_config("primary")
    client = await get_mongodb_client("primary")
    db = client[db_config.name]
    return db

def get_sync_db():
    """获取同步MongoDB数据库连接(兼容现有代码)"""
    db_config = get_db_config("primary")
    client = MongoClient(db_config.uri)
    db = client[db_config.name]
    return db, client

# 关闭所有连接的方法
async def close_all_connections():
    """关闭所有数据库和中间件连接"""
    # 关闭数据库连接
    for key, client in _db_clients.items():
        if "mongodb:" in key:
            client.close()
        elif "postgres:" in key:
            await client.close()
    
    # 关闭中间件连接
    for key, client in _middleware_clients.items():
        if "redis:" in key:
            await client.close()
        elif "elasticsearch:" in key:
            await client.close()
    
    # 清空缓存
    _db_clients.clear()
    _middleware_clients.clear()