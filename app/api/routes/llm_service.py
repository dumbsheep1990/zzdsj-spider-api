from fastapi import APIRouter, HTTPException, Depends, Body, Query
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from config.database import get_llm_service_config, LLMServiceType
from services.llm_factory import chat_completion, get_embedding, LLMClientError
from utils.logger import get_api_logger

router = APIRouter(prefix="/llm-service", tags=["llm-service"])
logger = get_api_logger()


class ChatMessage(BaseModel):
    """聊天消息模型"""
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """聊天补全请求"""
    messages: List[ChatMessage]
    model: Optional[str] = None
    service_name: str = "primary"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False


class EmbeddingRequest(BaseModel):
    """嵌入向量请求"""
    text: List[str]
    model: Optional[str] = None
    service_name: str = "primary"


@router.post("/chat",
            summary="LLM聊天补全",
            description="使用配置的LLM服务进行聊天补全，支持不同的服务提供商。",
            response_description="LLM聊天补全结果"
            )
async def llm_chat_completion(request: ChatCompletionRequest):
    """使用配置的LLM服务进行聊天补全"""
    try:
        logger.info(f"接收聊天补全请求: 服务名={request.service_name}, 模型={request.model if request.model else '默认'}, 消息数={len(request.messages)}")
        logger.debug(f"请求参数: {request.dict(exclude={'messages'})}")
        
        # 转换消息格式
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # 构建参数
        kwargs = {}
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        kwargs["stream"] = request.stream
        
        # 调用LLM服务
        response = await chat_completion(
            messages=messages,
            model=request.model,
            service_name=request.service_name,
            **kwargs
        )
        
        logger.info(f"聊天补全请求成功: 服务名={request.service_name}, 模型={request.model if request.model else '默认'}")
        
        return response
    except LLMClientError as e:
        error_msg = str(e)
        logger.error(f"聊天补全请求失败: {error_msg}", exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        error_msg = f"LLM聊天补全服务异常: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/embedding",
             summary="获取文本嵌入向量",
             description="使用配置的LLM服务将文本转换为嵌入向量，支持不同的服务提供商。",
             response_description="文本嵌入向量结果"
             )
async def llm_embedding(request: EmbeddingRequest):
    """获取文本嵌入向量"""
    try:
        # 记录请求信息
        text_type = "list" if isinstance(request.text, list) else "string"
        text_count = len(request.text) if isinstance(request.text, list) else 1
        text_length = sum(len(t) for t in request.text) if isinstance(request.text, list) else len(request.text)
        
        logger.info(f"接收嵌入向量请求: 服务名={request.service_name}, 模型={request.model if request.model else '默认'}, 文本类型={text_type}, 文本数={text_count}, 文本长度={text_length}")
        
        # 调用LLM服务
        embeddings = await get_embedding(
            text=request.text,
            model=request.model,
            service_name=request.service_name
        )
        
        # 构建响应
        response = {
            "embeddings": embeddings,
            "dimensions": len(embeddings[0]) if embeddings and len(embeddings) > 0 else 0
        }
        
        logger.info(f"嵌入向量请求成功: 服务名={request.service_name}, 模型={request.model if request.model else '默认'}, 嵌入向量数={len(embeddings)}, 维度={response['dimensions']}")
        
        return response
    except LLMClientError as e:
        error_msg = str(e)
        logger.error(f"嵌入向量请求失败: {error_msg}", exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        error_msg = f"LLM嵌入向量服务异常: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/services",
           summary="获取可用的LLM服务",
           description="返回所有配置且启用的LLM服务列表。",
           response_description="可用的LLM服务列表"
           )
async def list_llm_services():
    """获取可用的LLM服务列表"""
    from config.database import get_app_config
    
    try:
        logger.info("接收获取LLM服务列表请求")
        config = get_app_config()
        services = {}
        
        for name, service_config in config.llm_services.items():
            if service_config.enabled:
                service_info = {
                    "type": service_config.type,
                    "name": service_config.name or name,
                    "description": service_config.description,
                    "models": service_config.models,
                    "default_model": service_config.default_model or (service_config.models[0] if service_config.models else "")
                }
                services[name] = service_info
        
        logger.info(f"LLM服务列表请求成功: 服务数={len(services)}")
        return {"services": services}
    except Exception as e:
        error_msg = f"获取LLM服务列表异常: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/test/{service_name}",
           summary="测试LLM服务连接",
           description="测试指定LLM服务的连接和基本功能。",
           response_description="测试结果"
           )
async def test_llm_service(service_name: str):
    """测试LLM服务连接"""
    try:
        logger.info(f"接收测试LLM服务请求: 服务名={service_name}")
        config = get_llm_service_config(service_name)
        if not config:
            raise HTTPException(status_code=404, detail=f"LLM服务不存在: {service_name}")
        
        if not config.enabled:
            raise HTTPException(status_code=400, detail=f"LLM服务未启用: {service_name}")
        
        # 简单测试消息
        test_messages = [
            {"role": "system", "content": "你是一个智能助手。请用简短的一句话回答问题。"},
            {"role": "user", "content": "你好，请简单介绍一下你自己。"}
        ]
        
        # 调用LLM服务
        response = await chat_completion(
            messages=test_messages,
            service_name=service_name,
            max_tokens=50  # 限制响应长度
        )
        
        # 提取结果
        assistant_message = response.choices[0].message.content if hasattr(response, "choices") else response["choices"][0]["message"]["content"]
        
        logger.info(f"LLM服务测试成功: 服务名={service_name}, 响应长度={len(assistant_message)}")
        return {
            "success": True,
            "service": service_name,
            "type": config.type,
            "response": assistant_message,
            "model": response.model if hasattr(response, "model") else response.get("model", "未知模型")
        }
    except LLMClientError as e:
        error_msg = str(e)
        logger.error(f"LLM服务测试失败: {error_msg}", exc_info=True)
        return {"success": False, "service": service_name, "error": error_msg}
    except Exception as e:
        error_msg = f"LLM服务测试异常: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"success": False, "service": service_name, "error": error_msg}


@router.get("/configs",
            summary="获取LLM服务配置",
            description="获取所有配置且启用的LLM服务列表，包括服务类型、名称、描述、模型、默认模型等信息。",
            response_description="LLM服务配置列表"
            )
async def get_llm_configs():
    """获取LLM服务配置"""
    logger.info("接收获取LLM服务配置请求")
    
    try:
        # 从配置文件获取所有LLM服务配置
        from config.database import get_app_config
        app_config = get_app_config()
        
        # 处理并返回LLM服务配置
        llm_configs = {}
        for name, config in app_config.llm_services.items():
            if config.enabled:
                llm_configs[name] = {
                    "name": config.name,
                    "type": config.type,
                    "description": config.description,
                    "models": config.models,
                    "default_model": config.default_model,
                    "is_openai_compatible": getattr(config, "is_openai_compatible", False)
                }
        
        # 返回LLM服务配置
        return {
            "status": "success",
            "llm_services": llm_configs,
            "primary_service": app_config.primary_llm_service,
            "timestamp": "获取时间"
        }
    except Exception as e:
        logger.error(f"获取LLM服务配置异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取LLM服务配置异常: {str(e)}")
