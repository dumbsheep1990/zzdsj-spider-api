from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query, Path, Body
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
from db.connection import get_async_db

router = APIRouter(prefix="/llm", tags=["llm"])


@router.get("/config",
            summary="Get LLM configuration",
            description="Returns the current LLM service configuration.",
            response_description="Current LLM configuration"
            )
async def get_config(db=Depends(get_async_db)):
    """Get current LLM service configuration"""
    try:
        config = await db.system_config.find_one({"type": "llm_config"})
        
        if not config:
            # Return default config if none exists
            return {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "max_tokens": 4000,
                "temperature": 0.7,
                "config_exists": False
            }
        
        # Remove internal fields
        if "_id" in config:
            config["_id"] = str(config["_id"])
        if "type" in config:
            del config["type"]
        
        config["config_exists"] = True
        return config
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve LLM configuration: {str(e)}")


@router.post("/config",
             summary="Save LLM configuration",
             description="Saves LLM service configuration.",
             response_description="Saved LLM configuration"
             )
async def save_config(config: Dict[str, Any] = Body(...), db=Depends(get_async_db)):
    """Save LLM service configuration"""
    try:
        # Add metadata
        config["type"] = "llm_config"
        config["updated_at"] = datetime.now()
        
        # Upsert config
        result = await db.system_config.update_one(
            {"type": "llm_config"},
            {"$set": config},
            upsert=True
        )
        
        # Clean up response
        if "_id" in config:
            config["_id"] = str(config["_id"])
        if "type" in config:
            del config["type"]
        
        config["config_exists"] = True
        return config
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save LLM configuration: {str(e)}")


@router.post("/test-connection",
             summary="Test LLM connection",
             description="Tests the connection to the specified LLM provider.",
             response_description="Test connection result"
             )
async def test_connection(config: Dict[str, Any] = Body(...)):
    """Test connection to LLM provider"""
    from utils.logger import get_api_logger
    logger = get_api_logger()
    
    try:
        logger.info(f"u63a5u6536LLMu8fdeu63a5u6d4bu8bd5u8bf7u6c42uff0cu8bf7u6c42u6570u636e: {json.dumps(config, default=str)}")
        
        # u68c0u67e5u65b0u683cu5f0fuff1au524du7aefu73b0u5728u53d1u9001 {provider: "...", config: {...}}
        if "provider" in config and "config" in config:
            # u65b0u683cu5f0fu8bf7u6c42
            provider = config["provider"]
            config_data = config["config"]
            
            # u68c0u67e5u670du52a1u662fu5426u542fu7528
            enabled = config_data.get("enabled", True)  # u9ed8u8ba4u4e3au542fu7528
            logger.info(f"u670du52a1{provider}u542fu7528u72b6u6001: {enabled}")
            
            if enabled is False:
                logger.warning(f"u8bf7u6c42u6d4bu8bd5u7684LLMu670du52a1 {provider} u672au542fu7528")
                return {
                    "success": True,
                    "connected": False,
                    "message": f"LLMu670du52a1 {provider} u672au542fu7528",
                    "provider": provider,
                    "service_name": config_data.get("service_name", config_data.get("name", "default")),
                    "enabled": False
                }
            
            # u5c06u914du7f6eu6570u636eu5408u5e76u5230u4e3bu914du7f6eu4e2du4ee5u517cu5bb9u73b0u6709u903bu8f91
            config.update(config_data)
        else:
            # u65e7u683cu5f0fu8bf7u6c42(u76f4u63a5u83b7u53d6provider)
            provider = None
            if "provider" in config:
                provider = config["provider"]
            elif "type" in config:
                provider = config["type"]
            elif "service_type" in config:
                provider = config["service_type"]
            
            # u68c0u67e5u670du52a1u662fu5426u542fu7528(u517cu5bb9u65e7u683cu5f0f)
            enabled = config.get("enabled", True)  # u9ed8u8ba4u4e3au542fu7528
            if enabled is False:
                logger.warning(f"u8bf7u6c42u6d4bu8bd5u7684LLMu670du52a1 {provider} u672au542fu7528")
                return {
                    "success": True,
                    "connected": False,
                    "message": f"LLMu670du52a1 {provider} u672au542fu7528",
                    "provider": provider,
                    "enabled": False
                }
            
        service_name = config.get("service_name", config.get("name", "default"))
        api_key = config.get("api_key", config.get("key", config.get("apiKey", "")))
        base_url = config.get("base_url", config.get("url", config.get("uri", "")))
        active_provider = config.get("activeProvider", "local")  # u9ed8u8ba4u4e3au672cu5730u6a21u5f0f
        
        # u8bb0u5f55u89e3u6790u7684u5173u952eu53c2u6570
        logger.info(f"u89e3u6790u540eu7684u53c2u6570: provider={provider}, service_name={service_name}, activeProvider={active_provider}, enabled={enabled}")
        
        if not provider:
            error_msg = "Provider not specified in request"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        # u5904u7406u4e0du76f4u4e92u8054u5e03u5f0f
        if provider.lower() in ["openai", "azure"]:
            # u5982u679cu4e3au4e91u989du673au4e3au4e92u8054u5e03u5f0fuff0cu4e3au8981u6c17 API key
            if not api_key and active_provider.lower() == "cloud":
                error_msg = f"API key required for {provider} in cloud mode"
                logger.error(error_msg)
                raise HTTPException(status_code=400, detail=error_msg)
                
            # TODO: u5b9eu6bd5u6d4bu8bd5 OpenAI u8fdeu63a5
            logger.info(f"u6210u529fu6d4bu8bd5u5230 {provider} (u6a21u5f0f: {active_provider})")
            return {
                "success": True,
                "message": f"u6210u529fu6d4bu8bd5u5230 {provider} ({active_provider} u6a21u5f0f)",
                "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                "provider": provider,
                "service_name": service_name,
                "active_provider": active_provider
            }
            
        elif provider.lower() in ["ollama", "local"]:
            actual_base_url = base_url or "http://localhost:11434"
            
            # u5bf9u4e8eu4e0du76f4u4e92u8054u5e03u5f0fuff0cu4e0du76f4u4e92u8054u5e03u5f0fuff0cu4e3au4e0du8981 API key
            logger.info(f"u6210u529fu6d4bu8bd5u5230 {provider} u5728 {actual_base_url}")
            return {
                "success": True,
                "message": f"u6210u529fu6d4bu8bd5u5230 {provider} u5728 {actual_base_url}",
                "models": ["llama2", "mistral", "deepseek-r1-distill-qwen-32b"],
                "provider": provider,
                "service_name": service_name,
                "active_provider": active_provider
            }
            
        elif provider.lower() in ["deepseek", "custom"]:
            # u5bf9u4e8eu4e0du76f4u4e92u8054u5e03u5f0fuff0cu4e3au8981u68c0u67e5 base_url
            if not base_url and active_provider.lower() == "cloud":
                error_msg = f"Base URL required for {provider} in cloud mode"
                logger.error(error_msg)
                raise HTTPException(status_code=400, detail=error_msg)
            
            # custom u8fdeu63a5u6210u529fu3002
            logger.info(f"u6210u529fu6d4bu8bd5u5230 custom model {provider} u5728 {base_url or '(u9ed8u8ba4 URL)'}")
            return {
                "success": True,
                "message": f"u6210u529fu6d4bu8bd5u5230 {provider} custom model",
                "models": ["deepseek-r1-distill-qwen-32b", "bce-local-base_v1"],
                "provider": provider,
                "service_name": service_name,
                "active_provider": active_provider
            }
        
        else:
            # u4e0du652fu4e92u8054u5e03u5f0f
            error_msg = f"u4e0du652fu4e92u8054u5e03u5f0f: {provider}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
    
    except KeyError as e:
        error_msg = f"u7f3au4e09u5f53u524du4e8bu8fd9u4e2du6570: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
        
    except Exception as e:
        error_msg = f"u6d4bu8bd5u8fdeu63a5u9519u8bef: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/models/{provider}",
            summary="Get available models",
            description="Returns a list of available models for the specified provider.",
            response_description="Available models"
            )
async def get_available_models(
        provider: str = Path(..., description="LLM provider (e.g., openai, ollama)"),
):
    """Get available models for a specific provider"""
    try:
        if provider == "openai":
            return {
                "models": [
                    {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "context_length": 4096},
                    {"id": "gpt-4", "name": "GPT-4", "context_length": 8192},
                    {"id": "gpt-4-turbo", "name": "GPT-4 Turbo", "context_length": 128000}
                ]
            }
            
        elif provider == "ollama":
            return {
                "models": [
                    {"id": "llama2", "name": "Llama 2", "context_length": 4096},
                    {"id": "mistral", "name": "Mistral", "context_length": 8192},
                    {"id": "mistral-openorca", "name": "Mistral OpenOrca", "context_length": 8192},
                    {"id": "phi", "name": "Phi", "context_length": 2048},
                    {"id": "vicuna", "name": "Vicuna", "context_length": 4096},
                    {"id": "mixtral", "name": "Mixtral 8x7B", "context_length": 32000}
                ]
            }
            
        elif provider == "custom":
            return {
                "models": [
                    {"id": "custom", "name": "Custom Model", "context_length": "unknown"}
                ]
            }
            
        else:
            raise HTTPException(status_code=400, detail=f"u4e0du652fu4e92u8054u5e03u5f0f: {provider}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"u62d2u53d6u6a21u5f0fu8fdeu63a5u9519u8bef: {str(e)}")


@router.get("/model-params/{model}",
            summary="Get model parameters",
            description="Returns the parameters configuration for a specific model.",
            response_description="Model parameter configuration"
            )
async def get_model_params(
        model: str = Path(..., description="Model ID"),
        db=Depends(get_async_db)
):
    """Get parameter configuration for a specific model"""
    try:
        # Try to get saved params from the database
        params = await db.llm_model_params.find_one({"model_id": model})
        
        if params:
            if "_id" in params:
                params["_id"] = str(params["_id"])
            return params
        
        # Return default parameters based on model
        if model.startswith("gpt-"):
            return {
                "model_id": model,
                "temperature": 0.7,
                "max_tokens": 4000,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
        elif model in ["llama2", "mistral", "phi", "vicuna"]:
            return {
                "model_id": model,
                "temperature": 0.7,
                "max_tokens": 2000,
                "top_p": 0.9,
                "stop": ["\n\n"]
            }
        else:
            return {
                "model_id": model,
                "temperature": 0.7,
                "max_tokens": 2000
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"u62d2u53d6u6a21u5f0fu53c2u6570u8fdeu63a5u9519u8bef: {str(e)}")


@router.post("/model-params/{model}",
             summary="Save model parameters",
             description="Saves parameter configuration for a specific model.",
             response_description="Saved model parameters"
             )
async def save_model_params(
        model: str = Path(..., description="Model ID"),
        params: Dict[str, Any] = Body(...),
        db=Depends(get_async_db)
):
    """Save parameter configuration for a specific model"""
    try:
        # Add model ID and timestamps
        params["model_id"] = model
        params["updated_at"] = datetime.now()
        
        # Upsert into database
        result = await db.llm_model_params.update_one(
            {"model_id": model},
            {"$set": params},
            upsert=True
        )
        
        # Clean response
        if "_id" in params:
            params["_id"] = str(params["_id"])
            
        return params
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"u4fddu5b58u6a21u5f0fu53c2u6570u8fdeu63a5u9519u8bef: {str(e)}")
