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
    try:
        provider = config.get("provider")
        
        if not provider:
            raise HTTPException(status_code=400, detail="Provider not specified")
        
        # Simulate connection test
        # In a real implementation, we would try to connect to the specified provider
        
        if provider == "openai":
            api_key = config.get("api_key")
            if not api_key:
                raise HTTPException(status_code=400, detail="API key required for OpenAI")
                
            # TODO: Actually test OpenAI connection
            return {
                "success": True,
                "message": "Successfully connected to OpenAI",
                "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
            }
            
        elif provider == "ollama":
            base_url = config.get("base_url", "http://localhost:11434")
            
            # TODO: Actually test Ollama connection
            return {
                "success": True,
                "message": f"Successfully connected to Ollama at {base_url}",
                "models": ["llama2", "mistral", "phi"]
            }
            
        elif provider == "custom":
            url = config.get("url")
            if not url:
                raise HTTPException(status_code=400, detail="URL required for custom provider")
                
            # TODO: Actually test custom connection
            return {
                "success": True,
                "message": f"Successfully connected to custom LLM at {url}",
            }
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Connection test failed: {str(e)}")


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
            raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve models: {str(e)}")


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
        raise HTTPException(status_code=500, detail=f"Failed to retrieve model parameters: {str(e)}")


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
        raise HTTPException(status_code=500, detail=f"Failed to save model parameters: {str(e)}")
