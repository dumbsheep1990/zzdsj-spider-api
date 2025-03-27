import os
from typing import Optional, Dict, Any, Union, List
from functools import lru_cache

import openai
from openai import OpenAI, AzureOpenAI

from config.database import get_llm_service_config, LLMServiceType
from utils.logger import get_llm_logger

# 获取LLM服务日志记录器
logger = get_llm_logger()


class LLMClientError(Exception):
    """LLM客户端错误"""
    pass


class BaseLLMClient:
    """LLM客户端基类"""
    def __init__(self, config_name: str):
        self.config_name = config_name
        logger.info(f"初始化LLM客户端: {config_name}")
        self.config = get_llm_service_config(config_name)
        if not self.config:
            logger.error(f"无法找到LLM服务配置: {config_name}")
            raise LLMClientError(f"无法找到LLM服务配置: {config_name}")
        if not self.config.enabled:
            logger.warning(f"LLM服务未启用: {config_name}")
            raise LLMClientError(f"LLM服务未启用: {config_name}")
        self.client = None
        self._initialize()
        logger.info(f"LLM客户端初始化完成: {config_name}, 类型: {self.config.type}")
    
    def _initialize(self):
        """Initialize the client - to be implemented by subclasses"""
        raise NotImplementedError()
    
    async def chat_completion(self, messages: List[Dict[str, Any]], model: str = None, **kwargs) -> Dict[str, Any]:
        """Send a chat completion request to the LLM service"""
        raise NotImplementedError()
    
    async def embedding(self, text: Union[str, List[str]], model: str = None, **kwargs) -> List[List[float]]:
        """Get embeddings for the provided text"""
        raise NotImplementedError()
    
    def get_default_model(self, type: str = "chat") -> str:
        """Get the default model for the specified type"""
        if self.config.default_model:
            return self.config.default_model
        elif self.config.models and len(self.config.models) > 0:
            return self.config.models[0]
        return ""


class OpenAIClient(BaseLLMClient):
    """OpenAI API客户端"""
    def _initialize(self):
        if not self.config.api_key:
            logger.error(f"OpenAI API密钥未配置: {self.config_name}")
            raise LLMClientError("OpenAI API密钥未配置")
        
        # 创建OpenAI客户端
        logger.debug(f"创建OpenAI客户端: {self.config_name}, URI: {self.config.uri}")
        self.client = OpenAI(
            api_key=self.config.api_key,
            organization=self.config.organization if self.config.organization else None,
            base_url=self.config.uri if self.config.uri else None
        )
    
    async def chat_completion(self, messages: List[Dict[str, Any]], model: str = None, **kwargs) -> Dict[str, Any]:
        if not model:
            model = self.get_default_model("chat")
        
        # 合并默认参数和自定义参数
        params = {}
        if self.config.params:
            params.update(self.config.params)
        if kwargs:
            params.update(kwargs)
        
        # 发送请求
        try:
            logger.info(f"发送OpenAI聊天请求: {self.config_name}, 模型: {model}, 消息数: {len(messages)}")
            logger.debug(f"OpenAI请求参数: {params}")
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                **params
            )
            
            usage = response.usage.dict() if hasattr(response.usage, "dict") else vars(response.usage)
            logger.info(f"OpenAI请求成功: 模型: {model}, 令牌使用: {usage}")
            
            return response
        except Exception as e:
            logger.error(f"OpenAI请求失败: {str(e)}", exc_info=True)
            raise LLMClientError(f"OpenAI请求失败: {str(e)}")
    
    async def embedding(self, text: Union[str, List[str]], model: str = None, **kwargs) -> List[List[float]]:
        if not model:
            model = "text-embedding-3-small"  # OpenAI默认向量模型
        
        try:
            # 处理单个文本和文本列表情况
            input_text = [text] if isinstance(text, str) else text
            count = len(input_text)
            logger.info(f"发送OpenAI向量请求: {self.config_name}, 模型: {model}, 文本数: {count}")
            
            response = await self.client.embeddings.create(
                model=model,
                input=input_text,
                **kwargs
            )
            
            # 提取向量数据
            embeddings = [item.embedding for item in response.data]
            dimensions = len(embeddings[0]) if embeddings else 0
            logger.info(f"OpenAI向量请求成功: 模型: {model}, 向量数: {len(embeddings)}, 维度: {dimensions}")
            
            return embeddings
        except Exception as e:
            logger.error(f"OpenAI向量请求失败: {str(e)}", exc_info=True)
            raise LLMClientError(f"OpenAI向量请求失败: {str(e)}")


class AzureOpenAIClient(BaseLLMClient):
    """Azure OpenAI客户端"""
    def _initialize(self):
        if not self.config.api_key:
            logger.error(f"Azure OpenAI API密钥未配置: {self.config_name}")
            raise LLMClientError("Azure OpenAI API密钥未配置")
        if not self.config.api_base:
            logger.error(f"Azure OpenAI API基础URL未配置: {self.config_name}")
            raise LLMClientError("Azure OpenAI API基础URL未配置")
        if not self.config.api_version:
            logger.error(f"Azure OpenAI API版本未配置: {self.config_name}")
            raise LLMClientError("Azure OpenAI API版本未配置")
        
        # 创建Azure OpenAI客户端
        logger.debug(f"创建Azure OpenAI客户端: {self.config_name}, API基础URL: {self.config.api_base}")
        self.client = AzureOpenAI(
            api_key=self.config.api_key,
            api_version=self.config.api_version,
            azure_endpoint=self.config.api_base
        )
        
        # 保存部署名称
        self.deployment_name = self.config.deployment_name
        logger.debug(f"Azure OpenAI部署名称: {self.deployment_name}")
    
    async def chat_completion(self, messages: List[Dict[str, Any]], model: str = None, **kwargs) -> Dict[str, Any]:
        # Azure使用部署名称而非模型名称
        deployment_name = self.deployment_name or self.get_default_model("chat")
        
        # 合并参数
        params = {}
        if self.config.params:
            params.update(self.config.params)
        if kwargs:
            params.update(kwargs)
        
        try:
            logger.info(f"发送Azure OpenAI聊天请求: {self.config_name}, 部署: {deployment_name}, 消息数: {len(messages)}")
            logger.debug(f"Azure OpenAI请求参数: {params}")
            
            response = await self.client.chat.completions.create(
                deployment_name=deployment_name,
                messages=messages,
                **params
            )
            
            usage = response.usage.dict() if hasattr(response.usage, "dict") else vars(response.usage)
            logger.info(f"Azure OpenAI请求成功: 部署: {deployment_name}, 令牌使用: {usage}")
            
            return response
        except Exception as e:
            logger.error(f"Azure OpenAI请求失败: {str(e)}", exc_info=True)
            raise LLMClientError(f"Azure OpenAI请求失败: {str(e)}")
    
    async def embedding(self, text: Union[str, List[str]], model: str = None, **kwargs) -> List[List[float]]:
        # 使用指定的向量部署或默认部署
        deployment_name = model or "text-embedding-ada-002"  # Azure常用向量模型
        
        try:
            # 处理单个文本和文本列表情况
            input_text = [text] if isinstance(text, str) else text
            count = len(input_text)
            logger.info(f"发送Azure OpenAI向量请求: {self.config_name}, 部署: {deployment_name}, 文本数: {count}")
            
            response = await self.client.embeddings.create(
                deployment_name=deployment_name,
                input=input_text,
                **kwargs
            )
            
            # 提取向量数据
            embeddings = [item.embedding for item in response.data]
            dimensions = len(embeddings[0]) if embeddings else 0
            logger.info(f"Azure OpenAI向量请求成功: 部署: {deployment_name}, 向量数: {len(embeddings)}, 维度: {dimensions}")
            
            return embeddings
        except Exception as e:
            logger.error(f"Azure OpenAI向量请求失败: {str(e)}", exc_info=True)
            raise LLMClientError(f"Azure OpenAI向量请求失败: {str(e)}")


class CustomOpenAIClient(BaseLLMClient):
    """OpenAI兼容的自定义/第三方客户端"""
    def _initialize(self):
        if not self.config.uri:
            logger.error(f"自定义LLM服务URL未配置: {self.config_name}")
            raise LLMClientError("自定义LLM服务URL未配置")
        
        # 构建客户端
        logger.debug(f"创建自定义OpenAI兼容客户端: {self.config_name}, URI: {self.config.uri}")
        self.client = OpenAI(
            api_key=self.config.api_key or "sk-",  # 某些自定义服务需要一个占位符密钥
            base_url=self.config.uri
        )
    
    async def chat_completion(self, messages: List[Dict[str, Any]], model: str = None, **kwargs) -> Dict[str, Any]:
        if not model:
            model = self.get_default_model("chat")
        
        # 合并参数
        params = {}
        if self.config.params:
            params.update(self.config.params)
        if kwargs:
            params.update(kwargs)
        
        try:
            logger.info(f"发送自定义LLM服务聊天请求: {self.config_name}, 模型: {model}, 消息数: {len(messages)}")
            logger.debug(f"自定义LLM服务请求参数: {params}")
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                **params
            )
            
            usage_info = "未知" if not hasattr(response, "usage") else (
                response.usage.dict() if hasattr(response.usage, "dict") else vars(response.usage)
            )
            logger.info(f"自定义LLM服务请求成功: 模型: {model}, 令牌使用: {usage_info}")
            
            return response
        except Exception as e:
            logger.error(f"自定义LLM服务请求失败: {str(e)}", exc_info=True)
            raise LLMClientError(f"自定义LLM服务请求失败: {str(e)}")
    
    async def embedding(self, text: Union[str, List[str]], model: str = None, **kwargs) -> List[List[float]]:
        if not model:
            model = self.get_default_model("embedding") or self.get_default_model("chat")
        
        try:
            # 处理单个文本和文本列表情况
            input_text = [text] if isinstance(text, str) else text
            count = len(input_text)
            logger.info(f"发送自定义LLM服务向量请求: {self.config_name}, 模型: {model}, 文本数: {count}")
            
            response = await self.client.embeddings.create(
                model=model,
                input=input_text,
                **kwargs
            )
            
            # 提取向量数据
            embeddings = [item.embedding for item in response.data]
            dimensions = len(embeddings[0]) if embeddings else 0
            logger.info(f"自定义LLM服务向量请求成功: 模型: {model}, 向量数: {len(embeddings)}, 维度: {dimensions}")
            
            return embeddings
        except Exception as e:
            logger.error(f"自定义LLM服务向量请求失败: {str(e)}", exc_info=True)
            raise LLMClientError(f"自定义LLM服务向量请求失败: {str(e)}")


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API客户端"""
    def _initialize(self):
        try:
            import anthropic
            self.anthropic = anthropic
        except ImportError:
            logger.error(f"未安装anthropic包: {self.config_name}")
            raise LLMClientError("请安装 anthropic 包以使用 Claude API")
        
        if not self.config.api_key:
            logger.error(f"Anthropic API密钥未配置: {self.config_name}")
            raise LLMClientError("Anthropic API密钥未配置")
        
        # 创建Anthropic客户端
        logger.debug(f"创建Anthropic客户端: {self.config_name}")
        self.client = self.anthropic.Anthropic(
            api_key=self.config.api_key,
        )
    
    async def chat_completion(self, messages: List[Dict[str, Any]], model: str = None, **kwargs) -> Dict[str, Any]:
        if not model:
            model = self.get_default_model("chat")
        
        # 合并参数
        params = {}
        if self.config.params:
            params.update(self.config.params)
        if kwargs:
            params.update(kwargs)
        
        # Anthropic使用不同的消息格式，需要转换
        system_message = ""
        prompt_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                prompt_messages.append(msg)
        
        try:
            logger.info(f"发送Anthropic Claude聊天请求: {self.config_name}, 模型: {model}, 消息数: {len(messages)}")
            logger.debug(f"Anthropic Claude请求参数: {params}, 系统消息长度: {len(system_message)}, 对话消息数: {len(prompt_messages)}")
            
            message = self.client.messages.create(
                model=model,
                system=system_message,
                messages=[{"role": m["role"], "content": m["content"]} for m in prompt_messages],
                **params
            )
            
            # 转换为OpenAI兼容的格式
            response = {
                "id": message.id,
                "model": message.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": message.content[0].text
                    },
                    "finish_reason": message.stop_reason
                }],
                "usage": {
                    "prompt_tokens": message.usage.input_tokens,
                    "completion_tokens": message.usage.output_tokens,
                    "total_tokens": message.usage.input_tokens + message.usage.output_tokens
                }
            }
            
            logger.info(f"Anthropic Claude请求成功: 模型: {model}, 输入令牌: {message.usage.input_tokens}, 输出令牌: {message.usage.output_tokens}")
            
            return response
        except Exception as e:
            logger.error(f"Anthropic Claude请求失败: {str(e)}", exc_info=True)
            raise LLMClientError(f"Anthropic Claude请求失败: {str(e)}")
    
    async def embedding(self, text: Union[str, List[str]], model: str = None, **kwargs) -> List[List[float]]:
        # Claude当前不提供原生向量接口，需要第三方包或将来更新
        logger.error(f"Anthropic Claude不支持原生向量接口: {self.config_name}")
        raise LLMClientError("Anthropic Claude不支持原生向量接口")


class OllamaClient(BaseLLMClient):
    """Ollama本地模型客户端"""
    def _initialize(self):
        try:
            from langchain_community.llms import Ollama
            from langchain_core.prompts import ChatPromptTemplate
            self.ollama_module = Ollama
            self.prompt_template = ChatPromptTemplate
        except ImportError:
            logger.error(f"未安装langchain或langchain_community包: {self.config_name}")
            raise LLMClientError("请安装 langchain 和 langchain_community 包以使用 Ollama")
        
        if not self.config.uri:
            logger.error(f"Ollama服务URL未配置: {self.config_name}")
            raise LLMClientError("Ollama服务URL未配置")
        
        # Ollama客户端仅在调用时创建，每个模型实例独立
        logger.debug(f"配置Ollama客户端: {self.config_name}, URL: {self.config.uri}")
        self.base_url = self.config.uri
    
    async def chat_completion(self, messages: List[Dict[str, Any]], model: str = None, **kwargs) -> Dict[str, Any]:
        if not model:
            model = self.get_default_model("chat")
        
        # 合并参数
        params = {}
        if self.config.params:
            params.update(self.config.params)
        if kwargs:
            params.update(kwargs)
        
        try:
            # 创建Ollama客户端
            logger.info(f"发送Ollama聊天请求: {self.config_name}, 模型: {model}, 消息数: {len(messages)}")
            logger.debug(f"Ollama请求参数: {params}")
            
            ollama = self.ollama_module(
                base_url=self.base_url,
                model=model,
                **params
            )
            
            # 构建提示词
            template = self.prompt_template.from_messages(messages)
            prompt = template.format()
            
            # 调用Ollama
            logger.debug(f"调用Ollama模型: {model}, 提示词长度: {len(prompt)}")
            response_text = await ollama.ainvoke(prompt)
            
            # 构建兼容OpenAI格式的响应
            response = {
                "id": f"ollama-{model}-{id(response_text)}",
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": -1,  # Ollama不提供这些指标
                    "completion_tokens": -1,
                    "total_tokens": -1
                }
            }
            
            logger.info(f"Ollama请求成功: 模型: {model}, 响应长度: {len(response_text)}")
            
            return response
        except Exception as e:
            logger.error(f"Ollama请求失败: {str(e)}", exc_info=True)
            raise LLMClientError(f"Ollama请求失败: {str(e)}")
    
    async def embedding(self, text: Union[str, List[str]], model: str = None, **kwargs) -> List[List[float]]:
        import aiohttp
        import json
        
        if not model:
            model = self.get_default_model("embedding") or self.get_default_model("chat")
        
        try:
            # 处理文本列表
            input_texts = [text] if isinstance(text, str) else text
            count = len(input_texts)
            logger.info(f"发送Ollama向量请求: {self.config_name}, 模型: {model}, 文本数: {count}")
            
            embeddings = []
            
            async with aiohttp.ClientSession() as session:
                for idx, text_item in enumerate(input_texts):
                    # Ollama的向量接口
                    url = f"{self.base_url.rstrip('/')}/api/embeddings"
                    payload = {"model": model, "prompt": text_item, **kwargs}
                    
                    logger.debug(f"Ollama向量请求 [{idx+1}/{count}]: URL: {url}, 文本长度: {len(text_item)}")
                    
                    async with session.post(url, json=payload) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"Ollama向量请求失败: 状态码: {response.status}, 错误: {error_text}")
                            raise LLMClientError(f"Ollama向量请求失败: {error_text}")
                        
                        data = await response.json()
                        if "embedding" in data:
                            embeddings.append(data["embedding"])
            
            dimensions = len(embeddings[0]) if embeddings else 0
            logger.info(f"Ollama向量请求成功: 模型: {model}, 向量数: {len(embeddings)}, 维度: {dimensions}")
            
            return embeddings
        except Exception as e:
            logger.error(f"Ollama向量请求失败: {str(e)}", exc_info=True)
            raise LLMClientError(f"Ollama向量请求失败: {str(e)}")


@lru_cache(maxsize=8)
def get_llm_client(name: str = "primary") -> BaseLLMClient:
    """Get an LLM client based on configuration"""
    logger.debug(f"获取LLM客户端: {name}")
    config = get_llm_service_config(name)
    if not config:
        logger.error(f"无法找到LLM服务配置: {name}")
        raise LLMClientError(f"无法找到LLM服务配置: {name}")
    
    if not config.enabled:
        logger.error(f"LLM服务未启用: {name}")
        raise LLMClientError(f"LLM服务未启用: {name}")
    
    # 根据类型选择客户端
    logger.info(f"创建LLM客户端: {name}, 类型: {config.type}")
    if config.type == LLMServiceType.OPENAI:
        return OpenAIClient(name)
    elif config.type == LLMServiceType.AZURE:
        return AzureOpenAIClient(name)
    elif config.type == LLMServiceType.ANTHROPIC:
        return AnthropicClient(name)
    elif config.type == LLMServiceType.OLLAMA:
        return OllamaClient(name)
    elif config.type == LLMServiceType.CUSTOM:
        # 判断是否兼容OpenAI接口
        if config.is_openai_compatible:
            return CustomOpenAIClient(name)
        else:
            logger.error(f"不支持的自定义LLM服务类型: {name}, 需要设置is_openai_compatible=true")
            raise LLMClientError(f"不支持的自定义LLM服务类型: {name}, 需要设置 is_openai_compatible=true")
    else:
        logger.error(f"不支持的LLM服务类型: {config.type}")
        raise LLMClientError(f"不支持的LLM服务类型: {config.type}")


# 快捷方式
async def chat_completion(messages: List[Dict[str, Any]], model: str = None, service_name: str = "primary", **kwargs) -> Dict[str, Any]:
    """Send a chat completion request to the configured LLM service"""
    logger.debug(f"调用聊天补全API: 服务名称: {service_name}, 模型: {model if model else '默认'}")
    client = get_llm_client(service_name)
    return await client.chat_completion(messages, model, **kwargs)


async def get_embedding(text: Union[str, List[str]], model: str = None, service_name: str = "primary", **kwargs) -> List[List[float]]:
    """Get embeddings for text from the configured LLM service"""
    text_count = 1 if isinstance(text, str) else len(text)
    logger.debug(f"调用向量嵌入API: 服务名称: {service_name}, 模型: {model if model else '默认'}, 文本数: {text_count}")
    client = get_llm_client(service_name)
    return await client.embedding(text, model, **kwargs)
