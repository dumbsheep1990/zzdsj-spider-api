import os
from typing import Dict, Any, Optional, List
from enum import Enum
import json
from pydantic import BaseModel, Field
import yaml
try:
    import tomli_w
except ImportError:
    tomli_w = None

# 尝试导入TOML库，Python 3.11+使用内置tomllib，旧版本使用tomli
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


class DatabaseType(str, Enum):
    """支持的数据库类型"""
    MONGODB = "mongodb"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    NONE = "none"


class MiddlewareType(str, Enum):
    """支持的中间件类型"""
    REDIS = "redis"
    ELASTICSEARCH = "elasticsearch"
    PINECONE = "pinecone"
    QDRANT = "qdrant"
    WEAVIATE = "weaviate"
    MILVUS = "milvus"
    MINIO = "minio"  # 对象存储
    OTHER = "other"
    NONE = "none"


class LLMServiceType(str, Enum):
    """LLM服务类型枚举"""
    NONE = "none"          # 未配置
    OPENAI = "openai"      # OpenAI API
    AZURE = "azure"        # Azure OpenAI
    ANTHROPIC = "anthropic"  # Anthropic Claude
    OLLAMA = "ollama"      # Ollama本地模型
    CUSTOM = "custom"      # 自定义OpenAI兼容接口


class DatabaseConfig(BaseModel):
    """数据库配置"""
    type: DatabaseType
    uri: Optional[str] = ""
    name: Optional[str] = ""
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    description: Optional[str] = None


class MiddlewareConfig(BaseModel):
    """中间件服务配置"""
    type: MiddlewareType
    uri: Optional[str] = ""
    name: Optional[str] = ""
    host: Optional[str] = None
    port: Optional[int] = None
    api_key: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    description: Optional[str] = None


class LLMServiceConfig(BaseModel):
    """LLM服务配置"""
    type: LLMServiceType = Field(default=LLMServiceType.NONE, description="服务类型")
    name: Optional[str] = Field(default="", description="服务名称")
    uri: Optional[str] = Field(default="", description="服务URI")
    api_key: Optional[str] = Field(default="", description="API密钥")
    api_version: Optional[str] = Field(default="", description="API版本")
    api_base: Optional[str] = Field(default="", description="API基础URL，仅当URI为空时使用")
    organization: Optional[str] = Field(default="", description="组织ID，用于OpenAI等服务")
    deployment_name: Optional[str] = Field(default="", description="部署名称，用于Azure OpenAI")
    enabled: bool = Field(default=False, description="是否启用")
    description: Optional[str] = Field(default="", description="描述")
    models: List[str] = Field(default_factory=list, description="支持的模型列表")
    default_model: Optional[str] = Field(default="", description="默认模型")
    is_openai_compatible: bool = Field(default=False, description="是否兼容OpenAI接口")
    params: Dict[str, Any] = Field(default_factory=dict, description="附加参数")


class AppConfig(BaseModel):
    """应用配置"""
    app_name: str = "LLM Spider"
    version: Optional[str] = "1.0.0"
    api_host: Optional[str] = "0.0.0.0"
    api_port: Optional[int] = 8000
    debug: bool = False
    log_level: Optional[str] = "info"
    databases: Dict[str, DatabaseConfig] = Field(default_factory=dict)
    middlewares: Dict[str, MiddlewareConfig] = Field(default_factory=dict)
    llm_services: Dict[str, LLMServiceConfig] = Field(default_factory=dict)


# 获取配置文件路径
def get_config_file_path() -> str:
    """获取配置文件路径"""
    # 优先使用环境变量中指定的配置文件
    config_file = os.getenv("CONFIG_FILE", "")
    if config_file:
        return config_file
    
    # 默认配置文件路径
    config_dir = os.path.dirname(os.path.abspath(__file__))
    default_paths = [
        os.path.join(config_dir, "config.yaml"),
        os.path.join(config_dir, "config.yml"),
        os.path.join(config_dir, "config.toml"),
        os.path.join(config_dir, "config.json")
    ]
    
    # 检查默认路径
    for path in default_paths:
        if os.path.exists(path):
            return path
    
    # 如果没有找到配置文件，返回默认的YAML配置文件路径
    return os.path.join(config_dir, "config.yaml")


# 默认配置
def get_default_config() -> AppConfig:
    """获取默认配置"""
    return AppConfig(
        databases={
            "primary": DatabaseConfig(
                type=DatabaseType.MONGODB,
                uri=os.getenv("MONGODB_URI", "mongodb://localhost:27017/"),
                name=os.getenv("DB_NAME", "gov_website"),
                enabled=True
            ),
            "cache": DatabaseConfig(
                type=DatabaseType.NONE,
                enabled=False
            )
        },
        middlewares={
            "cache": MiddlewareConfig(
                type=MiddlewareType.REDIS,
                uri=os.getenv("REDIS_URI", ""),
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                params={"db": int(os.getenv("REDIS_DB", "0"))},
                enabled=os.getenv("USE_REDIS", "False").lower() == "true"
            ),
            "search": MiddlewareConfig(
                type=MiddlewareType.ELASTICSEARCH,
                uri=os.getenv("ELASTICSEARCH_URI", ""),
                host=os.getenv("ELASTICSEARCH_HOST", "localhost"),
                port=int(os.getenv("ELASTICSEARCH_PORT", "9200")),
                enabled=os.getenv("USE_ELASTICSEARCH", "False").lower() == "true"
            ),
            "vector": MiddlewareConfig(
                type=MiddlewareType(os.getenv("VECTOR_DB_TYPE", "none")),
                uri=os.getenv("VECTOR_DB_URI", ""),
                api_key=os.getenv("VECTOR_DB_API_KEY", ""),
                params=json.loads(os.getenv("VECTOR_DB_PARAMS", "{}")),
                enabled=os.getenv("USE_VECTOR_DB", "False").lower() == "true"
            )
        },
        llm_services={
            "primary": LLMServiceConfig(
                type=LLMServiceType(os.getenv("LLM_SERVICE_TYPE", "openai")),
                api_key=os.getenv("OPENAI_API_KEY", ""),
                uri=os.getenv("LLM_SERVICE_URI", ""),
                model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
                params=json.loads(os.getenv("LLM_PARAMS", "{}")),
                enabled=os.getenv("USE_LLM", "False").lower() == "true"
            ),
            "local": LLMServiceConfig(
                type=LLMServiceType.OLLAMA,
                uri=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                model=os.getenv("OLLAMA_MODEL", "llama2"),
                enabled=os.getenv("USE_LOCAL_LLM", "False").lower() == "true"
            )
        }
    )


# 全局应用配置
_app_config: Optional[AppConfig] = None


def get_app_config() -> AppConfig:
    """获取应用配置"""
    global _app_config
    if _app_config is None:
        # 从环境变量或配置文件加载
        config_file = get_config_file_path()
        if os.path.exists(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                if config_file.endswith(".json"):
                    config_data = json.load(f)
                elif config_file.endswith(".yaml") or config_file.endswith(".yml"):
                    config_data = yaml.safe_load(f)
                elif config_file.endswith(".toml") and tomllib:
                    config_data = tomllib.load(f)
                else:
                    raise ValueError(f"不支持的配置文件格式：{config_file}")
                _app_config = AppConfig.parse_obj(config_data)
        else:
            # 如果没有找到配置文件，创建默认配置文件
            _app_config = get_default_config()
            create_default_config_file(config_file)
            print(f"已创建默认配置文件：{config_file}")
    
    return _app_config


def get_db_config(name: str = "primary") -> DatabaseConfig:
    """获取指定数据库配置"""
    config = get_app_config()
    if name not in config.databases:
        raise ValueError(f"数据库配置 '{name}' 不存在")
    return config.databases[name]


def get_middleware_config(name: str) -> Optional[MiddlewareConfig]:
    """获取指定中间件配置"""
    config = get_app_config()
    if name not in config.middlewares:
        return None
    return config.middlewares[name]


def get_llm_service_config(name: str = "primary") -> Optional[LLMServiceConfig]:
    """获取指定LLM服务配置"""
    config = get_app_config()
    if name not in config.llm_services:
        return None
    return config.llm_services[name]


def update_config(new_config: AppConfig) -> None:
    """更新应用配置"""
    global _app_config
    _app_config = new_config
    
    # 可以选择保存到文件
    config_file = get_config_file_path()
    if config_file:
        with open(config_file, "w", encoding="utf-8") as f:
            if config_file.endswith(".json"):
                f.write(new_config.json(indent=2, ensure_ascii=False))
            elif config_file.endswith(".yaml") or config_file.endswith(".yml"):
                yaml.dump(new_config.dict(), f, indent=2, allow_unicode=True, sort_keys=False)
            elif config_file.endswith(".toml"):
                if tomli_w:
                    f.write(tomli_w.dumps(new_config.dict()))
                else:
                    # 如果没有tomli_w库，使用JSON格式
                    f.write(new_config.json(indent=2, ensure_ascii=False))
                    print(f"警告: 没有tomli_w库，使用JSON格式")
            else:
                raise ValueError(f"不支持的配置文件格式：{config_file}")


# 创建默认配置文件
def create_default_config_file(file_path: str = None) -> str:
    """创建默认配置文件"""
    if file_path is None:
        file_path = get_config_file_path()
    
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 获取默认配置
    default_config = get_default_config()
    
    # 写入配置文件
    with open(file_path, "w", encoding="utf-8") as f:
        if file_path.endswith(".json"):
            f.write(default_config.json(indent=2, ensure_ascii=False))
        elif file_path.endswith(".yaml") or file_path.endswith(".yml"):
            yaml.dump(default_config.dict(), f, indent=2, allow_unicode=True, sort_keys=False)
        elif file_path.endswith(".toml"):
            if tomli_w:
                f.write(tomli_w.dumps(default_config.dict()))
            else:
                # 如果没有tomli_w库，使用YAML格式
                f.write(yaml.dump(default_config.dict(), indent=2, allow_unicode=True, sort_keys=False))
                print(f"警告: 没有tomli_w库，使用YAML格式")
                # 修改文件后缀为.yaml
                new_path = file_path.rsplit(".", 1)[0] + ".yaml"
                os.rename(file_path, new_path)
                file_path = new_path
        else:
            # 默认使用YAML格式
            yaml.dump(default_config.dict(), f, indent=2, allow_unicode=True, sort_keys=False)
    
    return file_path

import os
import json
import yaml
import toml
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from pydantic import BaseModel, Field
from utils.logger import get_db_logger

# 获取数据库日志记录器
logger = get_db_logger()


# 数据库类型枚举
class DatabaseType(str, Enum):
    MONGODB = "mongodb"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    OTHER = "other"


# 中间件服务类型枚举
class MiddlewareType(str, Enum):
    REDIS = "redis"
    ELASTICSEARCH = "elasticsearch"
    MILVUS = "milvus"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    PINECONE = "pinecone"
    MINIO = "minio"  # 对象存储
    OTHER = "other"


# LLM服务类型枚举
class LLMServiceType(str, Enum):
    OPENAI = "openai"
    AZURE = "azure"
    CUSTOM = "custom"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    OTHER = "other"


# 数据库配置模型
class DatabaseConfig(BaseModel):
    type: DatabaseType
    name: Optional[str] = ""
    uri: Optional[str] = ""
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    params: Dict[str, Any] = {}
    enabled: bool = True
    description: Optional[str] = None


# 中间件服务配置模型
class MiddlewareConfig(BaseModel):
    type: MiddlewareType
    name: Optional[str] = ""
    uri: Optional[str] = ""
    host: Optional[str] = None
    port: Optional[int] = None
    api_key: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    params: Dict[str, Any] = {}
    enabled: bool = True
    description: Optional[str] = None


# LLM服务配置模型
class LLMServiceConfig(BaseModel):
    type: LLMServiceType
    name: Optional[str] = ""
    uri: Optional[str] = None  # 对于第三方兼容OpenAI的API
    api_key: Optional[str] = ""
    api_base: Optional[str] = None  # 用于Azure OpenAI
    api_version: Optional[str] = None  # 用于Azure OpenAI
    organization: Optional[str] = None  # 用于OpenAI组织
    deployment_name: Optional[str] = None  # 用于Azure部署名称
    enabled: bool = True
    models: List[str] = []
    default_model: Optional[str] = None
    is_openai_compatible: bool = False  # 第三方API是否兼容OpenAI接口
    params: Dict[str, Any] = {}  # API请求的默认参数


# 应用配置模型
class AppConfig(BaseModel):
    app_name: str = "LLM Spider"
    version: Optional[str] = "1.0.0"
    api_host: Optional[str] = "0.0.0.0"
    api_port: Optional[int] = 8000
    debug: bool = False
    log_level: Optional[str] = "info"
    databases: Dict[str, DatabaseConfig] = {}
    middlewares: Dict[str, MiddlewareConfig] = {}
    llm_services: Dict[str, LLMServiceConfig] = {}


# 全局应用配置实例
_app_config: Optional[AppConfig] = None


def load_config_from_file(file_path: str) -> Dict[str, Any]:
    """从配置文件加载应用配置"""
    if not os.path.exists(file_path):
        logger.error(f"配置文件不存在: {file_path}")
        raise FileNotFoundError(f"配置文件不存在: {file_path}")
    
    extension = os.path.splitext(file_path)[-1].lower()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            logger.info(f"加载配置文件: {file_path}")
            if extension == ".json":
                return json.load(f)
            elif extension in [".yml", ".yaml"]:
                return yaml.safe_load(f)
            elif extension == ".toml":
                return toml.load(f)
            else:
                logger.error(f"不支持的配置文件格式: {extension}")
                raise ValueError(f"不支持的配置文件格式: {extension}，支持的格式有: .json, .yaml, .yml, .toml")
    except Exception as e:
        logger.error(f"加载配置文件失败: {file_path}, 错误: {str(e)}", exc_info=True)
        raise ValueError(f"加载配置文件失败: {str(e)}")


def save_config_to_file(config: Dict[str, Any], file_path: str) -> None:
    """保存应用配置到文件"""
    extension = os.path.splitext(file_path)[-1].lower()
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            logger.info(f"保存配置到文件: {file_path}")
            if extension == ".json":
                json.dump(config, f, ensure_ascii=False, indent=4)
            elif extension in [".yml", ".yaml"]:
                yaml.dump(config, f, allow_unicode=True, sort_keys=False)
            elif extension == ".toml":
                toml.dump(config, f)
            else:
                logger.error(f"不支持的配置文件格式: {extension}")
                raise ValueError(f"不支持的配置文件格式: {extension}，支持的格式有: .json, .yaml, .yml, .toml")
    except Exception as e:
        logger.error(f"保存配置文件失败: {file_path}, 错误: {str(e)}", exc_info=True)
        raise ValueError(f"保存配置文件失败: {str(e)}")


def initialize_app_config(config_path: str = None) -> AppConfig:
    """初始化应用配置"""
    global _app_config
    
    if config_path is None:
        # 使用默认配置文件路径
        config_dir = os.path.dirname(os.path.abspath(__file__))
        for ext in [".yaml", ".yml", ".json", ".toml"]:
            default_path = os.path.join(config_dir, f"config{ext}")
            if os.path.exists(default_path):
                config_path = default_path
                break
    
    if config_path and os.path.exists(config_path):
        logger.info(f"初始化应用配置，从文件加载: {config_path}")
        config_data = load_config_from_file(config_path)
        _app_config = AppConfig(**config_data)
    else:
        logger.warning("未找到配置文件，使用默认空配置")
        _app_config = AppConfig()
    
    # 记录已加载的配置信息
    db_count = len(_app_config.databases)
    middleware_count = len(_app_config.middlewares)
    llm_count = len(_app_config.llm_services)
    logger.info(f"应用配置初始化完成: {db_count}个数据库, {middleware_count}个中间件, {llm_count}个LLM服务")
    
    return _app_config


def get_app_config() -> AppConfig:
    """获取应用配置"""
    global _app_config
    if _app_config is None:
        logger.debug("应用配置未初始化，正在初始化默认配置")
        return initialize_app_config()
    return _app_config


def update_app_config(config: AppConfig) -> None:
    """更新应用配置"""
    global _app_config
    logger.info("更新应用配置")
    _app_config = config


def get_database_config(name: str) -> Optional[DatabaseConfig]:
    """获取数据库配置"""
    config = get_app_config()
    if name in config.databases:
        logger.debug(f"获取数据库配置: {name}")
        return config.databases[name]
    logger.warning(f"数据库配置不存在: {name}")
    return None


def get_all_database_configs() -> Dict[str, DatabaseConfig]:
    """获取所有数据库配置"""
    logger.debug("获取所有数据库配置")
    return get_app_config().databases


def update_database_config(name: str, config: DatabaseConfig) -> None:
    """更新数据库配置"""
    app_config = get_app_config()
    logger.info(f"更新数据库配置: {name}, 类型: {config.type}")
    app_config.databases[name] = config


def delete_database_config(name: str) -> bool:
    """删除数据库配置"""
    app_config = get_app_config()
    if name in app_config.databases:
        logger.info(f"删除数据库配置: {name}")
        del app_config.databases[name]
        return True
    logger.warning(f"尝试删除不存在的数据库配置: {name}")
    return False


def get_middleware_config(name: str) -> Optional[MiddlewareConfig]:
    """获取中间件服务配置"""
    config = get_app_config()
    if name in config.middlewares:
        logger.debug(f"获取中间件配置: {name}")
        return config.middlewares[name]
    logger.warning(f"中间件配置不存在: {name}")
    return None


def get_all_middleware_configs() -> Dict[str, MiddlewareConfig]:
    """获取所有中间件服务配置"""
    logger.debug("获取所有中间件配置")
    return get_app_config().middlewares


def update_middleware_config(name: str, config: MiddlewareConfig) -> None:
    """更新中间件服务配置"""
    app_config = get_app_config()
    logger.info(f"更新中间件配置: {name}, 类型: {config.type}")
    app_config.middlewares[name] = config


def delete_middleware_config(name: str) -> bool:
    """删除中间件服务配置"""
    app_config = get_app_config()
    if name in app_config.middlewares:
        logger.info(f"删除中间件配置: {name}")
        del app_config.middlewares[name]
        return True
    logger.warning(f"尝试删除不存在的中间件配置: {name}")
    return False


def get_llm_service_config(name: str) -> Optional[LLMServiceConfig]:
    """获取LLM服务配置"""
    config = get_app_config()
    if name in config.llm_services:
        logger.debug(f"获取LLM服务配置: {name}")
        return config.llm_services[name]
    logger.warning(f"LLM服务配置不存在: {name}")
    return None


def get_all_llm_service_configs() -> Dict[str, LLMServiceConfig]:
    """获取所有LLM服务配置"""
    logger.debug("获取所有LLM服务配置")
    return get_app_config().llm_services


def update_llm_service_config(name: str, config: LLMServiceConfig) -> None:
    """更新LLM服务配置"""
    app_config = get_app_config()
    logger.info(f"更新LLM服务配置: {name}, 类型: {config.type}, 已启用: {config.enabled}")
    app_config.llm_services[name] = config


def delete_llm_service_config(name: str) -> bool:
    """删除LLM服务配置"""
    app_config = get_app_config()
    if name in app_config.llm_services:
        logger.info(f"删除LLM服务配置: {name}")
        del app_config.llm_services[name]
        return True
    logger.warning(f"尝试删除不存在的LLM服务配置: {name}")
    return False
