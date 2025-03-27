import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Dict, Any
import json
import traceback
from datetime import datetime

# 日志级别映射
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

# 默认日志格式
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 彩色日志格式（控制台输出）
COLOR_FORMAT = {
    "DEBUG": "\033[36m%(asctime)s - %(name)s - %(levelname)s - %(message)s\033[0m",  # 青色
    "INFO": "\033[32m%(asctime)s - %(name)s - %(levelname)s - %(message)s\033[0m",  # 绿色
    "WARNING": "\033[33m%(asctime)s - %(name)s - %(levelname)s - %(message)s\033[0m",  # 黄色
    "ERROR": "\033[31m%(asctime)s - %(name)s - %(levelname)s - %(message)s\033[0m",  # 红色
    "CRITICAL": "\033[1;31m%(asctime)s - %(name)s - %(levelname)s - %(message)s\033[0m"  # 加粗红色
}


class ColorFormatter(logging.Formatter):
    """彩色日志格式化器"""
    def format(self, record):
        log_fmt = COLOR_FORMAT.get(record.levelname, DEFAULT_LOG_FORMAT)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


class JsonFormatter(logging.Formatter):
    """JSON格式日志格式化器"""
    def format(self, record):
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
            "name": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }
        
        if hasattr(record, "request_id"):
            log_record["request_id"] = record.request_id
            
        if record.exc_info:
            log_record["exception"] = traceback.format_exception(*record.exc_info)
            
        # 添加额外字段
        for key, value in record.__dict__.items():
            if key not in ["args", "asctime", "created", "exc_info", "exc_text", "filename",
                           "funcName", "id", "levelname", "levelno", "lineno", "module",
                           "msecs", "message", "msg", "name", "pathname", "process",
                           "processName", "relativeCreated", "stack_info", "thread", "threadName"]:
                log_record[key] = value
                
        return json.dumps(log_record, ensure_ascii=False)


class LLMLogger:
    """LLM Spider项目日志类"""
    def __init__(self, name: str, log_level: str = "info", log_to_console: bool = True, 
                 log_to_file: bool = True, log_dir: str = None, json_format: bool = False,
                 max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5):
        """
        初始化日志记录器
        
        Args:
            name: 日志记录器名称
            log_level: 日志级别 (debug, info, warning, error, critical)
            log_to_console: 是否输出到控制台
            log_to_file: 是否输出到文件
            log_dir: 日志文件目录，如果为None则使用默认目录
            json_format: 是否使用JSON格式输出日志
            max_bytes: 日志文件最大大小（默认10MB）
            backup_count: 备份文件数量（默认5个）
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(LOG_LEVELS.get(log_level.lower(), logging.INFO))
        self.logger.propagate = False  # 避免日志重复输出
        
        # 如果已经有处理器，则跳过
        if len(self.logger.handlers) > 0:
            return
            
        # 控制台输出
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(LOG_LEVELS.get(log_level.lower(), logging.INFO))
            
            if json_format:
                console_formatter = JsonFormatter()
            else:
                console_formatter = ColorFormatter()
                
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            
        # 文件输出
        if log_to_file:
            if log_dir is None:
                # 默认日志目录在项目根目录的logs文件夹下
                base_dir = Path(__file__).parent.parent
                log_dir = os.path.join(base_dir, "logs")
                
            # 确保日志目录存在
            os.makedirs(log_dir, exist_ok=True)
            
            log_file = os.path.join(log_dir, f"{name}.log")
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8"
            )
            file_handler.setLevel(LOG_LEVELS.get(log_level.lower(), logging.INFO))
            
            if json_format:
                file_formatter = JsonFormatter()
            else:
                file_formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
                
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, msg: str, **kwargs):
        """记录调试日志"""
        self._log(logging.DEBUG, msg, **kwargs)
    
    def info(self, msg: str, **kwargs):
        """记录信息日志"""
        self._log(logging.INFO, msg, **kwargs)
        
    def warning(self, msg: str, **kwargs):
        """记录警告日志"""
        self._log(logging.WARNING, msg, **kwargs)
        
    def error(self, msg: str, **kwargs):
        """记录错误日志"""
        self._log(logging.ERROR, msg, **kwargs)
        
    def critical(self, msg: str, **kwargs):
        """记录严重错误日志"""
        self._log(logging.CRITICAL, msg, **kwargs)
        
    def exception(self, msg: str, **kwargs):
        """记录异常日志，自动包含堆栈信息"""
        self.logger.exception(msg, extra=kwargs)
        
    def _log(self, level: int, msg: str, **kwargs):
        """内部日志记录方法"""
        self.logger.log(level, msg, extra=kwargs)


# 默认日志记录器
_app_logger = None
_api_logger = None
_llm_logger = None
_db_logger = None


def get_app_logger() -> LLMLogger:
    """获取应用全局日志记录器"""
    global _app_logger
    if _app_logger is None:
        _app_logger = LLMLogger("app")
    return _app_logger


def get_api_logger() -> LLMLogger:
    """获取API日志记录器"""
    global _api_logger
    if _api_logger is None:
        _api_logger = LLMLogger("api")
    return _api_logger


def get_llm_logger() -> LLMLogger:
    """获取LLM服务日志记录器"""
    global _llm_logger
    if _llm_logger is None:
        _llm_logger = LLMLogger("llm")
    return _llm_logger


def get_db_logger() -> LLMLogger:
    """获取数据库日志记录器"""
    global _db_logger
    if _db_logger is None:
        _db_logger = LLMLogger("db")
    return _db_logger


def configure_logging(config: Dict[str, Any]):
    """从配置中配置日志系统
    
    Args:
        config: 日志配置字典，应包含以下键:
            - level: 日志级别
            - console: 是否输出到控制台
            - file: 是否输出到文件
            - json_format: 是否使用JSON格式
            - log_dir: 日志目录
    """
    global _app_logger, _api_logger, _llm_logger, _db_logger
    
    # 获取配置参数
    log_level = config.get("level", "info")
    log_to_console = config.get("console", True)
    log_to_file = config.get("file", True)
    json_format = config.get("json_format", False)
    log_dir = config.get("log_dir", None)
    
    # 重新初始化日志记录器
    _app_logger = LLMLogger("app", log_level, log_to_console, log_to_file, log_dir, json_format)
    _api_logger = LLMLogger("api", log_level, log_to_console, log_to_file, log_dir, json_format)
    _llm_logger = LLMLogger("llm", log_level, log_to_console, log_to_file, log_dir, json_format)
    _db_logger = LLMLogger("db", log_level, log_to_console, log_to_file, log_dir, json_format)
