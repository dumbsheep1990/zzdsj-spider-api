from pydantic import BaseModel, Field, HttpUrl
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime


# LLM Provider models
class OpenAIConfig(BaseModel):
    api_key: str = Field(..., description="Your OpenAI API Key", example="sk-1234567890abcdef")
    model: str = Field("gpt-3.5-turbo", description="OpenAI model to use", example="gpt-3.5-turbo")

    class Config:
        schema_extra = {
            "example": {
                "api_key": "sk-1234567890abcdef",
                "model": "gpt-3.5-turbo"
            }
        }


class OllamaConfig(BaseModel):
    base_url: str = Field("http://localhost:11434", description="Ollama server URL", example="http://localhost:11434")
    model: str = Field("llama2", description="Ollama model to use", example="llama2")

    class Config:
        schema_extra = {
            "example": {
                "base_url": "http://localhost:11434",
                "model": "llama2"
            }
        }


class CustomModelConfig(BaseModel):
    url: str = Field(..., description="Custom model API URL", example="https://api.example.com/v1/completions")
    api_key: Optional[str] = Field(None, description="API key if required", example="api_123456789")
    headers: Optional[Dict[str, str]] = Field(None, description="Custom headers for the request",
                                              example={"Content-Type": "application/json"})

    class Config:
        schema_extra = {
            "example": {
                "url": "https://api.example.com/v1/completions",
                "api_key": "api_123456789",
                "headers": {"Content-Type": "application/json"}
            }
        }


class LLMConfig(BaseModel):
    provider: Literal["openai", "ollama", "custom"] = Field(..., description="LLM provider to use")
    openai_config: Optional[OpenAIConfig] = Field(None, description="OpenAI configuration")
    ollama_config: Optional[OllamaConfig] = Field(None, description="Ollama configuration")
    custom_model_config: Optional[CustomModelConfig] = Field(None, description="Custom model configuration")
    instruction: str = Field("从政府网站文章中提取标题、发布日期、发布部门和正文内容。如果附件存在，也提取附件信息。",
                             description="Instruction for LLM extraction")

    class Config:
        schema_extra = {
            "example": {
                "provider": "openai",
                "openai_config": {
                    "api_key": "sk-1234567890abcdef",
                    "model": "gpt-3.5-turbo"
                },
                "instruction": "从政府网站文章中提取标题、发布日期、发布部门和正文内容。如果附件存在，也提取附件信息。"
            }
        }


# API models
class CrawlerConfig(BaseModel):
    base_url: str = Field(..., description="Base URL to crawl", example="https://www.example.gov.cn/")
    max_pages: Optional[int] = Field(None, description="Maximum number of pages to crawl", example=1000)
    max_depth: Optional[int] = Field(None, description="Maximum crawl depth", example=5)
    include_subdomains: bool = Field(True, description="Whether to crawl subdomains")
    crawl_interval: float = Field(1.0, description="Interval between requests in seconds", example=1.0)
    use_llm: bool = Field(False, description="Whether to use LLM for content extraction")
    llm_config: Optional[LLMConfig] = Field(None, description="LLM configuration")
    max_concurrent_tasks: int = Field(10, ge=1, le=100, description="Maximum number of concurrent crawling tasks")

    class Config:
        schema_extra = {
            "example": {
                "base_url": "https://www.example.gov.cn/",
                "max_pages": 1000,
                "max_depth": 5,
                "include_subdomains": True,
                "crawl_interval": 1.0,
                "use_llm": True,
                "max_concurrent_tasks": 20,
                "llm_config": {
                    "provider": "openai",
                    "openai_config": {
                        "api_key": "sk-1234567890abcdef",
                        "model": "gpt-3.5-turbo"
                    },
                    "instruction": "从政府网站文章中提取标题、发布日期、发布部门和正文内容。如果附件存在，也提取附件信息。"
                }
            }
        }

class CleaningConfig(BaseModel):
    remove_html_tags: bool = Field(True, description="Whether to remove HTML tags")
    extract_tables: bool = Field(True, description="Whether to extract tables")
    extract_attachments: bool = Field(True, description="Whether to extract attachments")
    extract_images: bool = Field(True, description="Whether to extract images")
    custom_rules: Optional[Dict[str, str]] = Field(None, description="Custom regex rules for text processing",
                                                   example={r"\s+": " "})

    class Config:
        schema_extra = {
            "example": {
                "remove_html_tags": True,
                "extract_tables": True,
                "extract_attachments": True,
                "extract_images": True,
                "custom_rules": {
                    r"\s+": " ",
                    r"\[.*?\]": ""
                }
            }
        }


class CrawlerStatus(BaseModel):
    status: str = Field(..., description="Current crawler status", example="running")
    start_time: Optional[str] = Field(None, description="Start time of the crawler", example="2024-03-24T10:00:00")
    end_time: Optional[str] = Field(None, description="End time of the crawler", example="2024-03-24T11:30:00")
    visited_urls: int = Field(..., description="Number of URLs visited", example=150)
    articles_found: int = Field(..., description="Number of articles found", example=42)
    subdomains_found: int = Field(..., description="Number of subdomains found", example=3)
    current_url: Optional[str] = Field(None, description="Currently processing URL",
                                       example="https://www.example.gov.cn/news/")
    error: Optional[str] = Field(None, description="Error message if any", example="Connection timeout")

    class Config:
        schema_extra = {
            "example": {
                "status": "running",
                "start_time": "2024-03-24T10:00:00",
                "visited_urls": 150,
                "articles_found": 42,
                "subdomains_found": 3,
                "current_url": "https://www.example.gov.cn/news/"
            }
        }


class ArticleFilter(BaseModel):
    domain: Optional[str] = Field(None, description="Filter by domain", example="www.example.gov.cn")
    start_date: Optional[str] = Field(None, description="Filter by publish date (start)", example="2024-01-01")
    end_date: Optional[str] = Field(None, description="Filter by publish date (end)", example="2024-03-24")
    keyword: Optional[str] = Field(None, description="Filter by keyword in title or content", example="政策")

    class Config:
        schema_extra = {
            "example": {
                "domain": "www.example.gov.cn",
                "start_date": "2024-01-01",
                "end_date": "2024-03-24",
                "keyword": "政策"
            }
        }


class PaginatedResponse(BaseModel):
    total: int = Field(..., description="Total number of items", example=150)
    page: int = Field(..., description="Current page number", example=1)
    page_size: int = Field(..., description="Number of items per page", example=20)
    data: List[Dict[str, Any]] = Field(..., description="List of items")

    class Config:
        schema_extra = {
            "example": {
                "total": 150,
                "page": 1,
                "page_size": 20,
                "data": [
                    {
                        "_id": "6072f1b12c723a8c9d89a123",
                        "title": "关于推进智慧城市建设的通知",
                        "publish_date": "2024-03-15T00:00:00",
                        "department": "发展和改革委员会",
                        "content": "为深入贯彻落实党的二十大精神...",
                        "url": "https://www.example.gov.cn/news/123.html",
                        "domain": "www.example.gov.cn"
                    }
                ]
            }
        }


class LLMModelInfo(BaseModel):
    providers: List[Dict[str, Any]] = Field(..., description="List of LLM providers and their models")

    class Config:
        schema_extra = {
            "example": {
                "providers": [
                    {
                        "name": "openai",
                        "display_name": "OpenAI",
                        "models": [
                            {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo"},
                            {"id": "gpt-4", "name": "GPT-4"}
                        ],
                        "requires_api_key": True
                    },
                    {
                        "name": "ollama",
                        "display_name": "Ollama",
                        "models": [
                            {"id": "llama2", "name": "Llama 2"},
                            {"id": "mistral", "name": "Mistral"}
                        ],
                        "requires_api_key": False
                    }
                ]
            }
        }

# Add these new models to app/api/models/schemas.py

class ScheduleConfig(BaseModel):
    cron: str = Field(..., description="Cron expression for scheduling", example="0 0 * * *")
    config_id: str = Field(..., description="ID of saved crawler configuration", example="6072f1b12c723a8c9d89a123")
    name: Optional[str] = Field(None, description="Schedule name", example="Daily Crawl")
    enabled: bool = Field(True, description="Whether the schedule is enabled")

    class Config:
        schema_extra = {
            "example": {
                "cron": "0 0 * * *",
                "config_id": "6072f1b12c723a8c9d89a123",
                "name": "Daily Crawl",
                "enabled": True
            }
        }

class ScheduleInfo(BaseModel):
    id: str = Field(..., description="Schedule ID")
    cron: str = Field(..., description="Cron expression")
    config_id: str = Field(..., description="Configuration ID")
    name: Optional[str] = Field(None, description="Schedule name")
    enabled: bool = Field(..., description="Whether the schedule is enabled")
    next_run: Optional[str] = Field(None, description="Next scheduled run time")
    last_run: Optional[str] = Field(None, description="Last run time")
    created_at: str = Field(..., description="Creation time")

    class Config:
        schema_extra = {
            "example": {
                "id": "6072f1b12c723a8c9d89a124",
                "cron": "0 0 * * *",
                "config_id": "6072f1b12c723a8c9d89a123",
                "name": "Daily Crawl",
                "enabled": True,
                "next_run": "2024-03-25T00:00:00",
                "last_run": "2024-03-24T00:00:00",
                "created_at": "2024-03-20T15:30:00"
            }
        }

class DashboardSummary(BaseModel):
    total_articles: int = Field(..., description="Total number of articles")
    total_subdomains: int = Field(..., description="Total number of subdomains")
    total_crawled_urls: int = Field(..., description="Total number of crawled URLs")
    recent_articles: List[Dict[str, Any]] = Field(..., description="Recent articles")
    latest_run: Optional[Dict[str, Any]] = Field(None, description="Latest crawler run")

    class Config:
        schema_extra = {
            "example": {
                "total_articles": 1250,
                "total_subdomains": 15,
                "total_crawled_urls": 5000,
                "recent_articles": [
                    {
                        "_id": "6072f1b12c723a8c9d89a123",
                        "title": "关于推进智慧城市建设的通知",
                        "publish_date": "2024-03-15T00:00:00",
                        "domain": "www.example.gov.cn"
                    }
                ],
                "latest_run": {
                    "start_time": "2024-03-24T10:00:00",
                    "end_time": "2024-03-24T11:30:00",
                    "duration_seconds": 5400,
                    "visited_urls": 500,
                    "articles_found": 42
                }
            }
        }

class TestLLMExtractionConfig(BaseModel):
    provider: Literal["openai", "ollama", "custom"] = Field(..., description="LLM provider")
    openai_config: Optional[OpenAIConfig] = Field(None, description="OpenAI configuration")
    ollama_config: Optional[OllamaConfig] = Field(None, description="Ollama configuration")
    custom_model_config: Optional[CustomModelConfig] = Field(None, description="Custom model configuration")
    url: str = Field(..., description="URL to test extraction on", example="https://www.example.gov.cn/news/sample.html")

    class Config:
        schema_extra = {
            "example": {
                "provider": "openai",
                "openai_config": {
                    "api_key": "sk-1234567890abcdef",
                    "model": "gpt-3.5-turbo"
                },
                "url": "https://www.example.gov.cn/news/sample.html"
            }
        }

class CleaningStatus(BaseModel):
    status: str = Field(..., description="Cleaning process status", example="running")
    started_at: Optional[str] = Field(None, description="Start time")
    completed_at: Optional[str] = Field(None, description="Completion time")
    total_articles: int = Field(..., description="Total articles to clean")
    processed_articles: int = Field(..., description="Number of processed articles")
    current_article_id: Optional[str] = Field(None, description="Currently processing article ID")
    error: Optional[str] = Field(None, description="Error message if any")

    class Config:
        schema_extra = {
            "example": {
                "status": "running",
                "started_at": "2024-03-24T12:00:00",
                "total_articles": 1250,
                "processed_articles": 500,
                "current_article_id": "6072f1b12c723a8c9d89a123"
            }
        }

# AI Extraction API models
class AIExtractConfig(BaseModel):
    url: Optional[str] = Field(None, description="URL to extract content from")
    html_content: Optional[str] = Field(None, description="HTML content to extract from")
    ai_model: str = Field("gpt-3.5-turbo", description="AI model to use for extraction")
    custom_prompt: Optional[str] = Field(None, description="Custom extraction prompt")
    extraction_schema: Optional[Dict[str, Any]] = Field(None, description="Extraction schema")

    class Config:
        schema_extra = {
            "example": {
                "url": "https://www.example.gov.cn/news/123.html",
                "ai_model": "gpt-3.5-turbo",
                "custom_prompt": "从网页中提取标题、发布日期、发布部门和正文内容。"
            }
        }


class AIExtractTemplate(BaseModel):
    name: str = Field(..., description="Template name")
    prompt: str = Field(..., description="Extraction prompt")
    schema: Optional[Dict[str, Any]] = Field(None, description="Extraction schema")
    description: Optional[str] = Field(None, description="Template description")

    class Config:
        schema_extra = {
            "example": {
                "name": "标准政府网站提取",
                "prompt": "从政府网站文章中提取标题、发布日期、发布部门和正文内容。",
                "description": "标准的政府网站文章提取模板"
            }
        }


class BatchExtractConfig(BaseModel):
    urls: List[str] = Field(..., description="List of URLs to extract from")
    ai_model: str = Field("gpt-3.5-turbo", description="AI model to use for extraction")
    custom_prompt: Optional[str] = Field(None, description="Custom extraction prompt")
    template_id: Optional[str] = Field(None, description="Template ID to use")

    class Config:
        schema_extra = {
            "example": {
                "urls": ["https://www.example.gov.cn/news/123.html", "https://www.example.gov.cn/news/456.html"],
                "ai_model": "gpt-3.5-turbo"
            }
        }


class BatchExtractStatus(BaseModel):
    task_id: str = Field(..., description="Batch task ID")
    status: str = Field(..., description="Task status")
    total: int = Field(..., description="Total number of URLs")
    completed: int = Field(0, description="Number of completed extractions")
    failed: int = Field(0, description="Number of failed extractions")
    created_at: str = Field(..., description="Task creation time")
    completed_at: Optional[str] = Field(None, description="Task completion time")
    error: Optional[str] = Field(None, description="Error message if any")

    class Config:
        schema_extra = {
            "example": {
                "task_id": "6072f1b12c723a8c9d89a123",
                "status": "running",
                "total": 10,
                "completed": 3,
                "failed": 0,
                "created_at": "2024-03-24T12:00:00"
            }
        }


# Custom Crawl API models
class CustomCrawlConfig(BaseModel):
    url: str = Field(..., description="URL to crawl")
    depth: int = Field(1, ge=1, le=5, description="Crawl depth")
    use_browser: bool = Field(True, description="Whether to use browser for crawling")
    follow_links: bool = Field(False, description="Whether to follow links on the page")
    extract_content: bool = Field(True, description="Whether to extract content")
    use_llm: bool = Field(False, description="Whether to use LLM for extraction")
    ai_model: Optional[str] = Field(None, description="AI model to use for extraction")

    class Config:
        schema_extra = {
            "example": {
                "url": "https://www.example.gov.cn/news/123.html",
                "depth": 1,
                "use_browser": True,
                "follow_links": False,
                "extract_content": True,
                "use_llm": False
            }
        }


class CustomCrawlResult(BaseModel):
    crawl_id: str = Field(..., description="Crawl ID")
    url: str = Field(..., description="Crawled URL")
    status: str = Field(..., description="Crawl status")
    content: Optional[Dict[str, Any]] = Field(None, description="Extracted content")
    html: Optional[str] = Field(None, description="HTML content")
    links: Optional[List[str]] = Field(None, description="Extracted links")
    created_at: str = Field(..., description="Crawl time")
    error: Optional[str] = Field(None, description="Error message if any")

    class Config:
        schema_extra = {
            "example": {
                "crawl_id": "6072f1b12c723a8c9d89a123",
                "url": "https://www.example.gov.cn/news/123.html",
                "status": "completed",
                "content": {
                    "title": "关于推进智慧城市建设的通知",
                    "publish_date": "2024-03-15T00:00:00",
                    "department": "发展和改革委员会",
                    "content": "为深入贯彻落实党的二十大精神..."
                },
                "links": ["https://www.example.gov.cn/news/456.html"],
                "created_at": "2024-03-24T12:00:00"
            }
        }


class AdvancedCrawlerConfig(BaseModel):
    user_agent: Optional[str] = Field(None, description="Custom User-Agent string")
    headers: Optional[Dict[str, str]] = Field(None, description="Custom HTTP headers")
    proxy: Optional[str] = Field(None, description="Proxy URL")
    timeout: int = Field(30, description="Request timeout in seconds")
    retry_count: int = Field(3, description="Number of retries for failed requests")
    robots_txt: bool = Field(True, description="Whether to respect robots.txt")
    javascript_support: bool = Field(True, description="Whether to enable JavaScript support")
    url_patterns: Optional[List[str]] = Field(None, description="URL patterns to include")
    exclude_patterns: Optional[List[str]] = Field(None, description="URL patterns to exclude")

    class Config:
        schema_extra = {
            "example": {
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "headers": {"Accept-Language": "zh-CN,zh;q=0.9"},
                "timeout": 30,
                "retry_count": 3,
                "robots_txt": True,
                "javascript_support": True,
                "url_patterns": [".*/news/.*", ".*/notices/.*"],
                "exclude_patterns": [".*/login/.*", ".*/search/.*"]
            }
        }


# Vector API models
class VectorizationTemplate(BaseModel):
    name: str = Field(..., description="Template name")
    description: Optional[str] = Field(None, description="Template description")
    text_fields: List[str] = Field(..., description="Text fields to vectorize")
    metadata_fields: Optional[List[str]] = Field(None, description="Metadata fields to include")
    chunk_size: Optional[int] = Field(None, description="Text chunk size for long documents")
    chunk_overlap: Optional[int] = Field(None, description="Chunk overlap for long documents")

    class Config:
        schema_extra = {
            "example": {
                "name": "标准文章向量化",
                "description": "标准的政府文章向量化模板",
                "text_fields": ["title", "content"],
                "metadata_fields": ["publish_date", "department"],
                "chunk_size": 1000,
                "chunk_overlap": 200
            }
        }


class BatchVectorizeConfig(BaseModel):
    article_ids: Optional[List[str]] = Field(None, description="List of article IDs to vectorize")
    filter: Optional[Dict[str, Any]] = Field(None, description="Filter criteria for articles")
    template_id: Optional[str] = Field(None, description="Template ID to use")
    model: Optional[str] = Field(None, description="Vector model to use")

    class Config:
        schema_extra = {
            "example": {
                "article_ids": ["6072f1b12c723a8c9d89a123", "6072f1b12c723a8c9d89a124"],
                "template_id": "6072f1b12c723a8c9d89a125"
            }
        }


class SimilaritySearchConfig(BaseModel):
    query: str = Field(..., description="Search query")
    filter: Optional[Dict[str, Any]] = Field(None, description="Filter criteria")
    top_k: int = Field(10, description="Number of results to return")
    threshold: Optional[float] = Field(None, description="Similarity threshold")

    class Config:
        schema_extra = {
            "example": {
                "query": "智慧城市建设政策",
                "filter": {"department": "发展和改革委员会"},
                "top_k": 10,
                "threshold": 0.7
            }
        }