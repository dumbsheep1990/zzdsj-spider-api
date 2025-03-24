from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Attachment(BaseModel):
    url: str
    filename: str
    extension: str

class GovArticle(BaseModel):
    title: str = Field(..., description="文章标题")
    publish_date: Optional[str] = Field(None, description="发布日期，格式为YYYY-MM-DD")
    department: Optional[str] = Field(None, description="发布部门")
    content: str = Field(..., description="文章正文内容")
    attachments: List[Attachment] = Field([], description="附件列表，包含附件名称和URL")

class Article(GovArticle):
    url: str
    domain: str
    raw_html: Optional[str] = None
    images: List[str] = []
    crawled_at: str
    is_llm_extracted: bool = False
    cleaning_status: Optional[str] = None
    cleaned_at: Optional[str] = None
    extracted_tables: Optional[List[Dict[str, Any]]] = None
    cleaned_content: Optional[str] = None
    processed_content: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "title": "关于促进经济高质量发展的通知",
                "publish_date": "2023-05-15",
                "department": "发展和改革委员会",
                "content": "为深入贯彻落实党的二十大精神...",
                "attachments": [
                    {"url": "https://example.gov.cn/files/doc123.pdf",
                     "filename": "doc123.pdf",
                     "extension": "pdf"}
                ],
                "url": "https://example.gov.cn/news/123.html",
                "domain": "example.gov.cn",
                "images": ["https://example.gov.cn/img/123.jpg"],
                "crawled_at": "2023-05-16T14:30:25",
                "is_llm_extracted": True
            }
        }