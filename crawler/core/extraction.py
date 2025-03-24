from bs4 import BeautifulSoup
from typing import Dict, List, Any, Optional
from datetime import datetime
import re
from urllib.parse import urlparse
from crawler.core.utils import extract_publish_date, extract_attachments, is_article_page
from config.logging_config import setup_logging
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

logger = setup_logging("Extraction")


class ContentExtractor:
    def __init__(self, llm_strategy=None):
        self.llm_strategy = llm_strategy

    async def extract_article_data(self, url: str, result: Any) -> Dict[str, Any]:
        """Extract article data from crawl result"""
        article_data = {
            "url": url,
            "crawled_at": datetime.now(),
            "domain": urlparse(url).netloc,
            "raw_html": result.html
        }

        # LLM extraction
        if self.llm_strategy:
            try:
                # Create a new AsyncWebCrawler for LLM extraction
                browser_config = BrowserConfig(verbose=True)

                # For LLM extraction, we need to use a synchronous approach since we're in an async method
                # We'll create a crawler and use it in a new task
                async with AsyncWebCrawler(config=browser_config) as crawler:
                    run_config = CrawlerRunConfig(
                        extraction_strategy=self.llm_strategy,
                        cache_mode=CacheMode.BYPASS
                    )

                    # Run the extraction
                    llm_result = await crawler.arun(
                        url=url,
                        config=run_config
                    )

                    if llm_result and llm_result.extracted_content:
                        extracted = llm_result.extracted_content

                        article_data.update({
                            "title": extracted.get("title", ""),
                            "publish_date": extracted.get("publish_date", ""),
                            "department": extracted.get("department", ""),
                            "content": extracted.get("content", ""),
                            "is_llm_extracted": True
                        })

                        # Handle attachments
                        if "attachments" in extracted and extracted["attachments"]:
                            article_data["attachments"] = extracted["attachments"]

            except Exception as e:
                logger.error(f"LLM extraction failed: {url}, error: {str(e)}")

        # If LLM extraction failed or not used, use traditional extraction
        if "title" not in article_data or not article_data["title"]:
            # Extract metadata
            if result.metadata:
                article_data.update({
                    "title": result.metadata.get('title', '无标题'),
                    "description": result.metadata.get('description', ''),
                    "keywords": result.metadata.get('keywords', ''),
                    "author": result.metadata.get('author', '')
                })

            # Use markdown content if available
            article_data["content"] = result.markdown if hasattr(result, 'markdown') else ''

            # Extract publish date
            publish_date = extract_publish_date(result.html)
            if publish_date:
                article_data["publish_date"] = publish_date

        # Extract attachments if not already done
        if "attachments" not in article_data or not article_data["attachments"]:
            article_data["attachments"] = extract_attachments(url, result.html)

        # Extract images
        article_data["images"] = []
        if hasattr(result, 'images') and result.images:
            article_data["images"] = result.images

        return article_data


class DataCleaner:
    @staticmethod
    async def clean_article(article: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Clean article data based on configuration"""
        import re
        from bs4 import BeautifulSoup

        cleaned_data = {}

        # Extract text content
        if "raw_html" in article and config.get("remove_html_tags", True):
            try:
                soup = BeautifulSoup(article["raw_html"], "html.parser")

                # Remove scripts and styles
                for script in soup(["script", "style"]):
                    script.extract()

                # Extract text
                text = soup.get_text(separator="\n", strip=True)

                # Clean extra whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = "\n".join(chunk for chunk in chunks if chunk)

                cleaned_data["cleaned_content"] = text
            except Exception as e:
                logger.error(f"HTML cleaning error: {str(e)}")

        # Extract tables
        if "raw_html" in article and config.get("extract_tables", True):
            try:
                soup = BeautifulSoup(article["raw_html"], "html.parser")
                tables = soup.find_all("table")

                extracted_tables = []
                for i, table in enumerate(tables):
                    # Extract headers
                    headers = []
                    for th in table.find_all("th"):
                        headers.append(th.get_text(strip=True))

                    # Extract rows
                    rows = []
                    for tr in table.find_all("tr"):
                        cells = []
                        for td in tr.find_all("td"):
                            cells.append(td.get_text(strip=True))
                        if cells:
                            rows.append(cells)

                    extracted_tables.append({
                        "index": i,
                        "headers": headers,
                        "rows": rows
                    })

                cleaned_data["extracted_tables"] = extracted_tables
            except Exception as e:
                logger.error(f"Table extraction error: {str(e)}")

        # Apply custom rules
        if "content" in article and config.get("custom_rules"):
            try:
                content = article["content"]
                for pattern, replacement in config.get("custom_rules", {}).items():
                    content = re.sub(pattern, replacement, content)

                cleaned_data["processed_content"] = content
            except Exception as e:
                logger.error(f"Custom rule application error: {str(e)}")

        # Mark cleaning status
        cleaned_data["cleaning_status"] = "completed"
        cleaned_data["cleaned_at"] = datetime.now()

        return cleaned_data