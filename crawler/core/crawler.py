import asyncio
import os
from datetime import datetime
from urllib.parse import urlparse, urljoin
from typing import List, Dict, Set, Any, Optional, Tuple
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai import BrowserConfig, CrawlerRunConfig, CacheMode

from config.logging_config import setup_logging
from config.settings import MONGODB_URI, DB_NAME, USE_LLM
from crawler.core.utils import normalize_link, should_follow, is_article_page
from crawler.core.extraction import ContentExtractor
from crawler.models.article import GovArticle
from db.connection import get_sync_db

logger = setup_logging("GovCrawler")


class GovWebsiteCrawler:
    def __init__(self, base_url, mongodb_uri=MONGODB_URI, db_name=DB_NAME, use_llm=USE_LLM, llm_config=None,
                 max_concurrent_tasks=10):
        """Initialize the crawler

        Args:
            base_url: Main site URL
            mongodb_uri: MongoDB connection URI
            db_name: Database name
            use_llm: Whether to use LLM for content extraction
            llm_config: LLM configuration options
            max_concurrent_tasks: Maximum number of concurrent crawling tasks
        """
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.subdomains = set()
        self.visited_urls = set()
        self.article_count = 0
        self.start_time = None
        self.use_llm = use_llm
        self.llm_config = llm_config
        self.max_concurrent_tasks = max_concurrent_tasks

        # Set up LLM extraction strategy
        self.llm_strategy = None
        if use_llm and llm_config:
            self.llm_strategy = self._create_llm_strategy(llm_config)

        # Connect to MongoDB
        self.db, self.client = get_sync_db()
        self.articles = self.db.articles
        self.urls = self.db.urls
        self.stats = self.db.stats
        self.crawler_configs = self.db.crawler_configs

        # Initialize content extractor
        self.extractor = ContentExtractor(self.llm_strategy)

        # Initialize crawler status
        self.status = {
            "status": "idle",
            "start_time": None,
            "end_time": None,
            "visited_urls": 0,
            "articles_found": 0,
            "subdomains_found": 0,
            "current_url": None,
            "active_tasks": 0
        }

    def _create_llm_strategy(self, llm_config):
        """Create LLM extraction strategy based on configuration"""
        provider = llm_config.get("provider")
        instruction = llm_config.get("instruction",
                                     "从政府网站文章中提取标题、发布日期、发布部门和正文内容。如果附件存在，也提取附件信息。")

        if provider == "openai":
            config = llm_config.get("openai_config", {})
            api_key = config.get("api_key")
            model = config.get("model", "gpt-3.5-turbo")

            if not api_key:
                logger.warning("OpenAI API key not provided, skipping LLM extraction")
                return None

            return LLMExtractionStrategy(
                provider=f"openai/{model}",
                api_token=api_key,
                schema=GovArticle.schema(),
                extraction_type="schema",
                instruction=instruction
            )

        elif provider == "ollama":
            config = llm_config.get("ollama_config", {})
            base_url = config.get("base_url", "http://localhost:11434")
            model = config.get("model", "llama2")

            return LLMExtractionStrategy(
                provider=f"ollama/{model}",
                base_url=base_url,
                schema=GovArticle.schema(),
                extraction_type="schema",
                instruction=instruction
            )

        elif provider == "custom":
            config = llm_config.get("custom_model_config", {})
            url = config.get("url")
            api_key = config.get("api_key")
            headers = config.get("headers", {})

            if not url:
                logger.warning("Custom model URL not provided, skipping LLM extraction")
                return None

            return LLMExtractionStrategy(
                provider="custom",
                api_token=api_key,
                schema=GovArticle.schema(),
                extraction_type="schema",
                instruction=instruction,
                base_url=url,
                headers=headers
            )

        else:
            logger.warning(f"Unsupported LLM provider: {provider}")
            return None

    async def discover_subdomains(self) -> List[str]:
        """Discover all subdomains of the main site"""
        logger.info(f"Starting subdomain discovery, main site: {self.base_url}")
        self.status["status"] = "discovering_subdomains"

        browser_config = BrowserConfig(verbose=True)
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(
                url=self.base_url,
                config=CrawlerRunConfig(extract_links=True, cache_mode=CacheMode.BYPASS)
            )

            # Parse all links to find subdomains
            if result and result.links:
                for link in result.links:
                    try:
                        parsed = urlparse(link)
                        if parsed.netloc and self.domain in parsed.netloc and parsed.netloc != self.domain:
                            self.subdomains.add(parsed.netloc)
                            # Store subdomain info
                            self.urls.update_one(
                                {"url": f"https://{parsed.netloc}/"},
                                {"$set": {
                                    "url": f"https://{parsed.netloc}/",
                                    "type": "subdomain",
                                    "discovered_at": datetime.now(),
                                    "crawled": False
                                }},
                                upsert=True
                            )
                    except Exception as e:
                        logger.error(f"Error parsing link: {link}, error: {str(e)}")

            self.status["subdomains_found"] = len(self.subdomains)
            logger.info(f"Found {len(self.subdomains)} subdomains")
            logger.info(f"Subdomain list: {self.subdomains}")

            # Add main site to crawl queue
            self.urls.update_one(
                {"url": self.base_url},
                {"$set": {
                    "url": self.base_url,
                    "type": "main_domain",
                    "discovered_at": datetime.now(),
                    "crawled": False
                }},
                upsert=True
            )

            return list(self.subdomains)

    async def crawl_site(self, site_url: str) -> int:
        """Crawl a single site with all its pages

        Args:
            site_url: The site URL to crawl
        """
        logger.info(f"Starting to crawl site: {site_url}")
        self.status["current_url"] = site_url

        # Queue to store URLs to crawl
        to_visit = [site_url]
        local_visited = set()

        # Semaphore to control concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

        # Store active tasks
        active_tasks = set()

        # Track current URLs being processed
        current_urls = set()

        browser_config = BrowserConfig(verbose=True)
        async with AsyncWebCrawler(config=browser_config) as crawler:
            # Define the crawl function for a single URL
            async def process_url(url):
                async with semaphore:
                    self.status["active_tasks"] = len(active_tasks)

                    # Add to current processing URLs
                    current_urls.add(url)
                    self.status["current_url"] = f"Processing {len(current_urls)} URLs, including: {url}"

                    try:
                        # Use Crawl4AI to fetch the current page
                        run_config = CrawlerRunConfig(
                            extract_metadata=True,
                            extract_links=True,
                            extract_images=True,
                            download_attachments=True,
                            cache_mode=CacheMode.BYPASS
                        )

                        result = await crawler.arun(
                            url=url,
                            config=run_config
                        )

                        local_visited.add(url)
                        self.visited_urls.add(url)
                        self.status["visited_urls"] = len(self.visited_urls)

                        # Update URL status to crawled
                        self.urls.update_one(
                            {"url": url},
                            {"$set": {
                                "crawled": True,
                                "crawled_at": datetime.now()
                            }}
                        )

                        # Check if this is an article page
                        if is_article_page(url, result.html, result.metadata):
                            # Extract and store article content
                            article_data = await self.extractor.extract_article_data(url, result)
                            if article_data:
                                self.articles.insert_one(article_data)
                                self.article_count += 1
                                self.status["articles_found"] = self.article_count
                                logger.info(f"Found article: {article_data.get('title', 'No title')}")

                        # Extract all links from the page and add to queue
                        new_urls = []
                        if result.links:
                            for link in result.links:
                                try:
                                    # Normalize the link
                                    normalized_link = normalize_link(url, link)
                                    if normalized_link and should_follow(normalized_link, self.domain, self.subdomains,
                                                                         self.visited_urls) and normalized_link not in local_visited and normalized_link not in to_visit and normalized_link not in current_urls:
                                        # Add to queue
                                        new_urls.append(normalized_link)
                                        # Store link info
                                        self.urls.update_one(
                                            {"url": normalized_link},
                                            {"$set": {
                                                "url": normalized_link,
                                                "source_url": url,
                                                "discovered_at": datetime.now(),
                                                "crawled": False
                                            }},
                                            upsert=True
                                        )
                                except Exception as e:
                                    logger.error(f"Error processing link: {link}, error: {str(e)}")

                        # Return new URLs to add to the queue
                        return new_urls

                    except Exception as e:
                        logger.error(f"Error crawling page: {url}, error: {str(e)}")
                        return []

                    finally:
                        # Remove from current processing URLs
                        current_urls.remove(url)

                        # Add delay to avoid too frequent requests
                        await asyncio.sleep(1)

            # Process URLs in batches
            while to_visit:
                # Create new tasks up to max concurrency
                while to_visit and len(active_tasks) < self.max_concurrent_tasks:
                    url = to_visit.pop(0)

                    # Skip already visited URLs
                    if url in local_visited or url in self.visited_urls or url in current_urls:
                        continue

                    # Create task for this URL
                    task = asyncio.create_task(process_url(url))
                    active_tasks.add(task)

                    # Setup callback to handle task completion
                    def task_done(t):
                        active_tasks.remove(t)
                        if not t.cancelled() and t.exception() is None:
                            new_urls = t.result()
                            if new_urls:
                                to_visit.extend(new_urls)

                    task.add_done_callback(task_done)

                # Wait for at least one task to complete if there are active tasks
                if active_tasks:
                    done, _ = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        if task.exception():
                            logger.error(f"Task error: {task.exception()}")
                else:
                    # No more URLs to process
                    break

        logger.info(f"Site crawling completed: {site_url}, crawled {len(local_visited)} pages")
        return len(local_visited)

    async def crawl_urls_batch(self, urls: List[str]) -> int:
        """Crawl a batch of URLs concurrently

        Args:
            urls: List of URLs to crawl
        """
        if not urls:
            return 0

        # Filter out already visited URLs
        urls_to_crawl = [url for url in urls if url not in self.visited_urls]
        if not urls_to_crawl:
            return 0

        logger.info(f"Starting batch crawl of {len(urls_to_crawl)} URLs")

        # Semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        tasks = []

        browser_config = BrowserConfig(verbose=True)
        async with AsyncWebCrawler(config=browser_config) as crawler:
            # Function to process a single URL
            async def process_url(url):
                async with semaphore:
                    self.status["current_url"] = url

                    try:
                        # Use Crawl4AI to fetch the page
                        run_config = CrawlerRunConfig(
                            extract_metadata=True,
                            extract_links=False,  # Don't extract links for batch crawling
                            extract_images=True,
                            download_attachments=True,
                            cache_mode=CacheMode.BYPASS
                        )

                        result = await crawler.arun(
                            url=url,
                            config=run_config
                        )

                        self.visited_urls.add(url)
                        self.status["visited_urls"] = len(self.visited_urls)

                        # Update URL status to crawled
                        self.urls.update_one(
                            {"url": url},
                            {"$set": {
                                "crawled": True,
                                "crawled_at": datetime.now()
                            }}
                        )

                        # Check if this is an article page
                        if is_article_page(url, result.html, result.metadata):
                            # Extract and store article content
                            article_data = await self.extractor.extract_article_data(url, result)
                            if article_data:
                                self.articles.insert_one(article_data)
                                self.article_count += 1
                                self.status["articles_found"] = self.article_count
                                logger.info(f"Found article: {article_data.get('title', 'No title')}")

                        # Return success
                        return True

                    except Exception as e:
                        logger.error(f"Error crawling page: {url}, error: {str(e)}")
                        return False

                    finally:
                        # Add delay to avoid too frequent requests
                        await asyncio.sleep(1)

            # Create tasks for all URLs
            for url in urls_to_crawl:
                task = asyncio.create_task(process_url(url))
                tasks.append(task)

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Count successful crawls
            successful = sum(1 for result in results if result is True)
            logger.info(f"Batch crawl completed: {successful} of {len(urls_to_crawl)} URLs crawled successfully")

            return successful

    async def start_crawling(self) -> Dict[str, Any]:
        """Start the crawling process"""
        self.start_time = datetime.now()
        self.status["status"] = "running"
        self.status["start_time"] = self.start_time.isoformat()

        logger.info(f"Crawler started at {self.start_time}")

        try:
            # Save crawler configuration
            config = {
                "base_url": self.base_url,
                "start_time": self.start_time,
                "use_llm": self.use_llm,
                "llm_config": self.llm_config,
                "max_concurrent_tasks": self.max_concurrent_tasks
            }
            self.crawler_configs.insert_one(config)

            # First discover all subdomains
            await self.discover_subdomains()

            # Crawl the main site
            logger.info(f"Starting to crawl main site: {self.base_url}")
            await self.crawl_site(self.base_url)

            # Crawl all subdomains in parallel
            if self.subdomains:
                logger.info(f"Starting to crawl {len(self.subdomains)} subdomains")
                subdomain_tasks = []

                for subdomain in self.subdomains:
                    subdomain_url = f"https://{subdomain}/"
                    logger.info(f"Starting to crawl subdomain: {subdomain_url}")
                    task = asyncio.create_task(self.crawl_site(subdomain_url))
                    subdomain_tasks.append(task)

                # Wait for all subdomain crawling tasks to complete
                await asyncio.gather(*subdomain_tasks)

            self.status["status"] = "completed"
            self.status["end_time"] = datetime.now().isoformat()

            # Record crawl statistics
            stats = {
                "start_time": self.start_time,
                "end_time": datetime.now(),
                "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
                "visited_urls": len(self.visited_urls),
                "articles_found": self.article_count,
                "subdomains_found": len(self.subdomains),
                "max_concurrent_tasks": self.max_concurrent_tasks
            }
            self.stats.insert_one(stats)

            logger.info(f"Crawler completed, statistics: {stats}")
            return stats

        except Exception as e:
            self.status["status"] = "error"
            self.status["error"] = str(e)
            logger.error(f"Crawler error: {str(e)}")
            raise

    def get_status(self) -> Dict[str, Any]:
        """Get current crawler status"""
        return self.status