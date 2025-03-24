from urllib.parse import urlparse, urljoin
import re
from datetime import datetime
from typing import List, Dict, Optional, Any

def normalize_link(base_url: str, link: str) -> Optional[str]:
    """规范化链接，处理相对路径、锚点等，确保链接格式正确"""
    # 移除锚点
    link = link.split('#')[0]

    # 忽略空链接
    if not link:
        return None

    # 忽略JavaScript链接
    if link.startswith('javascript:'):
        return None

    # 处理相对路径
    if not link.startswith(('http://', 'https://')):
        link = urljoin(base_url, link)

    return link

def should_follow(url: str, domain: str, subdomains: set, visited_urls: set) -> bool:
    """判断是否应该爬取该链接，根据域名和URL规则决定是否爬取"""
    # 已访问过的URL不再爬取
    if url in visited_urls:
        return False

    # 只爬取目标域名和子域名
    parsed = urlparse(url)
    if not (parsed.netloc == domain or parsed.netloc in subdomains):
        return False

    # 忽略一些不需要的URL模式
    ignore_patterns = [
        '/print/', '/pdf/', '/rss/', '/login', '/logout',
        '.jpg', '.jpeg', '.png', '.gif', '.css', '.js'
    ]
    for pattern in ignore_patterns:
        if pattern in url:
            return False

    return True

def extract_publish_date(html_content: str) -> Optional[str]:
    """从HTML内容中提取发布日期"""
    date_patterns = [
        r'发布日期[：:]?\s*(\d{4}-\d{1,2}-\d{1,2})',
        r'发布时间[：:]?\s*(\d{4}-\d{1,2}-\d{1,2})',
        r'(\d{4}-\d{1,2}-\d{1,2})\s*来源',
        r'(\d{4}年\d{1,2}月\d{1,2}日)'
    ]

    for pattern in date_patterns:
        match = re.search(pattern, html_content)
        if match:
            try:
                date_str = match.group(1)
                # 转换为标准格式
                if '年' in date_str:
                    date_str = date_str.replace('年', '-').replace('月', '-').replace('日', '')
                return date_str
            except Exception:
                continue

    return None

def extract_attachments(base_url: str, html_content: str) -> List[Dict[str, str]]:
    """提取页面中的附件链接"""
    attachments = []
    attachment_patterns = [
        r'href=[\'"]([^\'"]+\.(doc|docx|xls|xlsx|pdf|zip|rar))[\'"]',
        r'src=[\'"]([^\'"]+\.(doc|docx|xls|xlsx|pdf|zip|rar))[\'"]'
    ]

    for pattern in attachment_patterns:
        matches = re.findall(pattern, html_content, re.IGNORECASE)
        for match in matches:
            file_url = match[0]
            # 处理相对路径
            if not file_url.startswith(('http://', 'https://')):
                file_url = urljoin(base_url, file_url)

            file_ext = match[1]
            file_name = file_url.split('/')[-1]

            attachments.append({
                "url": file_url,
                "filename": file_name,
                "extension": file_ext
            })

    return attachments

def is_article_page(url: str, html_content: str, metadata: Dict[str, Any]) -> bool:
    """判断页面是否为文章页面"""
    # URL模式判断
    if any(pattern in url for pattern in ['/zwgk/', '/xxgk/', '/zcwj/', '/tzgg/', '/t2']):
        return True

    # 元数据判断
    if metadata and 'title' in metadata and len(metadata['title']) > 10:
        return True

    # 内容判断：检查是否包含文章特征元素
    article_indicators = [
        'article', 'content_main', 'TRS_Editor', 'zwgk_content',
        '发布日期', '发布时间', '来源', '作者', '索引号'
    ]

    for indicator in article_indicators:
        if indicator in html_content:
            return True

    return False