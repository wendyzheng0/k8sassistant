"""
Kubernetes 文档爬取器
"""

import os
import asyncio
import aiohttp
import aiofiles
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Set
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class K8sCrawler:
    """Kubernetes 文档爬取器"""
    
    def __init__(self, output_dir: str = "docs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Kubernetes 官方文档基础URL
        self.base_url = "https://kubernetes.io"
        self.docs_url = "https://kubernetes.io/docs"
        
        # 已访问的URL集合
        self.visited_urls: Set[str] = set()
        
        # 要爬取的页面类型
        self.target_paths = [
            "/docs/concepts/",
            "/docs/tasks/",
            "/docs/reference/",
            "/docs/setup/",
            "/docs/tutorials/"
        ]
    
    async def crawl(self, max_pages: int = 100):
        """开始爬取文档"""
        logger.info("🚀 开始爬取 Kubernetes 文档...")
        
        # 创建输出目录
        for path in self.target_paths:
            (self.output_dir / path.strip("/")).mkdir(parents=True, exist_ok=True)
        
        # 收集所有要爬取的URL
        urls_to_crawl = []
        
        async with aiohttp.ClientSession() as session:
            for path in self.target_paths:
                full_url = urljoin(self.base_url, path)
                logger.info(f"🔍 扫描目录: {full_url}")
                
                try:
                    urls = await self._get_page_urls(session, full_url)
                    urls_to_crawl.extend(urls)
                except Exception as e:
                    logger.error(f"❌ 扫描目录失败 {full_url}: {e}")
        
        # 限制爬取页面数量
        urls_to_crawl = urls_to_crawl[:max_pages]
        logger.info(f"📋 准备爬取 {len(urls_to_crawl)} 个页面")
        
        # 并发爬取页面
        semaphore = asyncio.Semaphore(10)  # 限制并发数
        
        tasks = []
        for url in urls_to_crawl:
            task = asyncio.create_task(
                self._crawl_page_with_semaphore(session, url, semaphore)
            )
            tasks.append(task)
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 统计结果
        success_count = sum(1 for r in results if r is True)
        logger.info(f"✅ 爬取完成！成功: {success_count}/{len(urls_to_crawl)}")
    
    async def _crawl_page_with_semaphore(
        self, 
        session: aiohttp.ClientSession, 
        url: str, 
        semaphore: asyncio.Semaphore
    ):
        """使用信号量控制并发的页面爬取"""
        async with semaphore:
            return await self._crawl_page(session, url)
    
    async def _get_page_urls(self, session: aiohttp.ClientSession, url: str) -> List[str]:
        """获取页面中的所有链接"""
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    return []
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                urls = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    
                    # 只处理相对链接和同域名的绝对链接
                    if href.startswith('/'):
                        full_url = urljoin(self.base_url, href)
                    elif href.startswith(self.base_url):
                        full_url = href
                    else:
                        continue
                    
                    # 过滤掉非文档页面
                    if not self._is_valid_doc_url(full_url):
                        continue
                    
                    urls.append(full_url)
                
                return list(set(urls))  # 去重
                
        except Exception as e:
            logger.error(f"❌ 获取页面链接失败 {url}: {e}")
            return []
    
    def _is_valid_doc_url(self, url: str) -> bool:
        """检查是否为有效的文档URL"""
        if not url.startswith(self.base_url):
            return False
        
        # 排除一些不需要的页面
        exclude_patterns = [
            '/search',
            '/sitemap',
            '/404',
            '/blog',
            '/community',
            '/partners',
            '/case-studies',
            '#',
            'mailto:',
            'javascript:'
        ]
        
        for pattern in exclude_patterns:
            if pattern in url:
                return False
        
        # 只包含文档相关的路径
        valid_patterns = [
            '/docs/concepts/',
            '/docs/tasks/',
            '/docs/reference/',
            '/docs/setup/',
            '/docs/tutorials/'
        ]
        
        return any(pattern in url for pattern in valid_patterns)
    
    async def _crawl_page(self, session: aiohttp.ClientSession, url: str) -> bool:
        """爬取单个页面"""
        if url in self.visited_urls:
            return False
        
        self.visited_urls.add(url)
        
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"⚠️ 页面访问失败 {url}: {response.status}")
                    return False
                
                html = await response.text()
                
                # 解析HTML
                soup = BeautifulSoup(html, 'html.parser')
                
                # 提取主要内容
                content = self._extract_content(soup)
                
                if not content.strip():
                    logger.warning(f"⚠️ 页面内容为空 {url}")
                    return False
                
                # 生成文件名
                filename = self._generate_filename(url)
                filepath = self.output_dir / filename
                
                # 保存文件
                async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
                    await f.write(f"# {soup.title.string if soup.title else 'Untitled'}\n\n")
                    await f.write(f"URL: {url}\n\n")
                    await f.write(content)
                
                logger.info(f"✅ 保存页面: {filename}")
                return True
                
        except Exception as e:
            logger.error(f"❌ 爬取页面失败 {url}: {e}")
            return False
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """提取页面主要内容"""
        # 尝试找到主要内容区域
        content_selectors = [
            'main',
            '.td-content',
            '.content',
            '#content',
            'article',
            '.main-content'
        ]
        
        content_element = None
        for selector in content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                break
        
        if not content_element:
            # 如果没有找到主要内容区域，使用body
            content_element = soup.body
        
        if not content_element:
            return ""
        
        # 移除不需要的元素
        for element in content_element.find_all(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
        
        # 提取文本内容
        text = content_element.get_text(separator='\n', strip=True)
        
        # 清理文本
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if line and len(line) > 10:  # 过滤掉太短的行
                lines.append(line)
        
        return '\n\n'.join(lines)
    
    def _generate_filename(self, url: str) -> str:
        """根据URL生成文件名"""
        parsed = urlparse(url)
        path = parsed.path.strip('/')
        
        if not path:
            path = 'index'
        
        # 替换特殊字符
        filename = path.replace('/', '_').replace('-', '_')
        
        # 添加扩展名
        if not filename.endswith('.html'):
            filename += '.html'
        
        return filename


async def main():
    """主函数"""
    crawler = K8sCrawler()
    await crawler.crawl(max_pages=50)  # 限制爬取50个页面


if __name__ == "__main__":
    asyncio.run(main())
