"""
HTML Processor Step
HTML 文档处理步骤
"""

from bs4 import BeautifulSoup
import html2text

from .base import ProcessingStep, ProcessingContext
from ..utils.cleaner import clean_text, clean_html_content, clean_metadata
from ..utils.code_extractor import CodeExtractor


class HTMLProcessor(ProcessingStep):
    """
    HTML 处理器
    将 HTML 文档转换为纯文本
    """
    
    def __init__(
        self,
        extract_codes: bool = True,
        content_selector: str = "div.td-content",
        code_blocks_dir: str = None,
        skip_if_missing_selector: bool = True,
        html2text_ignore_links: bool = True,
        html2text_body_width: int = 0,
    ):
        """
        初始化 HTML 处理器
        
        Args:
            extract_codes: 是否提取代码块
            content_selector: 主要内容区域的 CSS 选择器
            code_blocks_dir: 代码块保存目录
        """
        super().__init__("HTMLProcessor")
        self.extract_codes = extract_codes
        self.content_selector = content_selector
        self.code_extractor = CodeExtractor(code_blocks_dir) if extract_codes else None
        # If a selector is specified but not found, skip the document instead of
        # falling back to extracting the entire page.
        self.skip_if_missing_selector = skip_if_missing_selector
        # html2text config
        self.html2text_ignore_links = html2text_ignore_links
        self.html2text_body_width = html2text_body_width
    
    async def process(self, context: ProcessingContext) -> ProcessingContext:
        """
        处理 HTML 文档
        
        Args:
            context: 包含文档的处理上下文
            
        Returns:
            ProcessingContext: 文档内容已转换为纯文本
        """
        self.logger.info(f"🔄 Processing {len(context.documents)} HTML documents...")
        
        processed_count = 0
        skipped_missing_selector = 0
        skipped_paths = []
        for doc in context.documents:
            try:
                # 只处理 HTML 文件
                file_type = doc.get("metadata", {}).get("file_type", "")
                if file_type not in [".html", ".htm"]:
                    continue
                
                # 处理 HTML 内容
                processed_content, extracted_codes, used_selector = self._process_html(doc["content"])
                if self.skip_if_missing_selector and self.content_selector and not used_selector:
                    skipped_missing_selector += 1
                    rel_path = doc.get("metadata", {}).get("relative_path") or doc.get("file_path", "unknown")
                    skipped_paths.append(rel_path)
                    # Mark as empty so downstream steps naturally ignore it
                    doc["content"] = ""
                    continue
                doc["content"] = processed_content
                
                # 更新元数据
                if extracted_codes:
                    doc["metadata"]["extracted_codes"] = extracted_codes
                    doc["metadata"]["code_blocks_count"] = len(extracted_codes)
                
                # 清理元数据
                doc["metadata"] = clean_metadata(doc["metadata"])
                
                processed_count += 1
                
            except Exception as e:
                context.add_error(
                    f"Failed to process HTML {doc.get('file_path', 'unknown')}: {str(e)}"
                )
        
        self.logger.info(f"✅ Processed {processed_count} HTML documents")
        if skipped_missing_selector:
            self.logger.warning(
                f"⚠️ Skipped {skipped_missing_selector} documents missing selector: {self.content_selector}"
            )
            # Log a small sample to avoid overly noisy logs
            for p in skipped_paths[:50]:
                self.logger.warning(f"   - missing selector, skipped: {p}")
        return context
    
    def _process_html(self, html_content: str) -> tuple:
        """
        处理单个 HTML 内容
        
        Args:
            html_content: HTML 内容
            
        Returns:
            tuple: (处理后的文本, 提取的代码块列表, 是否成功使用 content_selector)
        """
        if not html_content:
            return "", [], False
        
        # 清理 HTML
        cleaned_html = clean_html_content(html_content)
        
        # 提取代码块
        extracted_codes = []
        if self.code_extractor:
            cleaned_html, extracted_codes = self.code_extractor.extract_code_blocks(cleaned_html)
        
        # 解析 HTML
        soup = BeautifulSoup(cleaned_html, "html.parser")
        
        # 尝试找到主要内容区域
        content_div = None
        if self.content_selector:
            # 解析选择器
            parts = self.content_selector.split(".")
            tag_name = parts[0] if parts[0] else "div"
            class_name = parts[1] if len(parts) > 1 else None
            
            if class_name:
                content_div = soup.find(tag_name, class_=class_name)
            else:
                content_div = soup.find(tag_name)
        
        # 提取文本
        used_selector = content_div is not None
        html_fragment = str(content_div) if content_div is not None else str(soup)
        text = self._html_to_text(html_fragment)
        
        # 清理文本
        text = clean_text(text)
        
        return text, extracted_codes, used_selector

    def _html_to_text(self, html: str) -> str:
        """
        Convert HTML to markdown-ish text using html2text.

        We intentionally avoid extra manual post-processing here to preserve hierarchy
        (lists/tables/headings) as best as html2text can.
        """
        h = html2text.HTML2Text()
        h.ignore_links = self.html2text_ignore_links
        # 0 means no wrapping (avoid hard-wrapping that creates artificial newlines)
        h.body_width = int(self.html2text_body_width) if self.html2text_body_width is not None else 0
        # Prefer keeping structure
        h.ignore_images = True
        try:
            h.ignore_tables = False
        except Exception:
            pass
        return h.handle(html or "")

    # Note: previous versions contained manual newline/table heuristics here.
    # We intentionally removed them and rely on html2text to preserve structure.

