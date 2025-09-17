#!/usr/bin/env python3
"""
代码提取器模块
用于从HTML文档中提取<code></code>标签内的代码，并将其存储到meta信息中
"""

import re
import uuid
from typing import List, Dict, Any, Tuple, Optional
from bs4 import BeautifulSoup, Tag
from llama_index.core import Document


class CodeExtractor:
    """代码提取器类"""
    code_location = "../../codeblocks"
    
    def __init__(self):
        self.code_blocks = []  # 存储提取的代码块信息
        
    def extract_code_blocks(self, html_content: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        从HTML内容中提取代码块
        
        Args:
            html_content: HTML内容字符串
            
        Returns:
            Tuple[str, List[Dict]]: (清理后的HTML内容, 代码块列表)
        """
        if not html_content or not isinstance(html_content, str):
            return html_content, []
        
        # 解析HTML
        soup = BeautifulSoup(html_content, "html.parser")
        
        # 查找所有code标签
        code_tags = soup.find_all('code')
        
        extracted_codes = []
        
        for i, code_tag in enumerate(code_tags):
            # 提取代码内容
            code_content = code_tag.get_text(strip=False)
            
            if code_content and len(code_content.strip()) > 500:
                # 生成唯一的代码块ID
                code_id = f"code_block_{uuid.uuid4().hex[:8]}"

                with open(f"{self.code_location}/{code_id}.txt", "w") as f:
                    f.write(code_content)
                
                # 获取代码块的位置信息
                # position_info = self._get_code_position(code_tag, soup)
                
                # 创建代码块信息
                code_info = {
                    "id": code_id,
                    "path": f"{self.code_location}/{code_id}.txt",
                    # "content": code_content,
                    "index": i,
                    # "position": position_info,
                    # "language": self._detect_language(code_tag, code_content),
                    "length": len(code_content)
                }
                
                extracted_codes.append(code_info)
                
                # 用占位符替换原始代码标签
                # placeholder = self._create_placeholder(code_id, code_content)
                placeholder = f"[CODE_BLOCK:{code_id}]"
                code_tag.replace_with(placeholder)
        
        # 获取清理后的HTML内容
        cleaned_html = str(soup)
        
        return cleaned_html, extracted_codes
    
    def _get_code_position(self, code_tag: Tag, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        获取代码块在文档中的位置信息
        
        Args:
            code_tag: 代码标签对象
            soup: BeautifulSoup对象
            
        Returns:
            Dict: 位置信息
        """
        position_info = {
            "tag_name": code_tag.name,
            "parent_tag": code_tag.parent.name if code_tag.parent else None,
            "siblings_count": len(code_tag.parent.find_all()) if code_tag.parent else 0,
            "tag_index": list(code_tag.parent.children).index(code_tag) if code_tag.parent else 0
        }
        
        # 尝试获取更多上下文信息
        if code_tag.parent:
            # 获取父元素的类名和ID
            parent_classes = code_tag.parent.get('class', [])
            parent_id = code_tag.parent.get('id', '')
            
            position_info.update({
                "parent_classes": parent_classes,
                "parent_id": parent_id
            })
            
            # 获取前一个和后一个兄弟元素的信息
            prev_sibling = code_tag.previous_sibling
            next_sibling = code_tag.next_sibling
            
            if prev_sibling and hasattr(prev_sibling, 'get_text'):
                position_info["prev_sibling_text"] = prev_sibling.get_text(strip=True)[:50]
            
            if next_sibling and hasattr(next_sibling, 'get_text'):
                position_info["next_sibling_text"] = next_sibling.get_text(strip=True)[:50]
        
        return position_info
    
    def _detect_language(self, code_tag: Tag, code_content: str) -> str:
        """
        检测代码语言
        
        Args:
            code_tag: 代码标签对象
            code_content: 代码内容
            
        Returns:
            str: 检测到的语言
        """
        # 首先检查class属性中的语言信息
        classes = code_tag.get('class', [])
        for cls in classes:
            if cls.startswith('language-') or cls.startswith('lang-'):
                return cls.split('-', 1)[1]
        
        # 检查父元素的class
        if code_tag.parent:
            parent_classes = code_tag.parent.get('class', [])
            for cls in parent_classes:
                if cls.startswith('language-') or cls.startswith('lang-'):
                    return cls.split('-', 1)[1]
        
        # 基于代码内容进行简单检测
        code_lower = code_content.lower().strip()
        
        # 常见的语言特征
        language_patterns = {
            'bash': [r'#!/bin/bash', r'#!/usr/bin/env bash', r'^\$ ', r'^# '],
            'python': [r'^import ', r'^from ', r'^def ', r'^class ', r'^if __name__'],
            'javascript': [r'function\s+\w+', r'const\s+\w+', r'let\s+\w+', r'var\s+\w+'],
            'yaml': [r'^\s*[a-zA-Z_][a-zA-Z0-9_]*:\s*', r'^\s*-\s*'],
            'json': [r'^\s*[\{\[]', r'^\s*"[^"]*"\s*:'],
            'dockerfile': [r'^FROM\s+', r'^RUN\s+', r'^COPY\s+', r'^WORKDIR\s+'],
            'sql': [r'^SELECT\s+', r'^INSERT\s+', r'^UPDATE\s+', r'^DELETE\s+'],
            'go': [r'^package\s+\w+', r'^import\s+', r'^func\s+\w+'],
            'rust': [r'^use\s+', r'^fn\s+\w+', r'^struct\s+\w+'],
            'java': [r'^public\s+class\s+', r'^import\s+', r'^package\s+'],
            'cpp': [r'^#include\s+<', r'^using\s+namespace', r'^int\s+main'],
            'c': [r'^#include\s+<', r'^int\s+main', r'^void\s+\w+'],
            'html': [r'^<[a-zA-Z]', r'^<!DOCTYPE', r'^<html'],
            'css': [r'^\s*[.#]?\w+\s*\{', r'^\s*@\w+'],
            'xml': [r'^<\?xml', r'^<[a-zA-Z][^>]*>'],
            'markdown': [r'^#\s+', r'^##\s+', r'^\*\s+', r'^-\s+'],
            'powershell': [r'^\$', r'^Get-', r'^Set-', r'^New-'],
            'shell': [r'^\$', r'^#', r'^echo\s+', r'^ls\s+', r'^cd\s+']
        }
        
        for language, patterns in language_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code_content, re.MULTILINE | re.IGNORECASE):
                    return language
        
        return 'text'  # 默认返回text
    
    def _create_placeholder(self, code_id: str, code_content: str) -> str:
        """
        创建代码块的占位符
        
        Args:
            code_id: 代码块ID
            code_content: 代码内容
            
        Returns:
            str: 占位符字符串
        """
        # 创建一个简短的占位符，包含代码块的基本信息
        preview = code_content.strip()[:50].replace('\n', ' ')
        if len(code_content.strip()) > 50:
            preview += "..."
        
        placeholder = f"[CODE_BLOCK:{code_id}:{preview}]"
        return placeholder


    def extract_codes_from_document(self, doc: Document) -> Tuple[Document, List[Dict[str, Any]]]:
        """
        从文档中提取代码块
        
        Args:
            doc: 原始文档对象
            
        Returns:
            Tuple[Document, List[Dict]]: (处理后的文档, 代码块列表)
        """
        if not doc or not hasattr(doc, 'text'):
            return doc, []
        
        # 提取代码块
        cleaned_html, extracted_codes = self.extract_code_blocks(doc.text)
        
        # 创建新的文档对象
        processed_doc = Document(
            text=cleaned_html,
            metadata=doc.metadata.copy() if doc.metadata else {}
        )
        
        # 将代码块信息添加到文档元数据中
        if extracted_codes:
            processed_doc.metadata['extracted_codes'] = extracted_codes
            processed_doc.metadata['code_blocks_count'] = len(extracted_codes)
        
        return processed_doc, extracted_codes


def extract_codes_from_html(html_content: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    便捷函数：从HTML内容中提取代码块
    
    Args:
        html_content: HTML内容字符串
        
    Returns:
        Tuple[str, List[Dict]]: (清理后的HTML内容, 代码块列表)
    """
    extractor = CodeExtractor()
    return extractor.extract_code_blocks(html_content)


def process_document_with_code_extraction(doc: Document) -> Tuple[Document, List[Dict[str, Any]]]:
    """
    便捷函数：处理文档并提取代码块
    
    Args:
        doc: 原始文档对象
        
    Returns:
        Tuple[Document, List[Dict]]: (处理后的文档, 代码块列表)
    """
    extractor = CodeExtractor()
    return extractor.extract_codes_from_document(doc)
