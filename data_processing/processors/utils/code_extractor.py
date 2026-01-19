"""
Code Extractor Utility
代码提取器

从 code_extractor.py 迁移并优化
"""

import os
import re
import uuid
from typing import List, Dict, Any, Tuple, Optional
from bs4 import BeautifulSoup, Tag


class CodeExtractor:
    """
    代码提取器类
    从 HTML 文档中提取 <code> 标签内的代码，并将其存储到文件中
    """
    
    def __init__(self, code_location: str = None):
        """
        初始化代码提取器
        
        Args:
            code_location: 代码块保存目录
        """
        self.code_location = code_location or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "backend",
            "codeblocks"
        )
        self.code_blocks = []
        
        # 确保代码块目录存在
        try:
            os.makedirs(self.code_location, exist_ok=True)
        except (OSError, PermissionError) as e:
            print(f"Warning: Could not create code directory '{self.code_location}': {e}")
    
    def extract_code_blocks(self, html_content: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        从 HTML 内容中提取代码块
        
        Args:
            html_content: HTML 内容字符串
            
        Returns:
            Tuple[str, List[Dict]]: (清理后的 HTML 内容, 代码块列表)
        """
        if not html_content or not isinstance(html_content, str):
            return html_content, []
        
        # 解析 HTML
        soup = BeautifulSoup(html_content, "html.parser")
        
        # 查找所有 code 标签
        code_tags = soup.find_all('code')
        
        extracted_codes = []
        
        for i, code_tag in enumerate(code_tags):
            # 提取代码内容
            code_content = code_tag.get_text(strip=False)
            
            # 只处理较长的代码块（超过 500 字符）
            if code_content and len(code_content.strip()) > 500:
                # 生成唯一的代码块 ID
                code_id = f"code_block_{uuid.uuid4().hex[:8]}"
                
                try:
                    code_path = os.path.join(self.code_location, f"{code_id}.txt")
                    with open(code_path, "w", encoding='utf-8') as f:
                        f.write(code_content)
                except (OSError, PermissionError) as e:
                    print(f"Warning: Could not save code block '{code_id}': {e}")
                    continue
                
                # 创建代码块信息
                code_info = {
                    "id": code_id,
                    "path": code_path,
                    "index": i,
                    "length": len(code_content),
                    "language": self._detect_language(code_tag, code_content),
                }
                
                extracted_codes.append(code_info)
                
                # 用占位符替换原始代码标签
                placeholder = f"[CODE_BLOCK:{code_id}]"
                code_tag.replace_with(placeholder)
        
        # 获取清理后的 HTML 内容
        cleaned_html = str(soup)
        
        return cleaned_html, extracted_codes
    
    def _detect_language(self, code_tag: Tag, code_content: str) -> str:
        """
        检测代码语言
        
        Args:
            code_tag: 代码标签对象
            code_content: 代码内容
            
        Returns:
            str: 检测到的语言
        """
        # 首先检查 class 属性中的语言信息
        classes = code_tag.get('class', [])
        for cls in classes:
            if cls.startswith('language-') or cls.startswith('lang-'):
                return cls.split('-', 1)[1]
        
        # 检查父元素的 class
        if code_tag.parent:
            parent_classes = code_tag.parent.get('class', [])
            for cls in parent_classes:
                if cls.startswith('language-') or cls.startswith('lang-'):
                    return cls.split('-', 1)[1]
        
        # 基于代码内容进行简单检测
        language_patterns = {
            'bash': [r'#!/bin/bash', r'#!/usr/bin/env bash', r'^\$ ', r'^# '],
            'python': [r'^import ', r'^from ', r'^def ', r'^class ', r'^if __name__'],
            'javascript': [r'function\s+\w+', r'const\s+\w+', r'let\s+\w+', r'var\s+\w+'],
            'yaml': [r'^\s*[a-zA-Z_][a-zA-Z0-9_]*:\s*', r'^\s*-\s*'],
            'json': [r'^\s*[\{\[]', r'^\s*"[^"]*"\s*:'],
            'dockerfile': [r'^FROM\s+', r'^RUN\s+', r'^COPY\s+', r'^WORKDIR\s+'],
            'go': [r'^package\s+\w+', r'^import\s+', r'^func\s+\w+'],
            'shell': [r'^\$', r'^#', r'^echo\s+', r'^ls\s+', r'^cd\s+'],
        }
        
        for language, patterns in language_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code_content, re.MULTILINE | re.IGNORECASE):
                    return language
        
        return 'text'


def extract_codes_from_html(html_content: str, code_location: str = None) -> Tuple[str, List[Dict[str, Any]]]:
    """
    便捷函数：从 HTML 内容中提取代码块
    
    Args:
        html_content: HTML 内容字符串
        code_location: 代码块保存目录
        
    Returns:
        Tuple[str, List[Dict]]: (清理后的 HTML 内容, 代码块列表)
    """
    extractor = CodeExtractor(code_location)
    return extractor.extract_code_blocks(html_content)

