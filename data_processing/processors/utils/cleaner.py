"""
Data Cleaner Utilities
数据清理工具函数

从 data_cleaner.py 迁移并优化
"""

import re
from typing import Any, Dict, List, Union


def clean_text(text: Any) -> str:
    """
    清理文本数据，确保是有效的字符串
    
    Args:
        text: 原始文本数据
        
    Returns:
        str: 清理后的字符串
    """
    if text is None:
        return ""
    
    # 转换为字符串
    if not isinstance(text, str):
        text = str(text)
    
    # 移除控制字符和特殊字符
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # 规范化空白字符（保留换行符结构）
    # 先处理多个连续空格
    text = re.sub(r'[ \t]+', ' ', text)
    # 处理多个连续换行
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 去除首尾空白
    text = text.strip()
    
    return text


def clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Union[str, int, float, bool]]:
    """
    清理元数据，确保所有值都是支持的数据类型
    
    Args:
        metadata: 原始元数据字典
        
    Returns:
        Dict: 清理后的元数据字典
    """
    if not metadata:
        return {}
    
    cleaned_metadata = {}
    
    for key, value in metadata.items():
        # 确保 key 是字符串
        if not isinstance(key, str):
            key = str(key)
        
        # 清理 key，移除特殊字符
        key = re.sub(r'[^\w\-_.]', '_', key)
        
        # 处理 value
        if value is None:
            cleaned_metadata[key] = ""
        elif isinstance(value, str):
            cleaned_metadata[key] = clean_text(value)
        elif isinstance(value, (int, float, bool)):
            cleaned_metadata[key] = value
        elif isinstance(value, (list, tuple)):
            # 将列表/元组转换为字符串
            cleaned_metadata[key] = str(value)
        else:
            # 其他类型转换为字符串
            cleaned_metadata[key] = clean_text(str(value))
    
    return cleaned_metadata


def clean_html_content(html_content: str) -> str:
    """
    清理 HTML 内容
    
    Args:
        html_content: 原始 HTML 内容
        
    Returns:
        str: 清理后的 HTML 内容
    """
    if not html_content or not isinstance(html_content, str):
        return ""
    
    # 移除 HTML 注释
    html_content = re.sub(r'<!--.*?-->', '', html_content, flags=re.DOTALL)
    
    # 移除 script 和 style 标签及其内容
    html_content = re.sub(
        r'<(script|style)[^>]*>.*?</\1>', 
        '', 
        html_content, 
        flags=re.DOTALL | re.IGNORECASE
    )
    
    # 移除 noscript 标签内容
    html_content = re.sub(
        r'<noscript[^>]*>.*?</noscript>', 
        '', 
        html_content, 
        flags=re.DOTALL | re.IGNORECASE
    )
    
    # 移除导航、页脚等非内容元素
    for tag in ['nav', 'footer', 'header', 'aside']:
        html_content = re.sub(
            rf'<{tag}[^>]*>.*?</{tag}>', 
            '', 
            html_content, 
            flags=re.DOTALL | re.IGNORECASE
        )
    
    return html_content


def validate_node_text(text: str, min_length: int = 10) -> bool:
    """
    验证节点文本是否有效
    
    Args:
        text: 文本内容
        min_length: 最小长度要求
        
    Returns:
        bool: 是否有效
    """
    if not text or not isinstance(text, str):
        return False
    
    # 检查长度
    if len(text.strip()) < min_length:
        return False
    
    # 检查是否只包含空白字符
    if not text.strip():
        return False
    
    # 检查是否包含有效内容（至少包含字母或数字或中文）
    if not re.search(r'[a-zA-Z0-9\u4e00-\u9fff]', text):
        return False
    
    return True


def clean_and_validate_chunks(
    chunks: List[Any], 
    min_text_length: int = 10
) -> List[Any]:
    """
    清理和验证文本块列表
    
    Args:
        chunks: 原始文本块列表
        min_text_length: 最小文本长度
        
    Returns:
        List: 清理和验证后的有效块列表
    """
    valid_chunks = []
    skipped_count = 0
    
    for chunk in chunks:
        try:
            # 清理文本
            if hasattr(chunk, 'content'):
                chunk.content = clean_text(chunk.content)
                
                # 验证文本
                if validate_node_text(chunk.content, min_text_length):
                    valid_chunks.append(chunk)
                else:
                    skipped_count += 1
            elif hasattr(chunk, 'text'):
                chunk.text = clean_text(chunk.text)
                
                if validate_node_text(chunk.text, min_text_length):
                    valid_chunks.append(chunk)
                else:
                    skipped_count += 1
            else:
                valid_chunks.append(chunk)
                
        except Exception as e:
            skipped_count += 1
            continue
    
    return valid_chunks

