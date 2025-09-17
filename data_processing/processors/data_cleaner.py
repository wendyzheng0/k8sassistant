#!/usr/bin/env python3
"""
数据清理工具模块
用于清理和验证节点数据，避免数据类型错误
"""

import re
from typing import List, Any, Dict, Union
from llama_index.core import Document
from bs4 import BeautifulSoup
from code_extractor import CodeExtractor
    


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
    
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    
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
        # 确保key是字符串
        if not isinstance(key, str):
            key = str(key)
        
        # 清理key，移除特殊字符
        key = re.sub(r'[^\w\-_.]', '_', key)
        
        # 处理value
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
    
    # 检查是否包含有效内容（至少包含字母或数字）
    if not re.search(r'[a-zA-Z0-9\u4e00-\u9fff]', text):
        return False
    
    return True


def clean_document(doc: Document) -> Document:
    """
    清理文档数据
    
    Args:
        doc: 原始文档对象
        
    Returns:
        Document: 清理后的文档对象
    """
    # 清理文本
    cleaned_text = clean_text(doc.text)
    
    # 清理元数据
    cleaned_metadata = clean_metadata(doc.metadata)
    
    # 创建新的文档对象
    return Document(text=cleaned_text, metadata=cleaned_metadata)


def clean_node(node: Any) -> Any:
    """
    清理节点数据
    
    Args:
        node: 原始节点对象
        
    Returns:
        清理后的节点对象
    """
    try:
        # 清理文本
        if hasattr(node, 'text') and node.text is not None:
            node.text = clean_text(node.text)
        
        # 清理元数据
        if hasattr(node, 'metadata') and node.metadata is not None:
            node.metadata = clean_metadata(node.metadata)
        
        return node
        
    except Exception as e:
        print(f"⚠️ 清理节点时出错: {e}")
        return node


def validate_node(node: Any, min_text_length: int = 10) -> bool:
    """
    验证节点是否有效
    
    Args:
        node: 节点对象
        min_text_length: 最小文本长度
        
    Returns:
        bool: 是否有效
    """
    try:
        # 检查是否有text属性
        if not hasattr(node, 'text'):
            return False
        
        # 验证文本
        if not validate_node_text(node.text, min_text_length):
            return False
        
        return True
        
    except Exception as e:
        print(f"⚠️ 验证节点时出错: {e}")
        return False


def clean_and_validate_nodes(nodes: List[Any], min_text_length: int = 10) -> List[Any]:
    """
    清理和验证节点列表，并关联代码块
    
    Args:
        nodes: 原始节点列表
        min_text_length: 最小文本长度
        
    Returns:
        List: 清理和验证后的有效节点列表
    """
    print(f"🔍 清理和验证 {len(nodes)} 个节点...")
    
    valid_nodes = []
    skipped_count = 0
    
    for i, node in enumerate(nodes):
        try:
            # 清理节点
            cleaned_node = clean_node(node)
            
            # 关联代码块到节点
            cleaned_node = associate_codes_with_node(cleaned_node)
            
            # 验证节点
            if validate_node(cleaned_node, min_text_length):
                valid_nodes.append(cleaned_node)
            else:
                skipped_count += 1
                if skipped_count <= 5:  # 只显示前5个跳过的节点
                    print(f"⚠️ 跳过节点 {i}: 文本无效或太短")
                
        except Exception as e:
            skipped_count += 1
            print(f"⚠️ 处理节点 {i} 时出错: {e}")
            continue
    
    print(f"✅ 清理完成: {len(valid_nodes)}/{len(nodes)} 个有效节点")
    if skipped_count > 5:
        print(f"   跳过了 {skipped_count} 个无效节点")
    
    return valid_nodes


def associate_codes_with_node(node: Any) -> Any:
    """
    将代码块关联到节点
    
    Args:
        node: 节点对象
        
    Returns:
        关联了代码块的节点对象
    """
    if not hasattr(node, 'metadata') or not node.metadata:
        return node
    
    # 检查是否有提取的代码块
    extracted_codes = node.metadata.get('extracted_codes', [])
    if not extracted_codes:
        return node
    
    # 初始化节点的代码块列表
    if 'code_blocks' not in node.metadata:
        node.metadata['code_blocks'] = []
    
    # 检查节点文本中是否包含代码块占位符
    node_text = getattr(node, 'text', '')
    if not node_text:
        return node
    
    # 查找节点中的代码块占位符
    import re
    placeholder_pattern = r'\[CODE_BLOCK:([^:]+)\]'
    matches = re.findall(placeholder_pattern, node_text)
    
    for code_id in matches:
        # 查找对应的代码块
        code_block = None
        for code in extracted_codes:
            # 确保code是字典类型
            if isinstance(code, dict) and code.get('id') == code_id:
                code_block = code
                break
        
        if code_block:
            # 将代码块添加到节点的元数据中
            node.metadata['code_blocks'].append({
                'id': code_block.get('id', code_id),
                # 'content': code_block.get('content', ''),
                # 'language': code_block.get('language', ''),
                'length': code_block.get('length', 0),
                # 'preview': preview
            })
            
            # 从节点文本中移除占位符，替换为简短的标记
            placeholder = f"[CODE_BLOCK:{code_id}]"
            # node.text = node.text.replace(placeholder, f"[代码示例: {code_block.get('language', 'unknown')}]")
    
    # 移除已处理的代码块提取信息
    if 'extracted_codes' in node.metadata:
        del node.metadata['extracted_codes']

    return node


def clean_html_content(html_content: str) -> str:
    """
    清理HTML内容
    
    Args:
        html_content: 原始HTML内容
        
    Returns:
        str: 清理后的HTML内容
    """
    if not html_content or not isinstance(html_content, str):
        return ""
    
    # 移除HTML注释
    html_content = re.sub(r'<!--.*?-->', '', html_content, flags=re.DOTALL)
    
    # 移除script和style标签及其内容
    html_content = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    
    # 清理多余的空白字符
    html_content = re.sub(r'\s+', ' ', html_content)
    
    # 去除首尾空白
    html_content = html_content.strip()
    
    return html_content
