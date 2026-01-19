"""
管理界面 API 端点
用于查看向量数据库中的文档和分块信息
"""

import os
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, Query
from pydantic import BaseModel

from app.services.milvus_service import MilvusService
from app.core.logging import get_logger
from app.core.config import settings

router = APIRouter()
logger = get_logger("AdminAPI")


class TreeNode(BaseModel):
    """文档树节点"""
    label: str
    path: str
    is_file: bool
    children: Optional[List["TreeNode"]] = None


class ChunkInfo(BaseModel):
    """分块信息"""
    id: str
    content: str
    doc_id: Optional[str] = None
    chunk_index: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


def get_milvus_service(request: Request) -> MilvusService:
    """获取 Milvus 服务实例"""
    return request.app.state.milvus_service


def build_tree(base_path: str, relative_path: str = "") -> List[TreeNode]:
    """
    递归构建文档树结构
    """
    nodes = []
    current_path = os.path.join(base_path, relative_path) if relative_path else base_path
    
    try:
        items = sorted(os.listdir(current_path))
    except OSError:
        return nodes
    
    for item in items:
        # 跳过隐藏文件和特殊目录
        if item.startswith('.') or item.startswith('_'):
            continue
        
        item_full_path = os.path.join(current_path, item)
        item_relative_path = os.path.join(relative_path, item) if relative_path else item
        
        if os.path.isdir(item_full_path):
            children = build_tree(base_path, item_relative_path)
            nodes.append(TreeNode(
                label=item,
                path=item_relative_path,
                is_file=False,
                children=children if children else None
            ))
        elif item.endswith('.html'):
            # 只显示非 index.html 的 HTML 文件
            nodes.append(TreeNode(
                label=item,
                path=item_relative_path,
                is_file=True,
                children=None
            ))
    
    return nodes


@router.get("/document-tree")
async def get_document_tree():
    """
    获取文档目录树结构
    返回 data/zh-cn/docs 下的文档树
    """
    try:
        # 获取文档根目录
        # 支持多种可能的路径配置
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))), "data", "zh-cn"),
            "/app/data/zh-cn",
            os.path.join(os.getcwd(), "data", "zh-cn"),
        ]
        
        docs_path = None
        for path in possible_paths:
            if os.path.exists(path):
                docs_path = path
                break
        
        if not docs_path:
            logger.warning("⚠️ 文档目录不存在")
            return {"tree": [], "base_path": None, "error": "文档目录不存在"}
        
        logger.info(f"📂 构建文档树: {docs_path}")
        
        tree = build_tree(docs_path)
        
        return {
            "tree": tree,
            "base_path": docs_path
        }
        
    except Exception as e:
        logger.error(f"❌ 获取文档树失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取文档树失败: {str(e)}")


def extract_content(entity: Dict[str, Any]) -> str:
    """
    从实体中提取内容，支持多种字段名
    LlamaIndex 使用 'text'，其他实现可能使用 'content'
    """
    # 直接字段
    content = entity.get("text") or entity.get("content") or entity.get("data") or ""
    
    # 如果没找到，尝试从 metadata 中提取
    if not content:
        metadata = entity.get("metadata", {})
        if isinstance(metadata, dict):
            content = metadata.get("text") or metadata.get("content") or ""
    
    # 如果还没找到，尝试从 _node_content 中提取 (LlamaIndex 格式)
    if not content and "_node_content" in entity:
        try:
            import json
            node_content = entity.get("_node_content", "")
            if isinstance(node_content, str):
                node_data = json.loads(node_content)
                content = node_data.get("text") or node_data.get("content") or ""
        except (json.JSONDecodeError, TypeError):
            pass
    
    return content


def extract_metadata(entity: Dict[str, Any]) -> Dict[str, Any]:
    """
    从实体中提取元数据
    """
    metadata = entity.get("metadata", {})
    
    # 如果 metadata 是字符串，尝试解析为 JSON
    if isinstance(metadata, str):
        try:
            import json
            metadata = json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            metadata = {}
    
    # 添加一些可能在顶层的元数据字段
    extra_fields = ["file_path", "file_name", "doc_id", "document_title", 
                    "relative_path", "chunk_index", "creation_date"]
    for field in extra_fields:
        if field in entity and field not in metadata:
            metadata[field] = entity[field]
    
    return metadata if isinstance(metadata, dict) else {}


@router.get("/chunks")
async def get_chunks_by_path(
    file_path: str = Query(..., description="文档相对路径"),
    milvus_service: MilvusService = Depends(get_milvus_service)
):
    """
    根据文档路径获取向量数据库中的分块信息
    """
    try:
        logger.info(f"🔍 查询文档分块: {file_path}")
        
        # 在 Milvus 中查询与该文件路径相关的所有分块
        chunks = await milvus_service.get_chunks_by_file_path(file_path)
        
        # 转换为响应格式
        result = []
        for chunk in chunks:
            entity = chunk.get("entity", chunk)
            
            # 提取内容和元数据
            content = extract_content(entity)
            metadata = extract_metadata(entity)
            
            # 获取 chunk_index
            chunk_index = (
                entity.get("chunk_index") or 
                metadata.get("chunk_index") or 
                None
            )
            
            # 获取文件路径
            entity_file_path = (
                entity.get("file_path") or 
                metadata.get("file_path") or 
                metadata.get("relative_path") or 
                ""
            )
            
            result.append({
                "id": chunk.get("id", entity.get("id", "unknown")),
                "content": content,
                "doc_id": entity.get("doc_id") or metadata.get("doc_id") or "",
                "file_path": entity_file_path,
                "chunk_index": chunk_index,
                "metadata": metadata
            })
        
        # 按 chunk_index 排序（如果有的话）
        result.sort(key=lambda x: (x.get("chunk_index") or 0) if isinstance(x.get("chunk_index"), int) else 0)
        
        logger.info(f"✅ 找到 {len(result)} 个分块")
        
        # 调试日志：输出第一个分块的字段信息
        if result:
            first_chunk = result[0]
            logger.info(f"📋 第一个分块示例 - ID: {first_chunk['id']}, 内容长度: {len(first_chunk['content'])}")
            if chunks:
                first_entity = chunks[0].get("entity", chunks[0])
                logger.info(f"📋 原始实体字段: {list(first_entity.keys())}")
        
        return {
            "file_path": file_path,
            "chunks": result,
            "total": len(result)
        }
        
    except Exception as e:
        logger.error(f"❌ 查询分块失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询分块失败: {str(e)}")


@router.get("/stats")
async def get_admin_stats(
    milvus_service: MilvusService = Depends(get_milvus_service)
):
    """
    获取管理统计信息
    """
    try:
        stats = await milvus_service.get_collection_stats()
        
        return {
            "collection_name": stats.get("collection_name", "unknown"),
            "total_chunks": stats.get("row_count", 0),
            "vector_dimension": stats.get("vector_dim", 0),
            "status": stats.get("status", "unknown")
        }
        
    except Exception as e:
        logger.error(f"❌ 获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")

