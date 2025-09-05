"""
文档管理 API 端点
"""

import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse

from app.models.chat import SearchResult
from app.services.milvus_service import MilvusService
from app.services.embedding_service import EmbeddingService
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger("DocumentsAPI")


def get_milvus_service(request: Request) -> MilvusService:
    """获取 Milvus 服务实例"""
    return request.app.state.milvus_service


def get_embedding_service() -> EmbeddingService:
    """获取嵌入服务实例"""
    return EmbeddingService()


@router.get("/search")
async def search_documents(
    query: str,
    top_k: int = 5,
    milvus_service: MilvusService = Depends(get_milvus_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """
    搜索文档
    """
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query content cannot be empty")
        
        logger.info(f"🔍 Searching documents: {query[:50]}...")
        
        # 将查询编码为向量
        query_embedding = embedding_service.encode(query)[0]
        
        # 在向量数据库中搜索
        similar_docs = await milvus_service.search_similar(
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        # 转换为响应格式
        search_results = []
        for doc in similar_docs:
            search_results.append(SearchResult(
                id=doc["id"],
                title=doc.get("metadata", {}).get("title", "Unknown document"),
                content=doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                url=doc.get("metadata", {}).get("url", ""),
                score=doc["score"],
                metadata=doc.get("metadata", {})
            ))
        
        logger.info(f"✅ Search completed, returned {len(search_results)} results")
        return {
            "query": query,
            "results": search_results,
            "total": len(search_results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Document search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document search failed: {str(e)}")


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    milvus_service: MilvusService = Depends(get_milvus_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """
    上传文档
    """
    try:
        # 检查文件类型
        allowed_extensions = [".txt", ".md", ".pdf", ".docx", ".html"]
        file_extension = "." + file.filename.split(".")[-1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_extension}"
            )
        
        logger.info(f"📤 Uploading document: {file.filename}")
        
        # 读取文件内容
        content = await file.read()
        
        # TODO: 根据文件类型解析内容
        # 这里简化处理，假设是文本文件
        text_content = content.decode("utf-8")
        
        # 生成文档ID
        doc_id = str(uuid.uuid4())
        
        # 将文本编码为向量
        embedding = embedding_service.encode(text_content)[0]
        
        # 准备文档数据
        document = {
            "id": doc_id,
            "content": text_content,
            "metadata": {
                "filename": file.filename,
                "file_size": len(content),
                "file_type": file_extension,
                "title": file.filename
            },
            "embedding": embedding
        }
        
        # 插入到向量数据库
        await milvus_service.insert_documents([document])
        
        logger.info(f"✅ Document uploaded successfully: {doc_id}")
        
        return {
            "message": "Document uploaded successfully",
            "document_id": doc_id,
            "filename": file.filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document upload failed: {str(e)}")


@router.get("/stats")
async def get_document_stats(
    milvus_service: MilvusService = Depends(get_milvus_service)
):
    """
    获取文档统计信息
    """
    try:
        stats = await milvus_service.get_collection_stats()
        
        return {
            "total_documents": stats["row_count"],
            "collection_name": stats["collection_name"],
            "vector_dimension": stats["vector_dim"]
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get document statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document statistics: {str(e)}")


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    milvus_service: MilvusService = Depends(get_milvus_service)
):
    """
    删除指定文档
    """
    try:
        await milvus_service.delete_documents([document_id])
        
        logger.info(f"✅ Document deleted successfully: {document_id}")
        
        return {
            "message": "Document deleted successfully",
            "document_id": document_id
        }
        
    except Exception as e:
        logger.error(f"❌ Document deletion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document deletion failed: {str(e)}")


@router.get("/list")
async def list_documents(
    page: int = 1,
    size: int = 20,
    milvus_service: MilvusService = Depends(get_milvus_service)
):
    """
    列出文档（分页）
    """
    try:
        # TODO: 实现文档列表功能
        # 这里暂时返回空结果
        logger.info(f"📋 Getting document list, page: {page}, size: {size}")
        
        return {
            "documents": [],
            "total": 0,
            "page": page,
            "size": size
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get document list: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document list: {str(e)}")
