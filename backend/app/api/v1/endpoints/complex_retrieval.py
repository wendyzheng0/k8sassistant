"""
复杂检索 API 端点
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field

from app.services.complex_retrieval_service import ComplexRetrievalService, RetrievalResult
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger("ComplexRetrievalAPI")



class ComplexRetrievalRequest(BaseModel):
    """复杂检索请求"""
    query: str = Field(..., description="搜索查询", min_length=1, max_length=1000)
    top_k: int = Field(10, description="返回结果数量", ge=1, le=50)
    milvus_weight: float = Field(0.6, description="Milvus 结果权重", ge=0.0, le=1.0)
    elasticsearch_weight: float = Field(0.4, description="Elasticsearch 结果权重", ge=0.0, le=1.0)
    use_reranking: bool = Field(True, description="是否使用重排序")


class ComplexRetrievalResponse(BaseModel):
    """复杂检索响应"""
    query: str
    documents: list
    execution_time: float
    metadata: dict


async def get_complex_retrieval_service(request: Request) -> ComplexRetrievalService:
    """获取复杂检索服务实例"""
    return request.app.state.complex_retrieval_service


@router.post("/search", response_model=ComplexRetrievalResponse)
async def complex_search(
    request: ComplexRetrievalRequest,
    service: ComplexRetrievalService = Depends(get_complex_retrieval_service)
):
    """
    执行复杂检索
    
    结合 Milvus 向量数据库、Elasticsearch 和 CrossEncoder 重排序
    """
    try:
        logger.info(f"🔍 Complex search request: {request.query[:50]}...")
        
        # 验证权重总和
        total_weight = request.milvus_weight + request.elasticsearch_weight
        if abs(total_weight - 1.0) > 0.01:
            raise HTTPException(
                status_code=400,
                detail=f"权重总和必须为 1.0，当前为 {total_weight}"
            )
        
        # 执行检索
        result: RetrievalResult = await service.search(
            query=request.query,
            top_k=request.top_k,
            milvus_weight=request.milvus_weight,
            elasticsearch_weight=request.elasticsearch_weight,
            use_reranking=request.use_reranking
        )
        
        # 构建响应
        response = ComplexRetrievalResponse(
            query=result.query,
            documents=result.documents,
            execution_time=result.execution_time,
            metadata=result.metadata
        )
        
        logger.info(f"✅ Complex search completed in {result.execution_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Complex search failed: {e}")
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")


@router.get("/stats")
async def get_retrieval_stats(
    service: ComplexRetrievalService = Depends(get_complex_retrieval_service)
):
    """获取检索服务统计信息"""
    try:
        stats = await service.get_stats()
        return stats
    except Exception as e:
        logger.error(f"❌ Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


@router.get("/health")
async def health_check(
    service: ComplexRetrievalService = Depends(get_complex_retrieval_service)
):
    """健康检查"""
    try:
        stats = await service.get_stats()
        
        # 检查服务状态
        is_healthy = (
            stats.get('service_initialized', False) and
            stats.get('milvus_stats', {}).get('status') in ['exists', 'exists_but_no_details'] and
            stats.get('elasticsearch_connected', False)
        )
        
        if is_healthy:
            return {
                "status": "healthy",
                "message": "所有服务正常运行",
                "details": stats
            }
        else:
            return {
                "status": "unhealthy",
                "message": "部分服务不可用",
                "details": stats
            }
            
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            "status": "error",
            "message": f"健康检查失败: {str(e)}",
            "details": {}
        }


@router.post("/test")
async def test_retrieval(
    service: ComplexRetrievalService = Depends(get_complex_retrieval_service)
):
    """测试检索功能"""
    try:
        # 使用简单的测试查询
        test_query = "kubernetes deployment"
        
        result: RetrievalResult = await service.search(
            query=test_query,
            top_k=5,
            milvus_weight=0.6,
            elasticsearch_weight=0.4,
            use_reranking=True
        )
        
        return {
            "status": "success",
            "message": "测试检索成功",
            "test_query": test_query,
            "results_count": len(result.documents),
            "execution_time": result.execution_time,
            "metadata": result.metadata
        }
        
    except Exception as e:
        logger.error(f"❌ Test retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"测试检索失败: {str(e)}")

