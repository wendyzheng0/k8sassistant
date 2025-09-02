"""
健康检查 API 端点
"""

from fastapi import APIRouter, Request, HTTPException
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger("HealthAPI")


@router.get("/")
async def health_check():
    """
    基础健康检查
    """
    return {
        "status": "healthy",
        "message": "K8s Assistant 服务运行正常"
    }


@router.get("/detailed")
async def detailed_health_check(request: Request):
    """
    详细健康检查
    """
    try:
        health_status = {
            "status": "healthy",
            "services": {}
        }
        
        # 检查 Milvus 连接
        try:
            if hasattr(request.app.state, 'milvus_service'):
                stats = await request.app.state.milvus_service.get_collection_stats()
                health_status["services"]["milvus"] = {
                    "status": "healthy",
                    "collection_name": stats["collection_name"],
                    "row_count": stats["row_count"]
                }
            else:
                health_status["services"]["milvus"] = {
                    "status": "unavailable",
                    "message": "Milvus 服务未初始化"
                }
        except Exception as e:
            health_status["services"]["milvus"] = {
                "status": "error",
                "message": str(e)
            }
        
        # 检查嵌入服务
        try:
            from app.services.embedding_service import EmbeddingService
            embedding_service = EmbeddingService()
            dim = embedding_service.get_embedding_dimension()
            health_status["services"]["embedding"] = {
                "status": "healthy",
                "model": embedding_service.model_name,
                "dimension": dim
            }
        except Exception as e:
            health_status["services"]["embedding"] = {
                "status": "error",
                "message": str(e)
            }
        
        # 检查 LLM 服务
        try:
            from app.services.llm_service import LLMService
            llm_service = LLMService()
            model_info = llm_service.get_model_info()
            health_status["services"]["llm"] = {
                "status": "healthy",
                "model": model_info["model"],
                "base_url": model_info["base_url"]
            }
        except Exception as e:
            health_status["services"]["llm"] = {
                "status": "error",
                "message": str(e)
            }
        
        # 检查整体状态
        all_healthy = all(
            service["status"] == "healthy" 
            for service in health_status["services"].values()
        )
        
        if not all_healthy:
            health_status["status"] = "degraded"
        
        logger.info("🔍 详细健康检查完成")
        return health_status
        
    except Exception as e:
        logger.error(f"❌ 健康检查失败: {e}")
        raise HTTPException(status_code=500, detail=f"健康检查失败: {str(e)}")


@router.get("/ready")
async def readiness_check(request: Request):
    """
    就绪检查
    """
    try:
        # 检查关键服务是否就绪
        if not hasattr(request.app.state, 'milvus_service'):
            raise HTTPException(status_code=503, detail="Milvus 服务未就绪")
        
        # 检查集合是否存在
        try:
            stats = await request.app.state.milvus_service.get_collection_stats()
            if stats["row_count"] == 0:
                logger.warning("⚠️ 向量数据库为空")
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Milvus 连接失败: {str(e)}")
        
        return {
            "status": "ready",
            "message": "服务已就绪"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 就绪检查失败: {e}")
        raise HTTPException(status_code=503, detail=f"服务未就绪: {str(e)}")


@router.get("/live")
async def liveness_check():
    """
    存活检查
    """
    return {
        "status": "alive",
        "message": "服务存活"
    }
