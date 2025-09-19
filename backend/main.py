"""
K8s Assistant - FastAPI 主应用入口
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from contextlib import asynccontextmanager
import os

from app.core.config import settings
from app.core.logging import setup_logging
from app.api.v1.api import api_router
from app.services.milvus_service import MilvusService
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.complex_retrieval_service import ComplexRetrievalService
from app.services.graph_construction_service import GraphConstructionService
from app.services.hybrid_retrieval_service import HybridRetrievalService
from app.services.enhanced_llm_service import EnhancedLLMService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger = setup_logging()
    app.state.logger = logger
    app.state.logger.info("🚀 K8s Assistant starting...")
    
    # 初始化所有服务
    try:
        # 初始化 Milvus 服务
        app.state.milvus_service = MilvusService()
        await app.state.milvus_service.initialize()
        app.state.logger.info("✅ Milvus connection initialized successfully")
        
        # 初始化嵌入服务
        app.state.embedding_service = EmbeddingService()
        app.state.logger.info("✅ Embedding service initialized successfully")
        
        # 初始化 LLM 服务
        app.state.llm_service = LLMService()
        app.state.logger.info("✅ LLM service initialized successfully")
        
        # 初始化复杂检索服务
        app.state.complex_retrieval_service = ComplexRetrievalService()
        app.state.logger.info("✅ Complex retrieval service initialized successfully")
        
        # # 初始化图构建服务
        # app.state.graph_construction_service = GraphConstructionService()
        # app.state.logger.info("✅ Graph construction service initialized successfully")
        
        # # 初始化混合检索服务
        # app.state.hybrid_retrieval_service = HybridRetrievalService()
        # app.state.logger.info("✅ Hybrid retrieval service initialized successfully")
        
        # # 初始化增强LLM服务
        # app.state.enhanced_llm_service = EnhancedLLMService()
        # app.state.logger.info("✅ Enhanced LLM service initialized successfully")
        
    except Exception as e:
        app.state.logger.error(f"❌ Failed to initialize services: {e}")
        raise
    
    yield
    
    # 关闭时清理
    if hasattr(app.state, 'milvus_service'):
        await app.state.milvus_service.close()
    if hasattr(app.state, 'embedding_service'):
        del app.state.embedding_service
    if hasattr(app.state, 'llm_service'):
        del app.state.llm_service
    if hasattr(app.state, 'complex_retrieval_service'):
        del app.state.complex_retrieval_service
    if hasattr(app.state, 'graph_construction_service'):
        del app.state.graph_construction_service
    if hasattr(app.state, 'hybrid_retrieval_service'):
        del app.state.hybrid_retrieval_service
    if hasattr(app.state, 'enhanced_llm_service'):
        del app.state.enhanced_llm_service
    app.state.logger.info("👋 K8s Assistant closed")


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例"""
    
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Kubernetes intelligent Q&A assistant based on RAG",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # 配置 CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 注册路由
    app.include_router(api_router, prefix="/api")
    
    # 静态文件服务（仅当目录存在时挂载）
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    # 使用 FastAPI 默认的 Swagger UI 与 ReDoc（CDN 资源）
    
    # 健康检查
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION
        }
    
    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
