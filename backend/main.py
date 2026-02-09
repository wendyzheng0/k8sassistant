"""
K8s Assistant - FastAPI 主应用入口
"""

import sys
import os

# Add project root to path for shared module imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from app.core.config import settings, validate_required_settings
from app.core.logging import setup_logging
from app.api.v1.api import api_router
from app.core.database import db_manager
from app.services.milvus_service import MilvusService
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.complex_retrieval_service import ComplexRetrievalService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger = setup_logging()
    app.state.logger = logger
    app.state.logger.info("🚀 K8s Assistant starting...")
    
    # 验证必要配置
    try:
        validate_required_settings()
        app.state.logger.info("✅ Configuration validated successfully")
    except ValueError as e:
        app.state.logger.error(f"❌ Configuration validation failed: {e}")
        raise
    
    # 初始化所有服务
    try:
        # 初始化数据库连接
        app.state.logger.info("🔄 Initializing database connection...")
        await db_manager.initialize()
        app.state.logger.info("✅ Database connection initialized successfully")

        # 初始化 Milvus 服务
        app.state.milvus_service = MilvusService()
        await app.state.milvus_service.initialize()
        app.state.logger.info("✅ Milvus connection initialized successfully")
        
        # 初始化嵌入服务 (now using shared module)
        app.state.embedding_service = EmbeddingService()
        app.state.logger.info("✅ Embedding service initialized successfully")
        
        # 初始化 LLM 服务 (now using shared LLM provider)
        app.state.llm_service = LLMService()
        await app.state.llm_service.initialize()
        app.state.logger.info("✅ LLM service initialized successfully")
        
        # 初始化复杂检索服务
        app.state.complex_retrieval_service = ComplexRetrievalService()
        await app.state.complex_retrieval_service.initialize()
        app.state.logger.info("✅ Complex retrieval service initialized successfully")
        
        app.state.logger.info("✅ All services initialized successfully")
        app.state.logger.info("📋 Using shared LLM providers and embedding services")
        
    except Exception as e:
        app.state.logger.error(f"❌ Failed to initialize services: {e}")
        raise
    
    yield
    
    # 关闭时清理
    app.state.logger.info("🔄 Shutting down services...")
    
    await db_manager.close()

    if hasattr(app.state, 'milvus_service'):
        await app.state.milvus_service.close()
    if hasattr(app.state, 'llm_service'):
        await app.state.llm_service.close()
    if hasattr(app.state, 'complex_retrieval_service'):
        await app.state.complex_retrieval_service.close()
    if hasattr(app.state, 'embedding_service'):
        del app.state.embedding_service
        
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
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
    )
    
    # 注册路由
    app.include_router(api_router, prefix=settings.API_V1_STR)
    
    # 静态文件服务（仅当目录存在时挂载）
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    # 健康检查
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION
        }
    
    # 服务信息端点
    @app.get("/info")
    async def service_info():
        """Get service information including architecture details"""
        from shared.llm_providers import get_available_providers
        from shared.config import get_settings
        
        shared_settings = get_settings()
        
        return {
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "architecture": {
                "llm_provider": shared_settings.LLM_PROVIDER,
                "available_providers": get_available_providers(),
                "embedding_model": shared_settings.EMBEDDING_MODEL,
                "embedding_backend": shared_settings.EMBEDDING_BACKEND
            }
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
