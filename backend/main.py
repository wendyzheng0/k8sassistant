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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger = setup_logging()
    app.state.logger = logger
    app.state.logger.info("🚀 K8s Assistant starting...")
    
    # 初始化 Milvus 连接
    try:
        app.state.milvus_service = MilvusService()
        await app.state.milvus_service.initialize()
        app.state.logger.info("✅ Milvus connection initialized successfully")
    except Exception as e:
        app.state.logger.error(f"❌ Failed to initialize Milvus connection: {e}")
    
    yield
    
    # 关闭时清理
    if hasattr(app.state, 'milvus_service'):
        await app.state.milvus_service.close()
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
