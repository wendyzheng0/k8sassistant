"""
API 路由主文件
"""

from fastapi import APIRouter
from app.api.v1.endpoints import chat, documents, health, graphrag, complex_retrieval

# 创建主路由
api_router = APIRouter()

# 注册子路由
api_router.include_router(health.router, prefix="/health", tags=["健康检查"])
api_router.include_router(chat.router, prefix="/chat", tags=["聊天"])
api_router.include_router(documents.router, prefix="/documents", tags=["文档管理"])
#api_router.include_router(graphrag.router, prefix="/graphrag", tags=["GraphRAG"])
api_router.include_router(complex_retrieval.router, prefix="/complex-retrieval", tags=["复杂检索"])
