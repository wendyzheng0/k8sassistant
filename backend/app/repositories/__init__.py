"""
Repository 工厂类
提供统一的 Repository 访问接口
"""

from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession

from .user_repository import UserRepository
from .chat_repository import ChatRepository
from .message_repository import MessageRepository
from .factory import RepositoryFactory
from app.core.database import db_manager


# 上下文管理器版本的工厂函数
@asynccontextmanager
async def get_repository_factory() -> AsyncGenerator[RepositoryFactory, None]:
    """
    获取 Repository 工厂实例（上下文管理器版本）

    使用方式:
        async with get_repository_factory() as repo_factory:
            user = await repo_factory.users.get_user_by_id(user_id)
    """
    async with db_manager.async_session_maker() as session:
        try:
            yield RepositoryFactory(session)
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# FastAPI 依赖注入版本
async def get_db_repository_factory() -> RepositoryFactory:
    """
    获取 Repository 工厂实例（FastAPI 依赖注入版本）

    ⚠️ 仅在 FastAPI 依赖注入中使用！
    """
    async for factory in get_repository_factory():
        return factory