"""
数据库连接和会话管理
提供异步数据库连接和会话管理功能
"""

import asyncio
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy import text
import redis.asyncio as redis
from loguru import logger

from app.core.config import settings
from app.models.models import Base


class DatabaseManager:
    """数据库连接管理器"""
    
    def __init__(self):
        self.engine = None
        self.async_session_maker = None
        self.redis_client = None
        
    async def initialize(self):
        """初始化数据库连接"""
        try:
            # PostgreSQL 连接
            database_url = settings.DATABASE_URL
            if not database_url:
                raise ValueError("DATABASE_URL not configured")
                
            # 创建异步引擎
            self.engine = create_async_engine(
                database_url,
                echo=settings.DEBUG,
                poolclass=NullPool if settings.DEBUG else None,
                future=True,
                connect_args={"server_settings": {"search_path": "app_k8sassist"}}
            )
            
            # 测试连接
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
                logger.info("✅ PostgreSQL connection successful")
            
            # 创建会话工厂
            self.async_session_maker = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Redis 连接
            redis_url = settings.REDIS_URL
            if redis_url:
                self.redis_client = redis.from_url(
                    redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                # 测试 Redis 连接
                await self.redis_client.ping()
                logger.info("✅ Redis connection successful")
            else:
                logger.warning("⚠️ Redis URL not configured, caching disabled")
                
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    async def close(self):
        """关闭数据库连接"""
        if self.engine:
            await self.engine.dispose()
            logger.info("✅ PostgreSQL connection closed")
            
        if self.redis_client:
            await self.redis_client.close()
            logger.info("✅ Redis connection closed")
    
    async def create_tables(self):
        """创建数据库表"""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
                logger.info("✅ Database tables created successfully")
        except Exception as e:
            logger.error(f"❌ Failed to create tables: {e}")
            raise
    
    async def drop_tables(self):
        """删除数据库表（仅用于开发/测试）"""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
                logger.info("✅ Database tables dropped successfully")
        except Exception as e:
            logger.error(f"❌ Failed to drop tables: {e}")
            raise
    
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        获取数据库会话

        使用 async with 上下文管理器，会自动处理：
        - 成功时提交
        - 失败时回滚
        - 最终关闭会话
        """
        async with self.async_session_maker() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            # 注意：不需要在这里调用 session.close()
            # async with 退出时会自动关闭会话


# 全局数据库管理器实例
db_manager = DatabaseManager()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖注入用的数据库会话"""
    async for session in db_manager.get_session():
        yield session

# Alias for compatibility with existing code
get_session = get_db_session

async def get_redis_client() -> redis.Redis:
    """获取 Redis 客户端"""
    return db_manager.redis_client