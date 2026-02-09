"""
用户服务层
协调用户相关的业务逻辑，包括 Repository 和缓存
"""

import uuid
from typing import Optional, List, Dict, Any
from loguru import logger

from app.models.models import User
from app.repositories import get_repository_factory
from app.services.cache_service import CacheService
from app.services.cache_utils import build_user_from_cache
from app.core.security import get_password_hash, verify_password


class UserService:
    """用户服务层"""

    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache = cache_service

    async def create_user(self, user_in) -> User:
        """创建用户"""
        async with get_repository_factory() as repo:
            try:
                email = user_in.email
                username = getattr(user_in, 'username', email)
                password = user_in.password
                is_anonymous = getattr(user_in, 'is_anonymous', False)
                anonymous_id = getattr(user_in, 'anonymous_id', None)
                
                # 检查用户名和邮箱唯一性
                existing_user = await repo.users.get_user_by_username(username)
                if existing_user:
                    raise ValueError(f"Username '{username}' already exists")

                existing_user = await repo.users.get_user_by_email(email)
                if existing_user:
                    raise ValueError(f"Email '{email}' already exists")

                # 创建用户
                password_hash = get_password_hash(password)
                user = await repo.users.create_user(
                    username=username,
                    email=email,
                    password_hash=password_hash,
                    is_anonymous=is_anonymous,
                    anonymous_id=anonymous_id
                )

                # 缓存用户信息
                if self.cache:
                    await self.cache.cache_user(user)

                await repo.commit()
                return user

            except Exception as e:
                await repo.rollback()
                logger.error(f"❌ Failed to create user: {e}")
                raise

    async def create_anonymous_user(self, anonymous_id: uuid.UUID) -> User:
        """创建匿名用户"""
        async with get_repository_factory() as repo:
            try:
                username = f"anon_{anonymous_id.hex[:8]}"
                email = f"{username}@anonymous.local"

                user = await repo.users.create_user(
                    username=username,
                    email=email,
                    password_hash="anonymous",
                    is_anonymous=True,
                    anonymous_id=anonymous_id
                )

                # 缓存用户信息
                if self.cache:
                    await self.cache.cache_user(user)

                await repo.commit()
                logger.info(f"✅ Created anonymous user: {user.id}")
                return user

            except Exception as e:
                await repo.rollback()
                logger.error(f"❌ Failed to create anonymous user: {e}")
                raise

    async def get_user_by_id(self, user_id: uuid.UUID, use_cache: bool = True) -> Optional[User]:
        """根据ID获取用户"""
        # 先尝试从缓存获取
        if use_cache and self.cache:
            cached_user = await self.cache.get_cached_user(user_id)
            if cached_user:
                user = build_user_from_cache(cached_user)
                if user:
                    return user

        # 从数据库获取
        async with get_repository_factory() as repo:
            user = await repo.users.get_user_by_id(user_id)
            
            # 缓存用户信息
            if user and self.cache and use_cache:
                await self.cache.cache_user(user)
            
            return user

    async def get_user_by_email(self, email: str, use_cache: bool = True) -> Optional[User]:
        """根据邮箱获取用户"""
        async with get_repository_factory() as repo:
            user = await repo.users.get_user_by_email(email)
            if user and self.cache and use_cache:
                await self.cache.cache_user(user)
            return user

    async def get_user_by_anonymous_id(self, anonymous_id: uuid.UUID, use_cache: bool = True) -> Optional[User]:
        """根据匿名ID获取用户"""
        async with get_repository_factory() as repo:
            user = await repo.users.get_user_by_anonymous_id(anonymous_id)
            if user and self.cache and use_cache:
                await self.cache.cache_user(user)
            return user

    async def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """验证用户登录"""
        async with get_repository_factory() as repo:
            user = await repo.users.get_user_by_email(email)
            if not user:
                logger.warning(f"⚠️ User not found: {email}")
                return None

            if user.is_anonymous:
                logger.warning(f"⚠️ Anonymous user cannot login: {email}")
                return None

            if not verify_password(password, user.password_hash):
                logger.warning(f"⚠️ Invalid password for user: {email}")
                return None

            logger.info(f"✅ User authenticated: {email}")
            return user

    async def merge_anonymous_user(self, anonymous_user_id: uuid.UUID, login_user_id: uuid.UUID) -> bool:
        """合并匿名用户数据到登录用户"""
        async with get_repository_factory() as repo:
            try:
                # 数据库层面合并
                db_success = await repo.users.merge_anonymous_user(anonymous_user_id, login_user_id)

                if db_success:
                    # Redis缓存层面合并
                    if self.cache:
                        await self.cache.merge_anonymous_data(anonymous_user_id, login_user_id)

                    await repo.commit()
                    logger.info(f"✅ Merged anonymous user {anonymous_user_id} to login user {login_user_id}")
                else:
                    await repo.rollback()

                return db_success

            except Exception as e:
                await repo.rollback()
                logger.error(f"❌ Failed to merge anonymous user {anonymous_user_id}: {e}")
                return False

    async def update_user_profile(self, user_id: uuid.UUID, **kwargs) -> bool:
        """更新用户资料"""
        async with get_repository_factory() as repo:
            try:
                # 检查用户名和邮箱的唯一性
                if 'username' in kwargs:
                    existing_user = await repo.users.get_user_by_username(kwargs['username'])
                    if existing_user and existing_user.id != user_id:
                        raise ValueError(f"Username '{kwargs['username']}' already exists")

                if 'email' in kwargs:
                    existing_user = await repo.users.get_user_by_email(kwargs['email'])
                    if existing_user and existing_user.id != user_id:
                        raise ValueError(f"Email '{kwargs['email']}' already exists")

                # 更新密码需要特殊处理
                if 'password' in kwargs:
                    kwargs['password_hash'] = get_password_hash(kwargs.pop('password'))

                success = await repo.users.update_user(user_id, **kwargs)

                if success:
                    # 更新缓存
                    if self.cache:
                        await self.cache.invalidate_user_cache(user_id)

                    await repo.commit()
                    logger.info(f"✅ Updated user profile: {user_id}")
                else:
                    await repo.rollback()

                return success

            except Exception as e:
                await repo.rollback()
                logger.error(f"❌ Failed to update user profile {user_id}: {e}")
                return False

    async def delete_user(self, user_id: uuid.UUID) -> bool:
        """删除用户账户（软删除）"""
        async with get_repository_factory() as repo:
            try:
                success = await repo.users.delete_user(user_id)

                if success:
                    # 清除缓存
                    if self.cache:
                        await self.cache.invalidate_user_cache(user_id)

                    await repo.commit()
                    logger.info(f"✅ Deleted user account: {user_id}")
                else:
                    await repo.rollback()

                return success

            except Exception as e:
                await repo.rollback()
                logger.error(f"❌ Failed to delete user {user_id}: {e}")
                return False

    async def get_user_stats(self, user_id: uuid.UUID) -> Dict[str, Any]:
        """获取用户统计信息"""
        async with get_repository_factory() as repo:
            try:
                user = await self.get_user_by_id(user_id)
                if not user:
                    return {}

                # 获取聊天数量
                chat_count = await repo.chats.get_chat_count_by_user(user_id)

                stats = {
                    "user_id": str(user_id),
                    "username": user.username,
                    "email": user.email,
                    "is_anonymous": user.is_anonymous,
                    "created_at": user.created_at.isoformat(),
                    "chat_count": chat_count,
                    "last_active": user.updated_at.isoformat()
                }

                return stats

            except Exception as e:
                logger.error(f"❌ Failed to get user stats for {user_id}: {e}")
                return {}

    async def logout(self, user_id: uuid.UUID) -> bool:
        """用户登出"""
        try:
            # 清除用户相关的缓存
            if self.cache:
                await self.cache.invalidate_user_cache(user_id)

            logger.info(f"✅ User logged out: {user_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to logout user {user_id}: {e}")
            return False


# 快捷函数
async def get_user_service() -> UserService:
    """获取用户服务实例"""
    from app.services.cache_service import get_cache_service
    try:
        cache_service = await get_cache_service()
    except RuntimeError:
        cache_service = None
    return UserService(cache_service)