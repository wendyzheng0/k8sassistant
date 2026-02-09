"""
用户数据访问层 (Repository)
提供用户相关的数据库操作
"""

import uuid
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.orm import selectinload
from loguru import logger
from datetime import datetime, timedelta

from app.models.models import User


class UserRepository:
    """用户数据访问层"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_user(self, username: str, email: str, password_hash: str, 
                         is_anonymous: bool = False, anonymous_id: Optional[uuid.UUID] = None) -> User:
        """创建用户"""
        try:
            user = User(
                username=username,
                email=email,
                password_hash=password_hash,
                is_anonymous=is_anonymous,
                anonymous_id=anonymous_id
            )
            self.session.add(user)
            await self.session.flush()
            
            logger.info(f"✅ Created user: {username} ({user.id})")
            return user
            
        except Exception as e:
            logger.error(f"❌ Failed to create user {username}: {e}")
            await self.session.rollback()
            raise
    
    async def get_user_by_id(self, user_id: uuid.UUID) -> Optional[User]:
        """根据ID获取用户"""
        try:
            result = await self.session.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            
            if user:
                logger.debug(f"✅ Found user by ID: {user_id}")
            else:
                logger.debug(f"❌ User not found by ID: {user_id}")
                
            return user
            
        except Exception as e:
            logger.error(f"❌ Failed to get user by ID {user_id}: {e}")
            return None
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """根据邮箱获取用户"""
        try:
            result = await self.session.execute(
                select(User).where(User.email == email)
            )
            user = result.scalar_one_or_none()
            
            if user:
                logger.debug(f"✅ Found user by email: {email}")
            else:
                logger.debug(f"❌ User not found by email: {email}")
                
            return user
            
        except Exception as e:
            logger.error(f"❌ Failed to get user by email {email}: {e}")
            return None
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """根据用户名获取用户"""
        try:
            result = await self.session.execute(
                select(User).where(User.username == username)
            )
            user = result.scalar_one_or_none()
            
            if user:
                logger.debug(f"✅ Found user by username: {username}")
            else:
                logger.debug(f"❌ User not found by username: {username}")
                
            return user
            
        except Exception as e:
            logger.error(f"❌ Failed to get user by username {username}: {e}")
            return None
    
    async def get_user_by_anonymous_id(self, anonymous_id: uuid.UUID) -> Optional[User]:
        """根据匿名ID获取用户"""
        try:
            result = await self.session.execute(
                select(User).where(
                    User.anonymous_id == anonymous_id,
                    User.is_anonymous == True
                )
            )
            user = result.scalar_one_or_none()
            
            if user:
                logger.debug(f"✅ Found user by anonymous ID: {anonymous_id}")
            else:
                logger.debug(f"❌ User not found by anonymous ID: {anonymous_id}")
                
            return user
            
        except Exception as e:
            logger.error(f"❌ Failed to get user by anonymous ID {anonymous_id}: {e}")
            return None
    
    async def update_user(self, user_id: uuid.UUID, **kwargs) -> bool:
        """更新用户信息"""
        try:
            # 过滤掉不允许更新的字段
            allowed_fields = {'username', 'email', 'password_hash', 'is_active', 'is_anonymous', 'anonymous_id'}
            update_data = {k: v for k, v in kwargs.items() if k in allowed_fields}
            
            if not update_data:
                logger.warning(f"⚠️ No valid fields to update for user {user_id}")
                return False
            
            result = await self.session.execute(
                update(User)
                .where(User.id == user_id)
                .values(**update_data)
            )
            
            if result.rowcount > 0:
                logger.info(f"✅ Updated user: {user_id}")
                return True
            else:
                logger.warning(f"⚠️ User not found for update: {user_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to update user {user_id}: {e}")
            await self.session.rollback()
            return False
    
    async def delete_user(self, user_id: uuid.UUID) -> bool:
        """删除用户（软删除）"""
        try:
            result = await self.session.execute(
                update(User)
                .where(User.id == user_id)
                .values(is_active=False)
            )
            
            if result.rowcount > 0:
                logger.info(f"✅ Soft deleted user: {user_id}")
                return True
            else:
                logger.warning(f"⚠️ User not found for deletion: {user_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to delete user {user_id}: {e}")
            await self.session.rollback()
            return False
    
    async def get_all_users(self, limit: int = 100, offset: int = 0) -> List[User]:
        """获取所有用户（分页）"""
        try:
            result = await self.session.execute(
                select(User)
                .where(User.is_active == True)
                .order_by(User.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            users = result.scalars().all()
            
            logger.debug(f"✅ Retrieved {len(users)} users")
            return list(users)
            
        except Exception as e:
            logger.error(f"❌ Failed to get all users: {e}")
            return []
    
    async def get_anonymous_users(self, limit: int = 100) -> List[User]:
        """获取匿名用户列表"""
        try:
            result = await self.session.execute(
                select(User)
                .where(
                    User.is_anonymous == True,
                    User.is_active == True
                )
                .order_by(User.created_at.desc())
                .limit(limit)
            )
            users = result.scalars().all()
            
            logger.debug(f"✅ Retrieved {len(users)} anonymous users")
            return list(users)
            
        except Exception as e:
            logger.error(f"❌ Failed to get anonymous users: {e}")
            return []
    
    async def merge_anonymous_user(self, anonymous_user_id: uuid.UUID, login_user_id: uuid.UUID) -> bool:
        """合并匿名用户数据到登录用户"""
        try:
            from app.models.models import Chat
            
            # 更新匿名用户的所有聊天归属
            result = await self.session.execute(
                update(Chat)
                .where(Chat.user_id == anonymous_user_id)
                .values(user_id=login_user_id, updated_at=func.now())
            )
            
            # 软删除匿名用户
            await self.session.execute(
                update(User)
                .where(User.id == anonymous_user_id)
                .values(is_active=False, updated_at=func.now())
            )
            
            logger.info(f"✅ Merged anonymous user {anonymous_user_id} to login user {login_user_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to merge anonymous user {anonymous_user_id}: {e}")
            await self.session.rollback()
            return False


# 导入需要的函数
from sqlalchemy.sql import func