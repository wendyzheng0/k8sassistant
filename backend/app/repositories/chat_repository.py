"""
聊天会话数据访问层 (Repository)
提供聊天会话相关的数据库操作
"""

import uuid
from typing import Optional, List
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func
from sqlalchemy.orm import selectinload
from loguru import logger

from app.models.models import Chat, Message


class ChatRepository:
    """聊天会话数据访问层"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_chat(self, user_id: uuid.UUID, title: Optional[str] = None) -> Chat:
        """创建聊天会话"""
        try:
            logger.info(f"Repository: Creating chat for user {user_id}")
            chat = Chat(
                user_id=user_id,
                title=title or "New Chat",
                last_message_at=datetime.utcnow()
            )
            self.session.add(chat)
            await self.session.flush()
            await self.session.refresh(chat)
            logger.info(f"✅ Repository: Created chat {chat.id} and flushed to session.")
            return chat
            
        except Exception as e:
            logger.error(f"❌ Failed to create chat for user {user_id}: {e}")
            await self.session.rollback()
            raise
    
    async def get_chat_by_id(self, chat_id: uuid.UUID, include_messages: bool = False) -> Optional[Chat]:
        """根据ID获取聊天会话"""
        try:
            query = select(Chat).where(Chat.id == chat_id)
            
            if include_messages:
                query = query.options(selectinload(Chat.messages))
            
            result = await self.session.execute(query)
            chat = result.scalar_one_or_none()
            
            if chat:
                logger.debug(f"✅ Found chat by ID: {chat_id}")
            else:
                logger.debug(f"❌ Chat not found by ID: {chat_id}")
                
            return chat
            
        except Exception as e:
            logger.error(f"❌ Failed to get chat by ID {chat_id}: {e}")
            return None
    
    async def get_user_chats(self, user_id: uuid.UUID, limit: int = 100, offset: int = 0, 
                           include_archived: bool = False) -> List[Chat]:
        """获取用户的聊天会话列表"""
        try:
            query = select(Chat).where(Chat.user_id == user_id)
            
            if not include_archived:
                query = query.where(Chat.is_archived == False)
            
            query = query.order_by(Chat.last_message_at.desc()).limit(limit).offset(offset)
            
            result = await self.session.execute(query)
            chats = result.scalars().all()
            
            logger.debug(f"✅ Retrieved {len(chats)} chats for user {user_id}")
            return list(chats)
            
        except Exception as e:
            logger.error(f"❌ Failed to get chats for user {user_id}: {e}")
            return []
    
    async def get_recent_user_chats(self, user_id: uuid.UUID, limit: int = 10) -> List[Chat]:
        """获取用户最近的聊天会话"""
        try:
            result = await self.session.execute(
                select(Chat)
                .where(
                    Chat.user_id == user_id,
                    Chat.is_archived == False
                )
                .order_by(Chat.last_message_at.desc())
                .limit(limit)
            )
            chats = result.scalars().all()
            
            logger.debug(f"✅ Retrieved {len(chats)} recent chats for user {user_id}")
            return list(chats)
            
        except Exception as e:
            logger.error(f"❌ Failed to get recent chats for user {user_id}: {e}")
            return []
    
    async def update_chat(self, chat_id: uuid.UUID, **kwargs) -> bool:
        """更新聊天会话信息"""
        try:
            # 过滤掉不允许更新的字段
            allowed_fields = {'title', 'is_archived', 'last_message_at'}
            update_data = {k: v for k, v in kwargs.items() if k in allowed_fields}
            
            if not update_data:
                logger.warning(f"⚠️ No valid fields to update for chat {chat_id}")
                return False
            
            result = await self.session.execute(
                update(Chat)
                .where(Chat.id == chat_id)
                .values(**update_data)
            )
            
            if result.rowcount > 0:
                logger.info(f"✅ Updated chat: {chat_id}")
                return True
            else:
                logger.warning(f"⚠️ Chat not found for update: {chat_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to update chat {chat_id}: {e}")
            await self.session.rollback()
            return False
    
    async def delete_chat(self, chat_id: uuid.UUID) -> bool:
        """删除聊天会话"""
        try:
            result = await self.session.execute(
                delete(Chat).where(Chat.id == chat_id)
            )
            
            if result.rowcount > 0:
                logger.info(f"✅ Deleted chat: {chat_id}")
                return True
            else:
                logger.warning(f"⚠️ Chat not found for deletion: {chat_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to delete chat {chat_id}: {e}")
            await self.session.rollback()
            return False
    
    async def get_chat_count_by_user(self, user_id: uuid.UUID, include_archived: bool = False) -> int:
        """获取用户的聊天会话数量"""
        try:
            query = select(func.count(Chat.id)).where(Chat.user_id == user_id)
            
            if not include_archived:
                query = query.where(Chat.is_archived == False)
            
            result = await self.session.execute(query)
            count = result.scalar()
            
            logger.debug(f"✅ User {user_id} has {count} chats")
            return count or 0
            
        except Exception as e:
            logger.error(f"❌ Failed to get chat count for user {user_id}: {e}")
            return 0
    
    async def archive_old_chats(self, user_id: uuid.UUID, days_old: int = 30) -> int:
        """归档旧聊天（默认30天无消息）"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            result = await self.session.execute(
                update(Chat)
                .where(
                    Chat.user_id == user_id,
                    Chat.last_message_at < cutoff_date,
                    Chat.is_archived == False
                )
                .values(is_archived=True, updated_at=datetime.utcnow())
            )
            
            archived_count = result.rowcount
            logger.info(f"✅ Archived {archived_count} old chats for user {user_id}")
            return archived_count
            
        except Exception as e:
            logger.error(f"❌ Failed to archive old chats for user {user_id}: {e}")
            await self.session.rollback()
            return 0
    
    async def update_chat_last_message_time(self, chat_id: uuid.UUID) -> bool:
        """更新聊天最后消息时间"""
        try:
            result = await self.session.execute(
                update(Chat)
                .where(Chat.id == chat_id)
                .values(last_message_at=datetime.utcnow())
            )
            
            if result.rowcount > 0:
                logger.debug(f"✅ Updated last message time for chat: {chat_id}")
                return True
            else:
                logger.warning(f"⚠️ Chat not found for last message time update: {chat_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to update last message time for chat {chat_id}: {e}")
            await self.session.rollback()
            return False


# 导入需要的函数
from datetime import timedelta