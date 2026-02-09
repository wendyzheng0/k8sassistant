"""
消息数据访问层 (Repository)
提供消息相关的数据库操作
"""

import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, desc
from sqlalchemy.orm import selectinload
from loguru import logger

from app.models.models import Message


class MessageRepository:
    """消息数据访问层"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_message(self, chat_id: uuid.UUID, content: str, 
                           role: str, sender_id: Optional[uuid.UUID] = None,
                           message_type: str = "text",
                           metadata: Optional[Dict[str, Any]] = None) -> Message:
        """创建消息"""
        try:
            message = Message(
                chat_id=chat_id,
                content=content,
                role=role,
                sender_id=sender_id,
                message_type=message_type,
                metadata=metadata or {}
            )
            self.session.add(message)
            await self.session.flush()
            
            logger.info(f"✅ Created message: {message.id} in chat {chat_id}")
            return message
            
        except Exception as e:
            logger.error(f"❌ Failed to create message in chat {chat_id}: {e}")
            await self.session.rollback()
            raise
    
    async def get_message_by_id(self, message_id: uuid.UUID) -> Optional[Message]:
        """根据ID获取消息"""
        try:
            result = await self.session.execute(
                select(Message).where(Message.id == message_id)
            )
            message = result.scalar_one_or_none()
            
            if message:
                logger.debug(f"✅ Found message by ID: {message_id}")
            else:
                logger.debug(f"❌ Message not found by ID: {message_id}")
                
            return message
            
        except Exception as e:
            logger.error(f"❌ Failed to get message by ID {message_id}: {e}")
            return None
    
    async def get_chat_messages(self, chat_id: uuid.UUID, limit: int = 100, 
                               offset: int = 0, order_by: str = "desc") -> List[Message]:
        """获取聊天消息列表"""
        try:
            query = select(Message).where(Message.chat_id == chat_id)
            
            if order_by == "desc":
                query = query.order_by(desc(Message.created_at))
            else:
                query = query.order_by(Message.created_at)
            
            query = query.limit(limit).offset(offset)
            
            result = await self.session.execute(query)
            messages = result.scalars().all()
            
            logger.debug(f"✅ Retrieved {len(messages)} messages for chat {chat_id}")
            return list(messages)
            
        except Exception as e:
            logger.error(f"❌ Failed to get messages for chat {chat_id}: {e}")
            return []
    
    async def get_recent_chat_messages(self, chat_id: uuid.UUID, limit: int = 20) -> List[Message]:
        """获取聊天最近的消息"""
        try:
            result = await self.session.execute(
                select(Message)
                .where(Message.chat_id == chat_id)
                .order_by(desc(Message.created_at))
                .limit(limit)
            )
            messages = result.scalars().all()
            
            # 反转顺序，使最新消息在最后
            messages = list(reversed(messages))
            
            logger.debug(f"✅ Retrieved {len(messages)} recent messages for chat {chat_id}")
            return messages
            
        except Exception as e:
            logger.error(f"❌ Failed to get recent messages for chat {chat_id}: {e}")
            return []
    
    async def update_message(self, message_id: uuid.UUID, **kwargs) -> bool:
        """更新消息"""
        try:
            # 过滤掉不允许更新的字段
            allowed_fields = {'content', 'message_type', 'metadata'}
            update_data = {k: v for k, v in kwargs.items() if k in allowed_fields}
            
            if not update_data:
                logger.warning(f"⚠️ No valid fields to update for message {message_id}")
                return False
            
            result = await self.session.execute(
                update(Message)
                .where(Message.id == message_id)
                .values(**update_data)
            )
            
            if result.rowcount > 0:
                logger.info(f"✅ Updated message: {message_id}")
                return True
            else:
                logger.warning(f"⚠️ Message not found for update: {message_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to update message {message_id}: {e}")
            await self.session.rollback()
            return False
    
    async def delete_message(self, message_id: uuid.UUID) -> bool:
        """删除消息"""
        try:
            result = await self.session.execute(
                delete(Message).where(Message.id == message_id)
            )
            
            if result.rowcount > 0:
                logger.info(f"✅ Deleted message: {message_id}")
                return True
            else:
                logger.warning(f"⚠️ Message not found for deletion: {message_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to delete message {message_id}: {e}")
            await self.session.rollback()
            return False
    
    async def get_chat_message_count(self, chat_id: uuid.UUID) -> int:
        """获取聊天消息数量"""
        try:
            result = await self.session.execute(
                select(func.count(Message.id)).where(Message.chat_id == chat_id)
            )
            count = result.scalar()
            
            logger.debug(f"✅ Chat {chat_id} has {count} messages")
            return count or 0
            
        except Exception as e:
            logger.error(f"❌ Failed to get message count for chat {chat_id}: {e}")
            return 0
    
    async def get_messages_by_sender(self, sender_id: uuid.UUID, limit: int = 100) -> List[Message]:
        """获取用户发送的消息"""
        try:
            result = await self.session.execute(
                select(Message)
                .where(Message.sender_id == sender_id)
                .order_by(desc(Message.created_at))
                .limit(limit)
            )
            messages = result.scalars().all()
            
            logger.debug(f"✅ Retrieved {len(messages)} messages from sender {sender_id}")
            return list(messages)
            
        except Exception as e:
            logger.error(f"❌ Failed to get messages by sender {sender_id}: {e}")
            return []
    
    async def search_messages(self, chat_id: uuid.UUID, keyword: str, limit: int = 50) -> List[Message]:
        """在聊天中搜索消息"""
        try:
            result = await self.session.execute(
                select(Message)
                .where(
                    Message.chat_id == chat_id,
                    Message.content.ilike(f"%{keyword}%")
                )
                .order_by(desc(Message.created_at))
                .limit(limit)
            )
            messages = result.scalars().all()
            
            logger.debug(f"✅ Found {len(messages)} messages containing '{keyword}' in chat {chat_id}")
            return list(messages)
            
        except Exception as e:
            logger.error(f"❌ Failed to search messages in chat {chat_id}: {e}")
            return []
    
    async def get_messages_by_type(self, chat_id: uuid.UUID, message_type: str, limit: int = 100) -> List[Message]:
        """获取特定类型的消息"""
        try:
            result = await self.session.execute(
                select(Message)
                .where(
                    Message.chat_id == chat_id,
                    Message.message_type == message_type
                )
                .order_by(desc(Message.created_at))
                .limit(limit)
            )
            messages = result.scalars().all()
            
            logger.debug(f"✅ Retrieved {len(messages)} messages of type '{message_type}' in chat {chat_id}")
            return list(messages)
            
        except Exception as e:
            logger.error(f"❌ Failed to get messages by type in chat {chat_id}: {e}")
            return []
    
    async def delete_old_messages(self, chat_id: uuid.UUID, days_old: int = 90) -> int:
        """删除旧消息（默认90天前）"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            result = await self.session.execute(
                delete(Message)
                .where(
                    Message.chat_id == chat_id,
                    Message.created_at < cutoff_date
                )
            )
            
            deleted_count = result.rowcount
            logger.info(f"✅ Deleted {deleted_count} old messages from chat {chat_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Failed to delete old messages from chat {chat_id}: {e}")
            await self.session.rollback()
            return 0


# 导入需要的函数
