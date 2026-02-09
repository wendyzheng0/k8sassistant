"""
聊天服务层
协调聊天和消息的业务逻辑，包括缓存管理
"""

import uuid
from typing import Optional, List, Dict, Any
from loguru import logger

from app.models.models import User, Chat, Message
from app.repositories import get_repository_factory
from app.services.cache_service import CacheService
from app.services.cache_utils import build_chat_from_cache, build_message_from_cache, build_chats_from_cache, build_messages_from_cache


class ChatService:
    """聊天服务层"""

    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache = cache_service
    
    async def create_chat(self, user_id: uuid.UUID, title: Optional[str] = None) -> Chat:
        """创建聊天会话"""
        async with get_repository_factory() as repo:
            try:
                chat = await repo.chats.create_chat(user_id, title)

                # 缓存聊天信息
                if self.cache:
                    await self.cache.cache_chat(chat)
                    await self.cache.add_chat_to_user_list(user_id, chat.id)

                await repo.commit()
                logger.info(f"✅ Created chat {chat.id} for user {user_id}")
                return chat

            except Exception as e:
                await repo.rollback()
                logger.error(f"❌ Failed to create chat for user {user_id}: {e}")
                raise
    
    async def get_chat(self, chat_id: uuid.UUID, user_id: Optional[uuid.UUID] = None,
                      include_messages: bool = False) -> Optional[Chat]:
        """获取聊天会话"""
        # 先尝试从缓存获取
        if self.cache:
            cached_chat = await self.cache.get_cached_chat(chat_id)
            if cached_chat:
                chat = build_chat_from_cache(cached_chat)
                if chat:
                    # 验证用户权限
                    if user_id and chat.user_id != user_id:
                        logger.warning(f"⚠️ User {user_id} tried to access chat {chat_id} without permission")
                        return None
                    
                    # 如果需要消息，单独获取
                    if include_messages:
                        messages = await self.get_chat_messages(chat_id)
                        chat.messages = messages
                    
                    return chat

        # 从数据库获取
        async with get_repository_factory() as repo:
            chat = await repo.chats.get_chat_by_id(chat_id, include_messages)

            # 验证用户权限
            if user_id and chat and chat.user_id != user_id:
                logger.warning(f"⚠️ User {user_id} tried to access chat {chat_id} without permission")
                return None

            # 缓存聊天信息
            if chat and self.cache:
                await self.cache.cache_chat(chat)

            return chat
    
    async def get_user_chats(self, user_id: uuid.UUID, limit: int = 100,
                           include_archived: bool = False) -> List[Chat]:
        """获取用户的聊天会话列表"""
        # 先尝试从缓存获取
        if self.cache:
            cached_chats = await self.cache.get_user_chats(user_id, limit)
            if cached_chats and len(cached_chats) > 0:
                chats = build_chats_from_cache(cached_chats)
                if chats:
                    return chats

        # 从数据库获取
        async with get_repository_factory() as repo:
            chats = await repo.chats.get_user_chats(user_id, limit, include_archived)

            # 缓存聊天列表
            if self.cache and chats:
                for chat in chats:
                    await self.cache.cache_chat(chat)

            return chats
    
    async def get_recent_user_chats(self, user_id: uuid.UUID, limit: int = 10) -> List[Chat]:
        """获取用户最近的聊天会话"""
        # 先尝试从缓存获取
        if self.cache:
            cached_chats = await self.cache.get_user_chats(user_id, limit)
            if cached_chats and len(cached_chats) > 0:
                chats = build_chats_from_cache(cached_chats)
                if chats:
                    return chats

        # 从数据库获取
        async with get_repository_factory() as repo:
            chats = await repo.chats.get_recent_user_chats(user_id, limit)

            # 缓存聊天列表
            if self.cache and chats:
                for chat in chats:
                    await self.cache.cache_chat(chat)

            return chats
    
    async def add_message(self, chat_id: uuid.UUID, content: str,
                          role: str, sender_id: Optional[uuid.UUID] = None,
                          message_type: str = "text",
                          metadata: Optional[Dict[str, Any]] = None) -> Message:
        """创建消息"""
        async with get_repository_factory() as repo:
            try:
                # 创建消息
                message = await repo.messages.create_message(
                    chat_id=chat_id,
                    content=content,
                    role=role,
                    sender_id=sender_id,
                    message_type=message_type,
                    metadata=metadata
                )

                # 更新聊天的最后消息时间
                await repo.chats.update_chat_last_message_time(chat_id)

                # 缓存消息
                if self.cache:
                    await self.cache.cache_message(message)

                await repo.commit()
                logger.info(f"✅ Created message: {message.id} in chat {chat_id}")
                return message

            except Exception as e:
                await repo.rollback()
                logger.error(f"❌ Failed to create message in chat {chat_id}: {e}")
                raise
    
    async def get_chat_messages(self, chat_id: uuid.UUID, limit: int = 100,
                               offset: int = 0) -> List[Message]:
        """获取聊天消息列表"""
        # 先尝试从缓存获取最近的消息
        if self.cache and offset == 0 and limit <= 20:
            cached_messages = await self.cache.get_chat_messages(chat_id)
            if cached_messages and len(cached_messages) > 0:
                messages = build_messages_from_cache(cached_messages)
                if messages:
                    return messages

        # 从数据库获取
        async with get_repository_factory() as repo:
            messages = await repo.messages.get_chat_messages(chat_id, limit, offset)

            # 缓存最近的消息
            if self.cache and offset == 0 and messages:
                for message in messages[:20]:  # 只缓存最近20条
                    await self.cache.cache_message(message)

            return messages
    
    async def get_recent_chat_messages(self, chat_id: uuid.UUID, limit: int = 20) -> List[Message]:
        """获取聊天最近的消息"""
        # 先尝试从缓存获取
        if self.cache:
            cached_messages = await self.cache.get_chat_messages(chat_id)
            if cached_messages and len(cached_messages) > 0:
                messages = build_messages_from_cache(cached_messages)
                if messages:
                    return messages

        # 从数据库获取
        async with get_repository_factory() as repo:
            messages = await repo.messages.get_recent_chat_messages(chat_id, limit)

            # 缓存最近的消息
            if self.cache and messages:
                for message in messages:
                    await self.cache.cache_message(message)

            return messages
    
    async def update_chat_title(self, chat_id: uuid.UUID, title: str, user_id: Optional[uuid.UUID] = None) -> bool:
        """更新聊天标题"""
        async with get_repository_factory() as repo:
            try:
                # 验证用户权限
                if user_id:
                    chat = await self.get_chat(chat_id)
                    if not chat or chat.user_id != user_id:
                        logger.warning(f"⚠️ User {user_id} tried to update chat {chat_id} without permission")
                        return False

                success = await repo.chats.update_chat(chat_id, title=title)

                if success:
                    # 更新缓存
                    if self.cache:
                        await self.cache.invalidate_chat_cache(chat_id)

                    await repo.commit()
                    logger.info(f"✅ Updated chat title: {chat_id}")
                else:
                    await repo.rollback()

                return success

            except Exception as e:
                await repo.rollback()
                logger.error(f"❌ Failed to update chat title {chat_id}: {e}")
                return False
    
    async def archive_chat(self, chat_id: uuid.UUID, user_id: Optional[uuid.UUID] = None) -> bool:
        """归档聊天"""
        async with get_repository_factory() as repo:
            try:
                # 验证用户权限
                if user_id:
                    chat = await self.get_chat(chat_id)
                    if not chat or chat.user_id != user_id:
                        logger.warning(f"⚠️ User {user_id} tried to archive chat {chat_id} without permission")
                        return False

                success = await repo.chats.update_chat(chat_id, is_archived=True)

                if success:
                    # 更新缓存
                    if self.cache:
                        await self.cache.invalidate_chat_cache(chat_id)

                    await repo.commit()
                    logger.info(f"✅ Archived chat: {chat_id}")
                else:
                    await repo.rollback()

                return success

            except Exception as e:
                await repo.rollback()
                logger.error(f"❌ Failed to archive chat {chat_id}: {e}")
                return False
    
    async def delete_chat(self, chat_id: uuid.UUID, user_id: Optional[uuid.UUID] = None) -> bool:
        """删除聊天"""
        async with get_repository_factory() as repo:
            try:
                # 验证用户权限
                if user_id:
                    chat = await self.get_chat(chat_id)
                    if not chat or chat.user_id != user_id:
                        logger.warning(f"⚠️ User {user_id} tried to delete chat {chat_id} without permission")
                        return False

                success = await repo.chats.delete_chat(chat_id)

                if success:
                    # 清除缓存
                    if self.cache:
                        await self.cache.invalidate_chat_cache(chat_id)

                    await repo.commit()
                    logger.info(f"✅ Deleted chat: {chat_id}")
                else:
                    await repo.rollback()

                return success

            except Exception as e:
                await repo.rollback()
                logger.error(f"❌ Failed to delete chat {chat_id}: {e}")
                return False


# 快捷函数
async def get_chat_service() -> ChatService:
    """获取聊天服务实例"""
    from app.services.cache_service import get_cache_service
    try:
        cache_service = await get_cache_service()
    except RuntimeError:
        cache_service = None
    return ChatService(cache_service)