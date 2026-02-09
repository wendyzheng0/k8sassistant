"""
缓存服务
提供 Redis 缓存功能，包括用户会话、聊天消息等缓存
"""

import json
import uuid
from typing import Optional, List, Dict, Any
from datetime import timedelta
import redis.asyncio as redis
from loguru import logger

from app.core.config import settings
from app.models.models import User, Chat, Message

_redis_pool: Optional[redis.ConnectionPool] = None


def get_redis_client() -> redis.Redis:
    """获取 Redis 客户端"""
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = redis.ConnectionPool.from_url(settings.REDIS_URL, decode_responses=True)
    return redis.Redis(connection_pool=_redis_pool)


class CacheService:
    """缓存服务类"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.ttl = settings.CACHE_TTL
    
    def _serialize_metadata(self, metadata: Any) -> Any:
        """安全地序列化 metadata 为 JSON 兼容格式"""
        if metadata is None:
            return None
        
        try:
            # 如果已经是基本类型，直接返回
            if isinstance(metadata, (str, int, float, bool)) or metadata is None:
                return metadata
            
            # 如果是字典，递归处理
            if isinstance(metadata, dict):
                return {k: self._serialize_metadata(v) for k, v in metadata.items()}
            
            # 如果是列表或元组，递归处理
            if isinstance(metadata, (list, tuple)):
                return [self._serialize_metadata(item) for item in metadata]
            
            # 如果是 UUID 对象，转换为字符串
            if hasattr(metadata, '__class__') and metadata.__class__.__name__ == 'UUID':
                return str(metadata)
            
            # 如果是 datetime 对象，转换为 ISO 格式字符串
            if hasattr(metadata, 'isoformat'):
                return metadata.isoformat()
            
            # 尝试直接 JSON 序列化
            import json
            json.dumps(metadata)
            return metadata
            
        except (TypeError, ValueError, AttributeError) as e:
            logger.warning(f"⚠️ Failed to serialize metadata item: {type(metadata)} - {e}")
            return None
        
    def _get_user_key(self, user_id: uuid.UUID) -> str:
        """获取用户缓存键"""
        return f"user:{user_id}"
    
    def _get_anonymous_key(self, anonymous_id: uuid.UUID) -> str:
        """获取匿名用户映射键"""
        return f"anonymous:{anonymous_id}"
    
    def _get_chat_key(self, chat_id: uuid.UUID) -> str:
        """获取聊天缓存键"""
        return f"chat:{chat_id}"
    
    def _get_user_chats_key(self, user_id: uuid.UUID) -> str:
        """获取用户聊天列表键"""
        return f"user_chats:{user_id}"
    
    def _get_chat_messages_key(self, chat_id: uuid.UUID) -> str:
        """获取聊天消息列表键"""
        return f"chat_messages:{chat_id}"
    
    async def cache_user(self, user: User) -> None:
        """缓存用户信息"""
        try:
            user_data = {
                "id": str(user.id),
                "username": user.username,
                "email": user.email,
                "is_active": user.is_active,
                "is_anonymous": user.is_anonymous,
                "anonymous_id": str(user.anonymous_id) if user.anonymous_id else None,
                "created_at": user.created_at.isoformat() if user.created_at else None,
                "updated_at": user.updated_at.isoformat() if user.updated_at else None
            }
            
            key = self._get_user_key(user.id)
            await self.redis.setex(
                key, 
                timedelta(seconds=self.ttl), 
                json.dumps(user_data)
            )
            
            # 如果是匿名用户，也缓存匿名ID映射
            if user.is_anonymous and user.anonymous_id:
                anon_key = self._get_anonymous_key(user.anonymous_id)
                await self.redis.setex(
                    anon_key,
                    timedelta(seconds=self.ttl),
                    str(user.id)
                )
                
            logger.debug(f"✅ Cached user: {user.id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to cache user {user.id}: {e}")
    
    async def get_cached_user(self, user_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """获取缓存的用户信息"""
        try:
            key = self._get_user_key(user_id)
            data = await self.redis.get(key)
            
            if data:
                user_data = json.loads(data)
                logger.debug(f"✅ Found cached user: {user_id}")
                return user_data
                
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get cached user {user_id}: {e}")
            return None
    
    async def get_user_by_anonymous_id(self, anonymous_id: uuid.UUID) -> Optional[str]:
        """通过匿名ID获取用户ID"""
        try:
            key = self._get_anonymous_key(anonymous_id)
            user_id = await self.redis.get(key)
            return user_id
            
        except Exception as e:
            logger.error(f"❌ Failed to get user by anonymous ID {anonymous_id}: {e}")
            return None
    
    async def cache_chat(self, chat: Chat) -> None:
        """缓存聊天信息"""
        try:
            chat_data = {
                "id": str(chat.id),
                "user_id": str(chat.user_id),
                "title": chat.title,
                "created_at": chat.created_at.isoformat() if chat.created_at else None,
                "updated_at": chat.updated_at.isoformat() if chat.updated_at else None,
                "last_message_at": chat.last_message_at.isoformat() if chat.last_message_at else None,
                "is_archived": chat.is_archived
            }
            
            key = self._get_chat_key(chat.id)
            await self.redis.setex(
                key,
                timedelta(seconds=self.ttl),
                json.dumps(chat_data)
            )
            
            logger.debug(f"✅ Cached chat: {chat.id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to cache chat {chat.id}: {e}")
    
    async def get_cached_chat(self, chat_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """获取缓存的聊天信息"""
        try:
            key = self._get_chat_key(chat_id)
            data = await self.redis.get(key)
            
            if data:
                chat_data = json.loads(data)
                logger.debug(f"✅ Found cached chat: {chat_id}")
                return chat_data
                
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get cached chat {chat_id}: {e}")
            return None
    
    async def add_chat_to_user_list(self, user_id: uuid.UUID, chat_id: uuid.UUID) -> None:
        """将聊天添加到用户聊天列表（LRU策略）"""
        try:
            key = self._get_user_chats_key(user_id)
            
            # 使用 Redis 列表，左端插入最新元素
            await self.redis.rpush(key, str(chat_id))
            
            # 限制列表长度，保持最近10个聊天
            await self.redis.ltrim(key, 0, settings.MAX_CACHED_CHATS - 1)
            
            # 设置过期时间
            await self.redis.expire(key, self.ttl)
            
            logger.debug(f"✅ Added chat {chat_id} to user {user_id} chat list")
            
        except Exception as e:
            logger.error(f"❌ Failed to add chat {chat_id} to user {user_id} list: {e}")
    
    async def get_user_recent_chats(self, user_id: uuid.UUID) -> List[str]:
        """获取用户最近的聊天列表"""
        try:
            key = self._get_user_chats_key(user_id)
            chat_ids = await self.redis.lrange(key, 0, -1)
            return chat_ids
            
        except Exception as e:
            logger.error(f"❌ Failed to get user {user_id} recent chats: {e}")
            return []
    
    async def get_user_chats(self, user_id: uuid.UUID, limit: int = None) -> List[Dict[str, Any]]:
        """获取用户的聊天列表（完整数据）"""
        try:
            # 获取聊天ID列表
            chat_ids = await self.get_user_recent_chats(user_id)
            
            # 限制数量
            if limit:
                chat_ids = chat_ids[:limit]
            
            # 获取每个聊天的完整数据
            chats = []
            for chat_id in chat_ids:
                try:
                    chat_data = await self.get_cached_chat(uuid.UUID(chat_id))
                    if chat_data:
                        chats.append(chat_data)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to get cached chat {chat_id}: {e}")
                    continue
            
            return chats
            
        except Exception as e:
            logger.error(f"❌ Failed to get user {user_id} chats: {e}")
            return []
    
    async def cache_message(self, message: Message) -> None:
        """缓存消息"""
        try:
            # 获取 metadata 字段（可能是 meta_info 或 metadata）
            metadata = getattr(message, 'metadata', None)
            if metadata is None:
                metadata = getattr(message, 'meta_info', None)
            
            message_data = {
                "id": str(message.id),
                "chat_id": str(message.chat_id),
                "content": message.content,
                "message_type": message.message_type,
                "role": message.role,
                "sender_id": str(message.sender_id) if message.sender_id else None,
                "created_at": message.created_at.isoformat() if message.created_at else None,
                "metadata": self._serialize_metadata(metadata)
            }
            
            key = self._get_chat_messages_key(message.chat_id)
            
            # 使用 Redis 列表，左端插入最新消息
            await self.redis.rpush(key, json.dumps(message_data))
            
            # 限制消息数量，保持最近20条
            await self.redis.ltrim(key, 0, settings.MAX_CACHED_MESSAGES - 1)
            
            # 设置过期时间
            await self.redis.expire(key, self.ttl)
            
            logger.debug(f"✅ Cached message: {message.id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to cache message {message.id}: {e}")
    
    async def get_chat_messages(self, chat_id: uuid.UUID, limit: int = None) -> List[Dict[str, Any]]:
        """获取聊天消息列表"""
        try:
            key = self._get_chat_messages_key(chat_id)
            limit = limit or settings.MAX_CACHED_MESSAGES
            
            # 从 Redis 获取消息（按时间倒序）
            message_data_list = await self.redis.lrange(key, 0, limit - 1)
            
            messages = []
            for message_data in message_data_list:
                try:
                    message_dict = json.loads(message_data)
                    messages.append(message_dict)
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ Failed to decode message data: {message_data}")
                    continue
            
            return messages
            
        except Exception as e:
            logger.error(f"❌ Failed to get chat {chat_id} messages: {e}")
            return []
    
    async def invalidate_user_cache(self, user_id: uuid.UUID) -> None:
        """清除用户相关缓存"""
        try:
            # 清除用户信息
            await self.redis.delete(self._get_user_key(user_id))
            
            # 清除用户聊天列表
            await self.redis.delete(self._get_user_chats_key(user_id))
            
            logger.debug(f"✅ Invalidated user cache: {user_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to invalidate user {user_id} cache: {e}")
    
    async def invalidate_chat_cache(self, chat_id: uuid.UUID) -> None:
        """清除聊天相关缓存"""
        try:
            # 清除聊天信息
            await self.redis.delete(self._get_chat_key(chat_id))
            
            # 清除聊天消息
            await self.redis.delete(self._get_chat_messages_key(chat_id))
            
            logger.debug(f"✅ Invalidated chat cache: {chat_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to invalidate chat {chat_id} cache: {e}")
    
    async def merge_anonymous_data(self, anonymous_user_id: uuid.UUID, login_user_id: uuid.UUID) -> None:
        """合并匿名用户数据到登录用户"""
        try:
            # 获取匿名用户的聊天列表
            anon_chats_key = self._get_user_chats_key(anonymous_user_id)
            anon_chat_ids = await self.redis.lrange(anon_chats_key, 0, -1)
            
            # 将匿名用户的聊天添加到登录用户的聊天列表
            for chat_id in anon_chat_ids:
                await self.add_chat_to_user_list(login_user_id, uuid.UUID(chat_id))
            
            # 清除匿名用户的缓存
            await self.invalidate_user_cache(anonymous_user_id)
            
            logger.info(f"✅ Merged anonymous user {anonymous_user_id} data to login user {login_user_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to merge anonymous data: {e}")


# 全局缓存服务实例
_cache_service: Optional[CacheService] = None


async def get_cache_service() -> CacheService:
    """获取缓存服务实例"""
    global _cache_service
    if _cache_service is None:
        from app.core.database import db_manager
        if db_manager.redis_client:
            _cache_service = CacheService(db_manager.redis_client)
        else:
            raise RuntimeError("Redis client not initialized")
    return _cache_service
