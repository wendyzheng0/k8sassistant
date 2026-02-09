"""
缓存工具函数
提供缓存数据与 ORM 对象之间的转换功能
"""

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from app.models.models import User, Chat, Message


def build_user_from_cache(cache_data: Dict[str, Any]) -> Optional[User]:
    """
    从缓存数据构建 User 对象
    
    Args:
        cache_data: 从缓存获取的用户数据字典
        
    Returns:
        User 对象，如果数据不完整则返回 None
    """
    try:
        return User(
            id=uuid.UUID(cache_data['id']),
            username=cache_data['username'],
            email=cache_data['email'],
            is_active=cache_data['is_active'],
            is_anonymous=cache_data['is_anonymous'],
            anonymous_id=uuid.UUID(cache_data['anonymous_id']) if cache_data.get('anonymous_id') else None,
            created_at=datetime.fromisoformat(cache_data['created_at']) if cache_data.get('created_at') else None,
            updated_at=datetime.fromisoformat(cache_data['updated_at']) if cache_data.get('updated_at') else None
        )
    except (KeyError, ValueError, TypeError) as e:
        return None


def build_chat_from_cache(cache_data: Dict[str, Any]) -> Optional[Chat]:
    """
    从缓存数据构建 Chat 对象
    
    Args:
        cache_data: 从缓存获取的聊天数据字典
        
    Returns:
        Chat 对象，如果数据不完整则返回 None
    """
    try:
        return Chat(
            id=uuid.UUID(cache_data['id']),
            user_id=uuid.UUID(cache_data['user_id']),
            title=cache_data['title'],
            created_at=datetime.fromisoformat(cache_data['created_at']) if cache_data.get('created_at') else None,
            updated_at=datetime.fromisoformat(cache_data['updated_at']) if cache_data.get('updated_at') else None,
            last_message_at=datetime.fromisoformat(cache_data['last_message_at']) if cache_data.get('last_message_at') else None,
            is_archived=cache_data['is_archived']
        )
    except (KeyError, ValueError, TypeError) as e:
        return None


def build_message_from_cache(cache_data: Dict[str, Any]) -> Optional[Message]:
    """
    从缓存数据构建 Message 对象
    
    Args:
        cache_data: 从缓存获取的消息数据字典
        
    Returns:
        Message 对象，如果数据不完整则返回 None
    """
    try:
        return Message(
            id=uuid.UUID(cache_data['id']),
            chat_id=uuid.UUID(cache_data['chat_id']),
            content=cache_data['content'],
            message_type=cache_data['message_type'],
            role=cache_data['role'],
            sender_id=uuid.UUID(cache_data['sender_id']) if cache_data.get('sender_id') else None,
            created_at=datetime.fromisoformat(cache_data['created_at']) if cache_data.get('created_at') else None,
            updated_at=datetime.fromisoformat(cache_data['updated_at']) if cache_data.get('updated_at') else None,
            meta_info=cache_data.get('metadata')
        )
    except (KeyError, ValueError, TypeError) as e:
        return None


def build_users_from_cache(cache_data_list: List[Dict[str, Any]]) -> List[User]:
    """
    从缓存数据列表构建 User 对象列表
    
    Args:
        cache_data_list: 从缓存获取的用户数据字典列表
        
    Returns:
        User 对象列表
    """
    users = []
    for cache_data in cache_data_list:
        user = build_user_from_cache(cache_data)
        if user:
            users.append(user)
    return users


def build_chats_from_cache(cache_data_list: List[Dict[str, Any]]) -> List[Chat]:
    """
    从缓存数据列表构建 Chat 对象列表
    
    Args:
        cache_data_list: 从缓存获取的聊天数据字典列表
        
    Returns:
        Chat 对象列表
    """
    chats = []
    for cache_data in cache_data_list:
        chat = build_chat_from_cache(cache_data)
        if chat:
            chats.append(chat)
    return chats


def build_messages_from_cache(cache_data_list: List[Dict[str, Any]]) -> List[Message]:
    """
    从缓存数据列表构建 Message 对象列表
    
    Args:
        cache_data_list: 从缓存获取的消息数据字典列表
        
    Returns:
        Message 对象列表
    """
    messages = []
    for cache_data in cache_data_list:
        message = build_message_from_cache(cache_data)
        if message:
            messages.append(message)
    return messages