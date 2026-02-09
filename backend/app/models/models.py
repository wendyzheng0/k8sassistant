"""
数据库模型定义
包含用户、聊天和消息的数据库模型
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import (
    Column, String, Boolean, DateTime, Text, ForeignKey, JSON,
    Integer, Index, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
import uuid

Base = declarative_base()


class User(Base):
    """用户模型"""
    __tablename__ = "users"
    __table_args__ = {'schema': 'app_k8sassist'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_anonymous = Column(Boolean, default=False, nullable=False)
    anonymous_id = Column(UUID(as_uuid=True), unique=True, nullable=True, index=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # 关联关系
    chats = relationship("Chat", back_populates="user", cascade="all, delete-orphan")
    messages = relationship("Message", back_populates="sender", foreign_keys="Message.sender_id")
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, email={self.email})>"


class Chat(Base):
    """聊天会话模型"""
    __tablename__ = "chats"
    __table_args__ = {'schema': 'app_k8sassist'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("app_k8sassist.users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    last_message_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    is_archived = Column(Boolean, default=False, nullable=False)
    
    # 关联关系
    user = relationship("User", back_populates="chats")
    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan", 
                         order_by="Message.created_at.desc()")
    
    def __repr__(self):
        return f"<Chat(id={self.id}, user_id={self.user_id}, title={self.title})>"


class Message(Base):
    """消息模型"""
    __tablename__ = "messages"
    __table_args__ = {'schema': 'app_k8sassist'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_id = Column(UUID(as_uuid=True), ForeignKey("app_k8sassist.chats.id", ondelete="CASCADE"), nullable=False, index=True)
    content = Column(Text, nullable=False)  # 消息内容
    message_type = Column(String(20), nullable=False, default="text", index=True)  # 消息类型
    role = Column(String(20), nullable=False, index=True)  # 发送者角色: 'user' or 'assistant'
    sender_id = Column(UUID(as_uuid=True), ForeignKey("app_k8sassist.users.id"), nullable=True, index=True)  # 发送者ID
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    meta_info = Column(JSON, nullable=True)  # 额外信息
    
    # 关联关系
    chat = relationship("Chat", back_populates="messages")
    sender = relationship("User", back_populates="messages", foreign_keys=[sender_id])
    
    def __repr__(self):
        return f"<Message(id={self.id}, chat_id={self.chat_id}, role={self.role})>"


# 创建复合索引
Index('idx_chats_user_updated', Chat.user_id, Chat.updated_at.desc())
Index('idx_messages_chat_created', Message.chat_id, Message.created_at.desc())


# 数据库工具函数
def create_tables(engine):
    """创建所有数据库表"""
    Base.metadata.create_all(engine)


def drop_tables(engine):
    """删除所有数据库表"""
    Base.metadata.drop_all(engine)