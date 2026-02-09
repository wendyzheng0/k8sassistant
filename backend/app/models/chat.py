"""
聊天相关数据模型
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class MessageRole(str, Enum):
    """消息角色枚举"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageType(str, Enum):
    """消息类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    ERROR = "error"


class ChatMessage(BaseModel):
    """聊天消息模型"""
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(..., description="消息唯一标识")
    role: MessageRole = Field(..., description="消息角色")
    content: str = Field(..., description="消息内容")
    # Use camelCase alias for frontend compatibility
    message_type: MessageType = Field(default=MessageType.TEXT, alias="messageType", description="消息类型")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="消息时间戳")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="元数据")


class ChatRequest(BaseModel):
    """聊天请求模型"""
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "message": "如何创建一个 Kubernetes Pod？",
                "conversationId": "conv_123456",
                "temperature": 0.7,
                "maxTokens": 2048
            }
        }
    )

    message: str = Field(..., description="用户消息", min_length=1, max_length=2000)
    conversation_id: Optional[str] = Field(default=None, alias="conversationId", description="对话ID")
    context: Optional[List[ChatMessage]] = Field(default=None, description="对话上下文")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="生成温度")
    max_tokens: Optional[int] = Field(default=4096, alias="maxTokens", ge=1, le=8192, description="最大token数")


class ChatResponse(BaseModel):
    """聊天响应模型"""
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "messageId": "msg_123456",
                "content": "要创建一个 Kubernetes Pod，你可以使用以下 YAML 配置...",
                "conversationId": "conv_123456",
                "sources": [
                    {
                        "title": "Kubernetes Pod 文档",
                        "url": "https://kubernetes.io/docs/concepts/workloads/pods/",
                        "score": 0.95
                    }
                ]
            }
        }
    )

    message_id: str = Field(..., alias="messageId", description="消息ID")
    content: str = Field(..., description="回复内容")
    conversation_id: str = Field(..., alias="conversationId", description="对话ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="回复时间")
    sources: Optional[List[Dict[str, Any]]] = Field(default=None, description="参考来源")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="元数据")


class Conversation(BaseModel):
    """对话模型"""
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(..., description="对话唯一标识")
    title: str = Field(..., description="对话标题")
    messages: List[ChatMessage] = Field(default=[], description="消息列表")
    created_at: datetime = Field(default_factory=datetime.utcnow, alias="createdAt", description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.utcnow, alias="updatedAt", description="更新时间")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="元数据")


class SearchResult(BaseModel):
    """搜索结果模型"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "doc_123",
                "title": "Kubernetes Pod 概念",
                "content": "Pod 是 Kubernetes 中最小的可部署计算单元...",
                "url": "https://kubernetes.io/docs/concepts/workloads/pods/",
                "score": 0.95
            }
        }
    )

    id: str = Field(..., description="文档ID")
    title: str = Field(..., description="文档标题")
    content: str = Field(..., description="文档内容片段")
    url: Optional[str] = Field(default=None, description="文档URL")
    score: float = Field(..., description="相似度得分")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="元数据")


class ChatHistoryResponse(BaseModel):
    """聊天历史响应模型"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "conversations": [],
                "total": 0,
                "page": 1,
                "size": 20
            }
        }
    )
    conversations: List[Conversation] = Field(..., description="对话列表")
    total: int = Field(..., description="总数")
    page: int = Field(default=1, description="当前页码")
    size: int = Field(default=20, description="每页大小")