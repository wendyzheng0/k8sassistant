"""
Storage Backend Base Classes
存储后端基类定义
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging


logger = logging.getLogger(__name__)


@dataclass
class StorageResult:
    """
    存储操作结果
    """
    success: bool = True
    stored_count: int = 0
    error_count: int = 0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, error: str):
        """添加错误信息"""
        self.errors.append(error)
        self.error_count += 1
        self.success = False
    
    def __str__(self) -> str:
        if self.success:
            return f"✅ Stored {self.stored_count} documents successfully"
        else:
            return (
                f"⚠️ Stored {self.stored_count} documents with "
                f"{self.error_count} errors: {self.errors[:3]}..."
            )


class StorageBackend(ABC):
    """
    存储后端基类
    所有存储后端实现都应继承此类
    """
    
    # 子类可以覆盖此属性来指示是否需要向量化
    requires_embedding: bool = True
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"storage.{self.name}")
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        初始化存储连接
        应该在使用前调用
        """
        pass
    
    @abstractmethod
    async def store(self, documents: List[Dict[str, Any]]) -> StorageResult:
        """
        存储文档
        
        Args:
            documents: 文档列表，每个文档应包含:
                - id: 文档 ID
                - content: 文本内容
                - metadata: 元数据字典
                - embedding: 向量（如果适用）
                
        Returns:
            StorageResult: 存储操作结果
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """
        关闭存储连接
        """
        pass
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.close()
    
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized

