"""
Processing Step Base Classes
处理步骤基类定义
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, AsyncIterator
import uuid
import logging


logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """
    文档块数据结构
    表示一个经过处理的文本块
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    @property
    def is_valid(self) -> bool:
        """检查块是否有效"""
        return bool(self.content and len(self.content.strip()) >= 10)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
        }
        if self.embedding is not None:
            result["embedding"] = self.embedding
        return result


@dataclass
class ProcessingContext:
    """
    处理上下文
    在流水线各步骤间传递数据和状态
    """
    # 原始数据目录
    data_dir: str = ""
    
    # 已读取的文档列表 (file_path, content, metadata)
    documents: List[Dict[str, Any]] = field(default_factory=list)
    
    # 处理后的文本块
    chunks: List[DocumentChunk] = field(default_factory=list)
    
    # 统计信息
    stats: Dict[str, int] = field(default_factory=lambda: {
        "files_read": 0,
        "files_skipped": 0,
        "chunks_created": 0,
        "chunks_stored": 0,
        "errors": 0,
    })
    
    # 错误信息
    errors: List[str] = field(default_factory=list)
    
    def add_document(self, file_path: str, content: str, metadata: Dict[str, Any] = None):
        """添加文档"""
        self.documents.append({
            "file_path": file_path,
            "content": content,
            "metadata": metadata or {},
        })
        self.stats["files_read"] += 1
    
    def add_chunk(self, chunk: DocumentChunk):
        """添加文本块"""
        self.chunks.append(chunk)
        self.stats["chunks_created"] += 1
    
    def add_error(self, error: str):
        """添加错误信息"""
        self.errors.append(error)
        self.stats["errors"] += 1
        logger.error(error)
    
    def get_summary(self) -> str:
        """获取处理摘要"""
        return (
            f"📊 Processing Summary:\n"
            f"   Files Read: {self.stats['files_read']}\n"
            f"   Files Skipped: {self.stats['files_skipped']}\n"
            f"   Chunks Created: {self.stats['chunks_created']}\n"
            f"   Chunks Stored: {self.stats['chunks_stored']}\n"
            f"   Errors: {self.stats['errors']}"
        )


class ProcessingStep(ABC):
    """
    处理步骤基类
    所有处理步骤都应继承此类
    """
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"processors.{self.name}")
    
    @abstractmethod
    async def process(self, context: ProcessingContext) -> ProcessingContext:
        """
        处理数据
        
        Args:
            context: 处理上下文
            
        Returns:
            ProcessingContext: 更新后的上下文
        """
        pass
    
    async def __call__(self, context: ProcessingContext) -> ProcessingContext:
        """使步骤可调用"""
        self.logger.info(f"🔄 Starting {self.name}...")
        try:
            result = await self.process(context)
            self.logger.info(f"✅ {self.name} completed")
            return result
        except Exception as e:
            context.add_error(f"{self.name} failed: {str(e)}")
            raise


class CompositeStep(ProcessingStep):
    """
    组合步骤
    将多个步骤组合成一个
    """
    
    def __init__(self, steps: List[ProcessingStep], name: str = "CompositeStep"):
        super().__init__(name)
        self.steps = steps
    
    async def process(self, context: ProcessingContext) -> ProcessingContext:
        """依次执行所有步骤"""
        for step in self.steps:
            context = await step(context)
        return context

