"""
Document Reader Step
文档读取步骤 - 使用 LlamaIndex SimpleDirectoryReader
"""

import os
from pathlib import Path
from typing import List

from llama_index.core import SimpleDirectoryReader

from .base import ProcessingStep, ProcessingContext


class DocumentReader(ProcessingStep):
    """
    文档读取器
    基于 LlamaIndex SimpleDirectoryReader 从指定目录读取文档
    """
    
    def __init__(
        self,
        extensions: List[str] = None,
        exclude_patterns: List[str] = None,
        recursive: bool = True,
    ):
        """
        初始化文档读取器
        
        Args:
            extensions: 要读取的文件扩展名列表，如 [".html", ".md"]
            exclude_patterns: 要排除的路径模式列表，如 ["_print"]
            recursive: 是否递归读取子目录
        """
        super().__init__("DocumentReader")
        self.extensions = extensions or [".html"]
        self.exclude_patterns = exclude_patterns or ["_print"]
        self.recursive = recursive
    
    async def process(self, context: ProcessingContext) -> ProcessingContext:
        """
        读取文档
        
        Args:
            context: 处理上下文，需要包含 data_dir
            
        Returns:
            ProcessingContext: 包含读取的文档
        """
        data_dir = context.data_dir
        if not data_dir or not os.path.exists(data_dir):
            context.add_error(f"Data directory does not exist: {data_dir}")
            return context
        
        self.logger.info(f"📂 Reading documents from: {data_dir}")
        self.logger.info(f"   Extensions: {self.extensions}")
        self.logger.info(f"   Exclude patterns: {self.exclude_patterns}")
        
        # 使用 LlamaIndex SimpleDirectoryReader
        reader = SimpleDirectoryReader(
            input_dir=data_dir,
            required_exts=self.extensions,
            recursive=self.recursive,
            exclude_hidden=True,
        )
        
        # 读取并过滤文档
        for docs in reader.iter_data():
            for doc in docs:
                file_path = doc.metadata.get("file_path", "")
                rel_path = os.path.relpath(file_path, data_dir) if file_path else ""
                
                # 检查排除模式
                if self._should_exclude(rel_path):
                    self.logger.debug(f"🚫 Skipping: {rel_path}")
                    context.stats["files_skipped"] += 1
                    continue
                
                # 添加到上下文
                # NOTE:
                # LlamaIndex's doc.metadata may already contain keys like `file_type`.
                # We want OUR normalized values (e.g. ".html") to win so downstream
                # steps (HTMLProcessor) don't accidentally skip HTML processing.
                #
                # IMPORTANT: 使用相对路径作为 file_path，因为 LlamaIndex MilvusVectorStore
                # 会把 metadata 中的 file_path 提升到实体顶层，我们希望存储和检索时
                # 都使用相对路径以确保一致性
                metadata = dict(doc.metadata or {})
                metadata.update(
                    {
                        "file_path": rel_path,  # 使用相对路径，确保 Milvus/ES 存储一致
                        "absolute_path": file_path,  # 保留绝对路径以备需要
                        "relative_path": rel_path,
                        "file_name": os.path.basename(file_path),
                        "file_type": Path(file_path).suffix.lower(),
                        "file_size": len(doc.text),
                    }
                )
                
                context.add_document(file_path, doc.text, metadata)
                self.logger.debug(f"📄 Read: {rel_path}")
        
        self.logger.info(
            f"✅ Read {context.stats['files_read']} files, "
            f"skipped {context.stats['files_skipped']}"
        )
        return context
    
    def _should_exclude(self, rel_path: str) -> bool:
        """检查是否应该排除该路径"""
        for pattern in self.exclude_patterns:
            if pattern in rel_path:
                return True
        return False
