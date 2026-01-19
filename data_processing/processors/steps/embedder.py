"""
Embedding Step
向量化步骤
"""

import sys
import os
from typing import List, Optional

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .base import ProcessingStep, ProcessingContext, DocumentChunk


class EmbeddingStep(ProcessingStep):
    """
    向量化步骤
    使用 embedding 模型将文本块转换为向量
    """
    
    def __init__(
        self,
        batch_size: int = 32,
        embedding_service=None,
    ):
        """
        初始化向量化步骤
        
        Args:
            batch_size: 批处理大小
            embedding_service: 可选的 embedding 服务实例
        """
        super().__init__("EmbeddingStep")
        self.batch_size = batch_size
        self._embedding_service = embedding_service
        self._initialized = False
    
    async def _ensure_initialized(self):
        """确保 embedding 服务已初始化"""
        if self._initialized:
            return
        
        if self._embedding_service is None:
            try:
                from shared.embeddings import create_embedding_service
                self._embedding_service = create_embedding_service(use_singleton=True)
                self.logger.info(
                    f"✅ Embedding service initialized: "
                    f"{self._embedding_service.get_model_info()}"
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize embedding service: {e}")
                raise
        
        self._initialized = True
    
    async def process(self, context: ProcessingContext) -> ProcessingContext:
        """
        向量化处理
        
        Args:
            context: 包含文本块的处理上下文
            
        Returns:
            ProcessingContext: 文本块已包含向量
        """
        await self._ensure_initialized()
        
        chunks = context.chunks
        if not chunks:
            self.logger.warning("⚠️ No chunks to embed")
            return context
        
        self.logger.info(f"🔄 Embedding {len(chunks)} chunks (batch_size={self.batch_size})...")
        
        # 批量处理
        total_embedded = 0
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            texts = [chunk.content for chunk in batch]
            
            try:
                # 调用 embedding 服务
                embeddings = self._embedding_service.encode_batch(texts, batch_size=self.batch_size)
                
                # 将向量赋值给块
                for chunk, embedding in zip(batch, embeddings):
                    chunk.embedding = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
                
                total_embedded += len(batch)
                
                if total_embedded % 100 == 0 or total_embedded == len(chunks):
                    self.logger.info(f"   Embedded {total_embedded}/{len(chunks)} chunks")
                    
            except Exception as e:
                context.add_error(f"Embedding failed for batch {i//self.batch_size}: {str(e)}")
        
        self.logger.info(f"✅ Embedded {total_embedded} chunks")
        return context
    
    def get_embedding_dimension(self) -> int:
        """获取向量维度"""
        if self._embedding_service is None:
            from shared.embeddings import create_embedding_service
            self._embedding_service = create_embedding_service(use_singleton=True)
            self._initialized = True
        return self._embedding_service.get_embedding_dimension()

