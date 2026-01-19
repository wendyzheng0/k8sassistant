"""
Milvus vector database service
基于 shared.data_access.MilvusClient 的向量检索服务
"""

import asyncio
from typing import List, Dict, Any, Optional

from shared.data_access import MilvusClient, MilvusConfig

from app.core.config import settings
from app.core.logging import get_logger


class MilvusService:
    """
    Milvus vector database service class
    使用 shared.data_access.MilvusClient 进行向量检索
    """
    
    def __init__(self):
        self.logger = get_logger("MilvusService")
        self.collection_name = settings.COLLECTION_NAME
        self.vector_dim = settings.VECTOR_DIM
        self._embedding_service = None
        
        # 创建配置
        self._config = MilvusConfig(
            uri=settings.MILVUS_URI,
            collection_name=self.collection_name,
            vector_dim=self.vector_dim,
            overwrite=False,  # 服务端不应覆盖已有数据
            similarity_metric="COSINE",
        )
        
        # 创建共享客户端
        self._client: Optional[MilvusClient] = None
    
    def _get_actual_embedding_dimension(self) -> int:
        """Get the actual embedding dimension from the embedding service"""
        try:
            if self._embedding_service is None:
                from app.services.embedding_service import EmbeddingService
                self._embedding_service = EmbeddingService()
            return self._embedding_service.get_embedding_dimension()
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to get embedding dimension from service: {e}, using config value: {self.vector_dim}")
            return self.vector_dim
    
    async def initialize(self):
        """Initialize Milvus connection"""
        try:
            # 获取实际的嵌入维度
            actual_dim = self._get_actual_embedding_dimension()
            if actual_dim != self.vector_dim:
                self.logger.warning(f"Embedding model dimension mismatch: config={self.vector_dim}, actual={actual_dim}")
                self.vector_dim = actual_dim
                self._config.vector_dim = actual_dim
            
            self.logger.info(f"🚀 Initializing Milvus: {settings.MILVUS_URI}")
            self.logger.info(f"📊 Collection: {self.collection_name}, Dim: {self.vector_dim}")
            
            # 创建并初始化客户端
            self._client = MilvusClient(self._config)
            await self._client.initialize(for_storage=False)  # 用于检索
            
            self.logger.info(f"✅ Milvus connection initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Milvus connection: {e}")
            raise
    
    async def insert_documents(self, documents: List[Dict[str, Any]]):
        """Insert documents into vector database"""
        if not self._client:
            self.logger.error("Milvus not initialized")
            return
        
        try:
            if not documents:
                self.logger.warning("⚠️ No documents to insert")
                return
            
            result = await self._client.store_documents(documents)
            
            if result.success:
                self.logger.info(f"✅ Successfully inserted {result.stored_count} documents")
            else:
                self.logger.warning(f"⚠️ Insert completed with errors: {result.errors}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to insert documents: {e}")
            self.logger.error(f"Insertion failed, but application will continue running")
    
    async def search_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if not self._client:
            self.logger.error("Milvus not initialized")
            return []
        
        try:
            results = await self._client.search_similar(query_embedding, top_k)
            
            # 格式化为向后兼容的格式
            search_results = []
            for result in results:
                search_results.append({
                    "id": result.get("id", "unknown"),
                    "doc_id": result.get("entity", {}).get("doc_id", "unknown"),
                    "file_path": result.get("file_path", "unknown"),
                    "content": result.get("content", ""),
                    "metadata": result.get("metadata", {}),
                    "score": result.get("score"),
                    "distance": None,
                    "entity": result.get("entity", {}),
                })
            
            self.logger.info(f"🔍 Search completed, returned {len(search_results)} results, requested {top_k}")
            return search_results
            
        except Exception as e:
            self.logger.error(f"❌ Failed to search similar documents: {e}")
            return []
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        if not self._client:
            return {
                "collection_name": self.collection_name,
                "row_count": 0,
                "vector_dim": self.vector_dim,
                "status": "not_initialized"
            }
        
        try:
            stats = await self._client.get_collection_stats()
            stats["vector_dim"] = self.vector_dim
            return stats
        except Exception as e:
            self.logger.error(f"❌ Failed to get collection statistics: {e}")
            return {
                "collection_name": self.collection_name,
                "row_count": 0,
                "vector_dim": self.vector_dim,
                "status": "error"
            }
    
    async def delete_documents(self, document_ids: List[str]):
        """Delete specified documents"""
        if not self._client:
            self.logger.error("Milvus not initialized")
            return
        
        try:
            if not document_ids:
                self.logger.warning("⚠️ No document IDs to delete")
                return
            
            deleted = await self._client.delete_documents(document_ids)
            self.logger.info(f"✅ Successfully deleted {deleted} documents")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to delete documents: {e}")
            self.logger.error(f"Deletion failed, but application will continue running")
    
    async def get_chunks_by_file_path(self, file_path: str) -> List[Dict[str, Any]]:
        """Get all chunks associated with a file path"""
        if not self._client:
            self.logger.error("Milvus not initialized")
            return []
        
        try:
            chunks = await self._client.get_chunks_by_file_path(file_path)
            self.logger.info(f"🔍 Found {len(chunks)} chunks for file: {file_path}")
            return chunks
        except Exception as e:
            self.logger.error(f"❌ Failed to get chunks by file path: {e}")
            return []
    
    async def close(self):
        """Close connection"""
        try:
            if self._client:
                await self._client.close()
            self.logger.info("✅ Milvus connection closed")
        except Exception as e:
            self.logger.error(f"❌ Failed to close Milvus connection: {e}")
    
    def __del__(self):
        """Destructor, ensure connection is closed"""
        try:
            if self._client:
                asyncio.create_task(self.close())
        except:
            pass
