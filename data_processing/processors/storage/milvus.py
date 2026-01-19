"""
Milvus Storage Backend
基于 shared.data_access.MilvusClient 的向量存储后端
"""

from typing import Any, Dict, List, Optional

from shared.data_access import MilvusClient, MilvusConfig

from .base import StorageBackend, StorageResult


class MilvusStorage(StorageBackend):
    """
    Milvus 存储后端
    使用 shared.data_access.MilvusClient 进行文档存储
    """
    
    requires_embedding: bool = True
    
    def __init__(
        self,
        uri: str = "http://localhost:19530",
        collection_name: str = "k8s_docs",
        vector_dim: int = 512,
        overwrite: bool = True,
        consistency_level: str = "Strong",
        similarity_metric: str = "COSINE",
    ):
        """
        初始化 Milvus 存储
        
        Args:
            uri: Milvus 服务地址
            collection_name: 集合名称
            vector_dim: 向量维度
            overwrite: 是否覆盖已存在的集合
            consistency_level: 一致性级别（保留参数，由 shared 模块处理）
            similarity_metric: 相似度度量类型
        """
        super().__init__("MilvusStorage")
        
        # 创建配置
        self._config = MilvusConfig(
            uri=uri,
            collection_name=collection_name,
            vector_dim=vector_dim,
            overwrite=overwrite,
            similarity_metric=similarity_metric,
        )
        
        # 创建共享客户端
        self._client = MilvusClient(self._config)
    
    @property
    def vector_store(self):
        """获取底层的 LlamaIndex VectorStore（向后兼容）"""
        return self._client._vector_store
    
    async def initialize(self) -> None:
        """初始化 Milvus 连接"""
        await self._client.initialize(for_storage=True)
        self._initialized = True
    
    async def store(self, documents: List[Dict[str, Any]]) -> StorageResult:
        """
        存储文档到 Milvus
        
        Args:
            documents: 文档列表，每个文档需要包含 id, content, embedding
            
        Returns:
            StorageResult: 存储结果
        """
        result = await self._client.store_documents(documents)
        
        # 转换为本地 StorageResult 类型
        return StorageResult(
            success=result.success,
            stored_count=result.stored_count,
            error_count=result.error_count,
            errors=result.errors,
        )
    
    async def close(self) -> None:
        """关闭连接"""
        await self._client.close()
        self._initialized = False
