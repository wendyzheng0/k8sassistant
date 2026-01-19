"""
Elasticsearch Storage Backend
基于 shared.data_access.ElasticsearchClient 的存储后端
"""

import os
from typing import Any, Dict, List, Optional

from shared.data_access import ElasticsearchClient, ElasticsearchConfig

from .base import StorageBackend, StorageResult


class ElasticsearchStorage(StorageBackend):
    """
    Elasticsearch 存储后端
    使用 shared.data_access.ElasticsearchClient 进行文档存储
    仅用于关键字检索，不需要向量化
    """
    
    requires_embedding: bool = False
    
    def __init__(
        self,
        es_url: str = "http://localhost:9200",
        index_name: str = "k8s-docs",
        username: str = None,
        password: str = None,
        batch_size: int = 200,
    ):
        """
        初始化 Elasticsearch 存储
        
        Args:
            es_url: ES 服务地址
            index_name: 索引名称
            username: 用户名
            password: 密码
            batch_size: 批量写入大小
        """
        super().__init__("ElasticsearchStorage")
        
        # 创建配置
        self._config = ElasticsearchConfig(
            es_url=es_url,
            index_name=index_name,
            username=username or os.getenv("ELASTICSEARCH_USER", "elastic"),
            password=password or os.getenv("ELASTICSEARCH_PASSWORD", "password"),
            batch_size=batch_size,
        )
        
        # 创建共享客户端
        self._client = ElasticsearchClient(self._config)
    
    @property
    def vector_store(self):
        """获取底层的 LlamaIndex Store（向后兼容）"""
        return self._client._vector_store
    
    async def initialize(self) -> None:
        """初始化 Elasticsearch 连接"""
        await self._client.initialize(for_storage=True)
        self._initialized = True
    
    async def store(self, documents: List[Dict[str, Any]]) -> StorageResult:
        """
        存储文档到 Elasticsearch
        
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
