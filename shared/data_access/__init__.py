"""
Data Access Layer
统一的数据访问层，同时供 data_processing 和 backend 使用

包含:
- MilvusClient: Milvus 向量数据库访问（存储 + 向量检索）
- ElasticsearchClient: Elasticsearch 访问（存储 + 关键字检索）

使用示例:
    from shared.data_access import MilvusClient, ElasticsearchClient
    
    # Milvus 向量数据库
    milvus = MilvusClient()
    await milvus.initialize()
    await milvus.store_documents(documents)  # 存储
    results = await milvus.search_similar(embedding, top_k=5)  # 向量检索
    
    # Elasticsearch 关键字搜索
    es = ElasticsearchClient()
    await es.initialize()
    await es.store_documents(documents)  # 存储
    results = await es.text_search(query, top_k=5)  # 关键字检索
"""

from .milvus import MilvusClient, MilvusConfig
from .elasticsearch import ElasticsearchClient, ElasticsearchConfig

__all__ = [
    "MilvusClient",
    "MilvusConfig",
    "ElasticsearchClient",
    "ElasticsearchConfig",
]

