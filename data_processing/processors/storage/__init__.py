"""
Storage Backend Package
存储后端模块，支持多种向量数据库
"""

from .base import StorageBackend, StorageResult
from .milvus import MilvusStorage
from .elasticsearch import ElasticsearchStorage

__all__ = [
    "StorageBackend",
    "StorageResult",
    "MilvusStorage",
    "ElasticsearchStorage",
]


def create_storage_backend(backend: str, **kwargs) -> StorageBackend:
    """
    工厂函数：根据类型创建存储后端
    
    Args:
        backend: 存储后端类型 ("milvus" | "elasticsearch")
        **kwargs: 传递给后端的配置参数
        
    Returns:
        StorageBackend: 存储后端实例
    """
    if backend == "milvus":
        return MilvusStorage(**kwargs)
    elif backend == "elasticsearch":
        return ElasticsearchStorage(**kwargs)
    else:
        raise ValueError(f"Unknown storage backend: {backend}")

