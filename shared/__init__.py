"""
Shared modules for k8sassistant
Contains common components used by both backend and data_processing

包含以下共享模块:
- config: 统一配置管理
- embeddings: 嵌入向量服务
- data_access: 数据访问层 (Milvus, Elasticsearch)
- llm_providers: LLM 服务提供者

使用示例:
    # 配置
    from shared.config import get_settings
    settings = get_settings()
    
    # 嵌入服务
    from shared.embeddings import create_embedding_service
    embedding_service = create_embedding_service()
    
    # 数据访问
    from shared.data_access import MilvusClient, ElasticsearchClient
    milvus = MilvusClient()
    await milvus.initialize()
"""

from pathlib import Path

# Get the shared module root directory
SHARED_ROOT = Path(__file__).parent

__all__ = ["SHARED_ROOT"]
