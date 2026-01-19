"""
Document Processors Package
文档处理模块 - 用于处理下载的文档并建立知识库

================================================================================
通用流水线（用于 Elasticsearch 等非向量存储）
================================================================================

    from data_processing.processors import PipelineRunner
    
    runner = PipelineRunner()
    result = await runner.run(
        data_dir="./data/zh-cn",
        storage_backend="elasticsearch",  # 关键字搜索，不需要向量化
    )
"""

from .config import ProcessorSettings, get_processor_settings
from .runner import (
    PipelineRunner,
    PipelineResult,
    run_pipeline,
)
from .pipelines import (
    PipelineConfig,
    StepConfig,
    StepType,
    get_pipeline_config,
    register_pipeline,
    create_milvus_pipeline,
    create_elasticsearch_pipeline,
    create_hybrid_pipeline,
)
from .steps import (
    ProcessingStep,
    ProcessingContext,
    DocumentChunk,
    DocumentReader,
    HTMLProcessor,
    TextChunker,
    EmbeddingStep,
)
from .storage import (
    StorageBackend,
    StorageResult,
    MilvusStorage,
    ElasticsearchStorage,
    create_storage_backend,
)

__all__ = [
    # 配置
    "ProcessorSettings",
    "get_processor_settings",
    
    # 通用流水线
    "PipelineRunner",
    "PipelineResult",
    "run_pipeline",
    
    # 流水线配置
    "PipelineConfig",
    "StepConfig",
    "StepType",
    "get_pipeline_config",
    "register_pipeline",
    "create_milvus_pipeline",
    "create_elasticsearch_pipeline",
    "create_hybrid_pipeline",
    
    # 处理步骤
    "ProcessingStep",
    "ProcessingContext",
    "DocumentChunk",
    "DocumentReader",
    "HTMLProcessor",
    "TextChunker",
    "EmbeddingStep",
    
    # 存储后端
    "StorageBackend",
    "StorageResult",
    "MilvusStorage",
    "ElasticsearchStorage",
    "create_storage_backend",
]

__version__ = "2.0.0"
