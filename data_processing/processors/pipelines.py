"""
Pipeline Configurations
流水线配置 - 为不同存储后端定义独立的处理流程
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Type, Optional
from enum import Enum

from .steps.base import ProcessingStep


class StepType(Enum):
    """处理步骤类型"""
    READER = "reader"
    HTML_PROCESSOR = "html_processor"
    CHUNKER = "chunker"
    EMBEDDER = "embedder"
    # 可扩展更多步骤类型
    TITLE_EXTRACTOR = "title_extractor"
    KEYWORD_EXTRACTOR = "keyword_extractor"
    CODE_EXTRACTOR = "code_extractor"


@dataclass
class StepConfig:
    """
    步骤配置
    定义单个处理步骤及其参数
    """
    step_type: StepType
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class PipelineConfig:
    """
    流水线配置
    定义完整的处理流程
    """
    name: str
    description: str
    steps: List[StepConfig] = field(default_factory=list)
    
    def add_step(self, step_type: StepType, enabled: bool = True, **params) -> "PipelineConfig":
        """添加步骤（支持链式调用）"""
        self.steps.append(StepConfig(step_type=step_type, enabled=enabled, params=params))
        return self
    
    def get_enabled_steps(self) -> List[StepConfig]:
        """获取启用的步骤"""
        return [s for s in self.steps if s.enabled]


# ============================================================
# 预定义流水线配置
# ============================================================

def create_milvus_pipeline() -> PipelineConfig:
    """
    Milvus 向量搜索流水线
    
    流程：读取 → HTML处理 → 分块 → 向量化
    """
    config = PipelineConfig(
        name="milvus_vector_search",
        description="Milvus 向量搜索流水线：支持语义搜索"
    )
    
    # 1. 读取文档
    config.add_step(StepType.READER)
    
    # 2. HTML 处理
    config.add_step(StepType.HTML_PROCESSOR, extract_codes=True)
    
    # 3. 文本分块（向量搜索需要分块以提高检索精度）
    config.add_step(StepType.CHUNKER)
    
    # 4. 向量化
    config.add_step(StepType.EMBEDDER)
    
    return config


def create_elasticsearch_pipeline() -> PipelineConfig:
    """
    Elasticsearch 关键字搜索流水线
    
    流程：读取 → HTML处理 → 分块
    注意：关键字搜索不需要向量化，但需要分块以保持与 Milvus 相同的粒度
    这样可以在 RRF reranker 中正确融合两边的结果
    
    重要：HTML 处理配置必须与 Milvus pipeline 保持一致（extract_codes=True），
    否则处理后的文本内容不同，会导致分块数量和 doc_id 不匹配
    """
    config = PipelineConfig(
        name="elasticsearch_keyword_search",
        description="Elasticsearch 关键字搜索流水线：支持全文检索（与Milvus统一分块粒度）"
    )
    
    # 1. 读取文档
    config.add_step(StepType.READER)
    
    # 2. HTML 处理（与 Milvus 保持相同配置，确保文本内容一致）
    config.add_step(StepType.HTML_PROCESSOR, extract_codes=True)
    
    # 3. 文本分块（与 Milvus 保持相同的分块策略，确保粒度一致）
    # 这样两边的 doc_id 可以匹配，便于 reranker 融合
    config.add_step(StepType.CHUNKER)
    
    # 注意：关键字搜索不需要向量化，ES 自己会处理文本分词
    
    return config


def create_hybrid_pipeline() -> PipelineConfig:
    """
    混合搜索流水线（向量 + 关键字）
    
    适用于同时需要语义搜索和关键字搜索的场景
    """
    config = PipelineConfig(
        name="hybrid_search",
        description="混合搜索流水线：同时支持语义搜索和关键字搜索"
    )
    
    config.add_step(StepType.READER)
    config.add_step(StepType.HTML_PROCESSOR, extract_codes=True)
    config.add_step(StepType.CHUNKER)
    config.add_step(StepType.EMBEDDER)
    
    return config


# 预定义流水线注册表
PIPELINE_REGISTRY: Dict[str, Callable[[], PipelineConfig]] = {
    "milvus": create_milvus_pipeline,
    "elasticsearch": create_elasticsearch_pipeline,
    "hybrid": create_hybrid_pipeline,
}


def get_pipeline_config(backend: str) -> PipelineConfig:
    """
    根据存储后端获取流水线配置
    
    Args:
        backend: 存储后端名称
        
    Returns:
        PipelineConfig: 流水线配置
    """
    if backend not in PIPELINE_REGISTRY:
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Available: {list(PIPELINE_REGISTRY.keys())}"
        )
    return PIPELINE_REGISTRY[backend]()


def register_pipeline(name: str, factory: Callable[[], PipelineConfig]):
    """
    注册自定义流水线
    
    Args:
        name: 流水线名称
        factory: 创建流水线配置的工厂函数
    """
    PIPELINE_REGISTRY[name] = factory

