"""
Processor Configuration Management
统一的处理器配置管理，支持环境变量和代码配置
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Literal

# Add project root to path for shared module imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

# Load environment variables
env_path = PROJECT_ROOT / ".env"
try:
    if env_path.exists():
        load_dotenv(env_path)
except (PermissionError, OSError):
    # Ignore permission errors (e.g., in sandboxed environments)
    pass


@dataclass
class ProcessorSettings:
    """
    处理器配置类
    
    支持从环境变量读取配置，也可以通过代码直接设置
    """
    
    # 存储后端配置
    storage_backend: Literal["milvus", "elasticsearch"] = field(
        default_factory=lambda: os.getenv("STORAGE_BACKEND", "milvus")
    )
    
    # Milvus 配置
    milvus_uri: str = field(
        default_factory=lambda: os.getenv("MILVUS_URI", "http://localhost:19530")
    )
    collection_name: str = field(
        default_factory=lambda: os.getenv("COLLECTION_NAME", "k8s_docs")
    )
    milvus_overwrite: bool = field(
        default_factory=lambda: os.getenv("MILVUS_OVERWRITE", "true").lower() == "true"
    )
    
    # Elasticsearch 配置
    es_host: str = field(
        default_factory=lambda: os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
    )
    es_index: str = field(
        default_factory=lambda: os.getenv("ELASTICSEARCH_INDEX", "k8s-docs")
    )
    es_user: str = field(
        default_factory=lambda: os.getenv("ELASTICSEARCH_USER", "elastic")
    )
    es_password: str = field(
        default_factory=lambda: os.getenv("ELASTICSEARCH_PASSWORD", "password")
    )
    es_ca_certs: str = field(
        default_factory=lambda: os.getenv("ELASTICSEARCH_CA_CERTS", "")
    )
    es_num_shards: int = field(
        default_factory=lambda: int(os.getenv("ES_NUM_SHARDS", "1"))
    )
    es_num_replicas: int = field(
        default_factory=lambda: int(os.getenv("ES_NUM_REPLICAS", "0"))
    )
    
    # 文本处理配置
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1024"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "100"))
    )
    min_text_length: int = field(
        default_factory=lambda: int(os.getenv("MIN_TEXT_LENGTH", "10"))
    )
    
    # LLM 配置 (用于 TitleExtractor 等)
    enable_llm_extractors: bool = field(
        default_factory=lambda: os.getenv("ENABLE_LLM_EXTRACTORS", "false").lower() == "true"
    )
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", os.getenv("OPENAI_BASE_URL", ""))
    )
    llm_api_key: str = field(
        default_factory=lambda: os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "qwen3:14b")
    )
    llm_temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.7"))
    )
    keyword_count: int = field(
        default_factory=lambda: int(os.getenv("KEYWORD_COUNT", "3"))
    )
    
    # 文件处理配置
    data_dir: str = field(
        default_factory=lambda: os.getenv(
            "DATA_DIR", 
            str(PROJECT_ROOT / "data" / "zh-cn")
        )
    )
    required_extensions: List[str] = field(
        default_factory=lambda: [".html", ".md", ".txt"]
    )
    exclude_patterns: List[str] = field(
        default_factory=lambda: ["_print"]
    )
    
    # 缓存配置
    md_cache_dir: str = field(
        default_factory=lambda: os.getenv(
            "MD_CACHE_DIR",
            str(Path(__file__).parent / "md_cache")
        )
    )
    code_blocks_dir: str = field(
        default_factory=lambda: os.getenv(
            "CODE_BLOCKS_DIR",
            str(PROJECT_ROOT / "backend" / "codeblocks")
        )
    )
    
    # 批处理配置
    batch_size: int = field(
        default_factory=lambda: int(os.getenv("BATCH_SIZE", "32"))
    )
    es_batch_bytes: int = field(
        default_factory=lambda: int(os.getenv("ES_BATCH_BYTES", str(2 * 1024 * 1024)))
    )
    
    # 日志配置
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    
    def __post_init__(self):
        """验证配置"""
        if self.storage_backend not in ("milvus", "elasticsearch"):
            raise ValueError(f"Invalid storage_backend: {self.storage_backend}")
        
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive: {self.chunk_size}")
        
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative: {self.chunk_overlap}")
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "storage_backend": self.storage_backend,
            "milvus_uri": self.milvus_uri,
            "collection_name": self.collection_name,
            "es_host": self.es_host,
            "es_index": self.es_index,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "data_dir": self.data_dir,
            "batch_size": self.batch_size,
        }
    
    def print_config(self):
        """打印当前配置"""
        print("🔧 Processor Configuration:")
        print(f"   Storage Backend: {self.storage_backend}")
        if self.storage_backend == "milvus":
            print(f"   Milvus URI: {self.milvus_uri}")
            print(f"   Collection: {self.collection_name}")
        else:
            print(f"   ES Host: {self.es_host}")
            print(f"   ES Index: {self.es_index}")
        print(f"   Chunk Size: {self.chunk_size}")
        print(f"   Chunk Overlap: {self.chunk_overlap}")
        print(f"   Data Dir: {self.data_dir}")
        print(f"   Batch Size: {self.batch_size}")


# 全局配置实例
_settings: Optional[ProcessorSettings] = None


def get_processor_settings(**kwargs) -> ProcessorSettings:
    """
    获取处理器配置实例
    
    Args:
        **kwargs: 覆盖默认配置的参数
        
    Returns:
        ProcessorSettings: 配置实例
    """
    global _settings
    if _settings is None or kwargs:
        _settings = ProcessorSettings(**kwargs)
    return _settings


def reload_processor_settings(**kwargs) -> ProcessorSettings:
    """重新加载配置"""
    global _settings
    _settings = ProcessorSettings(**kwargs)
    return _settings

