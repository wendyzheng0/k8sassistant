"""
应用配置管理
"""

import os
import sys
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

# 获取项目根目录
ROOT_DIR = Path(__file__).parent.parent.parent.parent


class Settings(BaseSettings):
    """应用配置类"""
    
    # 应用基本信息
    APP_NAME: str = "K8s Assistant"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # 安全配置
    SECRET_KEY: str = "your-secret-key-here"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Milvus 配置
    MILVUS_MODE: str = "embedded"  # "standalone" 或 "embedded"
    MILVUS_URI: str = "http://localhost:19530"
    COLLECTION_NAME: str = "k8s_docs"
    VECTOR_DIM: int = 384  # 支持512维的BGE模型,目前发现BAAI/bge-small-en-v1.5是384的,BAAI/bge-small-zh-v1.5是512的
    
    # Elasticsearch 配置
    ELASTICSEARCH_HOST: str = "http://localhost:9200"  # 默认使用HTTP，避免SSL证书问题
    ELASTICSEARCH_INDEX: str = "k8s-docs"
    ELASTICSEARCH_USER: str = "elastic"
    ELASTICSEARCH_PASSWORD: str = "password"
    ELASTICSEARCH_CA_CERTS: str = ""  # 留空，让代码自动检测证书路径
    # Elasticsearch 网络与重试配置
    # request_timeout: 单次请求的总超时时间（秒）。如果 ES 压力大/网络抖动，适当调大。
    ELASTICSEARCH_REQUEST_TIMEOUT: float = 10.0
    # max_retries + retry_on_timeout: 在超时场景下自动重试，提升偶发网络抖动的可用性
    ELASTICSEARCH_MAX_RETRIES: int = 2
    ELASTICSEARCH_RETRY_ON_TIMEOUT: bool = True
    # Elasticsearch 查询性能相关配置
    # 服务端 search timeout（如 "8s"）。留空则不设置，让 ES 自己跑完；建议配合较大的 request_timeout 使用。
    ELASTICSEARCH_SEARCH_TIMEOUT: str = ""
    # 是否开启高亮（高亮通常比较耗时，尤其 top_k 大时）
    ELASTICSEARCH_ENABLE_HIGHLIGHT: bool = True
    # 当 top_k 超过该值时自动关闭高亮（避免 50 条高亮导致响应过慢）
    ELASTICSEARCH_HIGHLIGHT_MAX_TOP_K: int = 20
    # 是否启用模糊匹配（fuzziness 会明显增加查询开销）
    ELASTICSEARCH_ENABLE_FUZZINESS: bool = True
    # 当 query token 数超过该值时自动关闭 fuzziness（长 query 用 fuzziness 代价太高）
    ELASTICSEARCH_FUZZINESS_MAX_TOKENS: int = 8
    # fuzziness 的扩展与前缀参数（用于控制开销）
    ELASTICSEARCH_FUZZINESS_MAX_EXPANSIONS: int = 50
    ELASTICSEARCH_FUZZINESS_PREFIX_LENGTH: int = 1
    # 不需要总命中数时关闭，可减少额外开销
    ELASTICSEARCH_TRACK_TOTAL_HITS: bool = False
    
    # LLM 配置
    LLM_API_KEY: str = ""  # 必需配置
    LLM_BASE_URL: str = "https://api.deepseek.com/v1"
    LLM_MODEL: str = "deepseek-chat"
    MAX_TOKENS: int = 4096
    TEMPERATURE: float = 0.7
    
    # 嵌入模型配置
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_BACKEND: str = "torch"
    # 若提供本地模型目录，则优先从本地加载（离线）
    EMBEDDING_LOCAL_DIR: str = ""
    # 可选：指定模型缓存目录（传给 SentenceTransformer 的 cache_folder）
    EMBEDDING_CACHE_DIR: str = "hf_cache"
    # 可选：Hugging Face 镜像与离线设置
    HF_MIRROR_BASE_URL: str = ""  # 例如: https://hf-mirror.com
    HF_OFFLINE: bool = False
    
    # For dataloader in pipeline - 使用项目目录下的 nltk_data
    NLTK_DATA: str = str(ROOT_DIR / "data_processing" / "processors" / "nltk_data")

    # 数据处理配置
    DOCS_DIR: Path = ROOT_DIR / "docs"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    
    # 文件上传配置
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES: List[str] = [".txt", ".md", ".pdf", ".docx", ".html"]
    
    # GraphRAG 配置
    GRAPH_STORAGE_TYPE: str = "networkx"  # "networkx" 或 "neo4j"
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    
    # 图构建配置
    GRAPH_CONFIDENCE_THRESHOLD: float = 0.5
    GRAPH_BATCH_SIZE: int = 100
    MAX_ENTITIES_PER_DOC: int = 50
    MAX_RELATIONSHIPS_PER_DOC: int = 100
    
    # 混合检索配置
    DEFAULT_VECTOR_WEIGHT: float = 0.6
    DEFAULT_GRAPH_WEIGHT: float = 0.4
    MAX_GRAPH_DEPTH: int = 3
    ENTITY_BOOST_FACTOR: float = 1.2
    
    # RRF重排序配置
    RRF_K: int = 60  # RRF算法中的常数，默认60，可尝试40或80
    
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        case_sensitive=True,
        env_prefix="",
        env_file_encoding="utf-8"
    )


# 创建全局配置实例
settings = Settings()

# 验证必要的配置
if not settings.LLM_API_KEY:
    import os
    env_file_path = ROOT_DIR / ".env"
    error_msg = f"""
LLM_API_KEY must be set. Please check:

1. Environment variable: LLM_API_KEY={os.getenv('LLM_API_KEY', 'NOT_SET')}
2. .env file exists: {env_file_path.exists()}
3. .env file location: {env_file_path}

To fix this, either:
- Set the environment variable: export LLM_API_KEY=your-api-key
- Create a .env file in the project root with: LLM_API_KEY=your-api-key
"""
    raise ValueError(error_msg)


def get_settings() -> Settings:
    """获取配置实例"""
    return settings
