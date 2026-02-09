"""
应用配置管理
Backend 专属配置，基础能力配置复用 shared.config
"""

import os
import sys
from pathlib import Path
from typing import List, TYPE_CHECKING, Union
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator

# 获取项目根目录
ROOT_DIR = Path(__file__).parent.parent.parent.parent

# 确保 shared 模块可以被导入
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from shared.config import get_settings as get_shared_settings, SharedSettings


class BackendSettings(BaseSettings):
    """
    Backend 专属配置类
    
    基础能力配置（LLM/Embedding/Milvus/ES 基础连接/RRF 等）统一由 shared.config 管理，
    本类只定义 backend 应用专属的配置项。
    
    访问基础配置请使用 settings.shared.XXX 或直接 from shared.config import get_settings
    """
    
    # ============ 应用基本信息 ============
    PROJECT_NAME: str = "K8s Assistant"
    APP_NAME: str = "K8s Assistant"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # ============ 服务器配置 ============
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ENVIRONMENT: str = "development"  # "development" | "production"

    # ============ CORS 配置 ============
    # 从环境变量读取允许的跨域源
    # .env 文件格式: CORS_ORIGINS=["http://localhost:3000"]
    # 或者用逗号分隔: CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["GET", "POST", "OPTIONS"]
    CORS_ALLOW_HEADERS: List[str] = [
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "Accept",
    ]
    
    # ============ 安全配置 ============
    # SECRET_KEY: 必须从环境变量设置，不要使用默认值
    # 生成方式: python -c "import secrets; print(secrets.token_urlsafe(32))"
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    API_V1_STR: str = "/api/v1"
    
    # ============ 数据库配置 ============
    # 请在下方设置您的数据库密码
    DATABASE_URL: str = "postgresql+asyncpg://k8suser:pass@localhost:5432/k8sassistant"
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # ============ 缓存配置 ============
    CACHE_TTL: int = 3600  # 缓存过期时间（秒）
    MAX_CACHED_CHATS: int = 5  # 每个用户缓存的最大聊天数量
    MAX_CACHED_MESSAGES: int = 20  # 每个聊天缓存的最大消息数量
    
    # ============ Elasticsearch 高级配置（Backend 专属） ============
    # 基础连接配置（HOST/INDEX/USER/PASSWORD/CA_CERTS）由 shared.config 管理
    # 以下为 backend 服务专属的网络与查询性能配置
    
    # request_timeout: 单次请求的总超时时间（秒）
    ELASTICSEARCH_REQUEST_TIMEOUT: float = 10.0
    # max_retries + retry_on_timeout: 在超时场景下自动重试
    ELASTICSEARCH_MAX_RETRIES: int = 2
    ELASTICSEARCH_RETRY_ON_TIMEOUT: bool = True
    # 服务端 search timeout（如 "8s"）。留空则不设置
    ELASTICSEARCH_SEARCH_TIMEOUT: str = ""
    # 是否开启高亮
    ELASTICSEARCH_ENABLE_HIGHLIGHT: bool = True
    # 当 top_k 超过该值时自动关闭高亮
    ELASTICSEARCH_HIGHLIGHT_MAX_TOP_K: int = 20
    # 是否启用模糊匹配
    ELASTICSEARCH_ENABLE_FUZZINESS: bool = True
    # 当 query token 数超过该值时自动关闭 fuzziness
    ELASTICSEARCH_FUZZINESS_MAX_TOKENS: int = 8
    # fuzziness 的扩展与前缀参数
    ELASTICSEARCH_FUZZINESS_MAX_EXPANSIONS: int = 50
    ELASTICSEARCH_FUZZINESS_PREFIX_LENGTH: int = 1
    # 不需要总命中数时关闭，可减少额外开销
    ELASTICSEARCH_TRACK_TOTAL_HITS: bool = False
    
    # ============ 数据处理配置 ============
    NLTK_DATA: str = str(ROOT_DIR / "data_processing" / "processors" / "nltk_data")
    DOCS_DIR: Path = ROOT_DIR / "docs"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    
    # ============ 文件上传配置 ============
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES: List[str] = [".txt", ".md", ".pdf", ".docx", ".html"]
    
    # ============ GraphRAG 配置 ============
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
    
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        case_sensitive=True,
        env_prefix="",
        env_file_encoding="utf-8",
        extra="ignore"  # 忽略 shared config 管理的字段
    )

    # ============ 验证器 ============
    @field_validator("SECRET_KEY")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """验证 SECRET_KEY 是否安全"""
        # 检查长度（至少32字节/256位）
        if len(v) < 32:
            raise ValueError(
                f"SECRET_KEY must be at least 32 characters long for security. "
                f"Current length: {len(v)}. "
                f"Generate a secure key with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
            )

        # 检查是否使用了常见的默认值
        insecure_defaults = [
            "your-secret-key-here",
            "your-super-secret-key-change-this-in-production",
            "secret",
            "password",
            "secret-key",
            "jwt-secret-key",
        ]

        v_lower = v.lower()
        for insecure in insecure_defaults:
            if insecure in v_lower:
                raise ValueError(
                    f"SECRET_KEY is using an insecure default value. "
                    f"Please generate a secure key with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
                )

        return v

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """解析 CORS_ORIGINS，支持字符串和列表格式"""
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            # 尝试解析 JSON 格式: ["http://a.com","http://b.com"]
            if v.startswith("[") and v.endswith("]"):
                import json
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            # 解析逗号分隔格式: http://a.com,http://b.com
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("CORS_ORIGINS")
    @classmethod
    def validate_cors_origins(cls, v: list[str], info) -> list[str]:
        """验证 CORS 配置的安全性"""
        # 获取环境配置
        environment = info.data.get("ENVIRONMENT", "development")
        allow_credentials = info.data.get("CORS_ALLOW_CREDENTIALS", True)

        # 生产环境检查
        if environment == "production":
            for origin in v:
                if "localhost" in origin or "127.0.0.1" in origin:
                    raise ValueError(
                        f"Production CORS_ORIGINS cannot contain localhost/127.0.0.1. Found: {origin}"
                    )
                # 生产环境推荐 HTTPS
                if origin.startswith("http://") and origin not in ["http://localhost", "http://127.0.0.1"]:
                    # 允许 localhost 的 HTTP，但警告生产环境应该用 HTTPS
                    pass

        # credentials 和通配符不能同时使用
        if allow_credentials and "*" in v:
            raise ValueError(
                'Cannot use "*" in CORS_ORIGINS when CORS_ALLOW_CREDENTIALS=True'
            )

        # 验证每个 origin 格式
        for origin in v:
            if origin != "*":
                if not (origin.startswith("http://") or origin.startswith("https://")):
                    raise ValueError(
                        f"Invalid CORS origin format: {origin}. Must start with http:// or https://"
                    )

        return v

    # ============ Shared Config 访问 ============
    @property
    def shared(self) -> SharedSettings:
        """获取 shared 配置实例"""
        return get_shared_settings()
    
    # ============ 向后兼容属性（代理到 shared） ============
    # 这些属性保持向后兼容，新代码应直接使用 settings.shared.XXX
    
    @property
    def LLM_API_KEY(self) -> str:
        return self.shared.LLM_API_KEY
    
    @property
    def LLM_BASE_URL(self) -> str:
        return self.shared.LLM_BASE_URL
    
    @property
    def LLM_MODEL(self) -> str:
        return self.shared.LLM_MODEL
    
    @property
    def MAX_TOKENS(self) -> int:
        return self.shared.MAX_TOKENS
    
    @property
    def TEMPERATURE(self) -> float:
        return self.shared.TEMPERATURE
    
    @property
    def EMBEDDING_MODEL(self) -> str:
        return self.shared.EMBEDDING_MODEL
    
    @property
    def EMBEDDING_DEVICE(self) -> str:
        return self.shared.EMBEDDING_DEVICE
    
    @property
    def EMBEDDING_BACKEND(self) -> str:
        return self.shared.EMBEDDING_BACKEND
    
    @property
    def EMBEDDING_LOCAL_DIR(self) -> str:
        return self.shared.EMBEDDING_LOCAL_DIR
    
    @property
    def EMBEDDING_CACHE_DIR(self) -> str:
        return self.shared.EMBEDDING_CACHE_DIR
    
    @property
    def HF_MIRROR_BASE_URL(self) -> str:
        return self.shared.HF_MIRROR_BASE_URL
    
    @property
    def HF_OFFLINE(self) -> bool:
        return self.shared.HF_OFFLINE
    
    @property
    def MILVUS_MODE(self) -> str:
        return self.shared.MILVUS_MODE
    
    @property
    def MILVUS_URI(self) -> str:
        return self.shared.MILVUS_URI
    
    @property
    def COLLECTION_NAME(self) -> str:
        return self.shared.COLLECTION_NAME
    
    @property
    def VECTOR_DIM(self) -> int:
        return self.shared.VECTOR_DIM
    
    @property
    def ELASTICSEARCH_HOST(self) -> str:
        return self.shared.ELASTICSEARCH_HOST
    
    @property
    def ELASTICSEARCH_INDEX(self) -> str:
        return self.shared.ELASTICSEARCH_INDEX
    
    @property
    def ELASTICSEARCH_USER(self) -> str:
        return self.shared.ELASTICSEARCH_USER
    
    @property
    def ELASTICSEARCH_PASSWORD(self) -> str:
        return self.shared.ELASTICSEARCH_PASSWORD
    
    @property
    def ELASTICSEARCH_CA_CERTS(self) -> str:
        return self.shared.ELASTICSEARCH_CA_CERTS
    
    @property
    def RRF_K(self) -> int:
        return self.shared.RRF_K


# 创建全局配置实例
settings = BackendSettings()


def validate_required_settings():
    """
    验证必要的配置

    注意：此函数应在应用启动时调用（如 FastAPI lifespan），
    而不是在模块导入时立即执行，以便 shared 模块可以被安全导入。
    """
    shared = get_shared_settings()

    # 验证 LLM_API_KEY
    if not shared.LLM_API_KEY:
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

    # 验证 SECRET_KEY
    secret_key = os.getenv("SECRET_KEY")
    if not secret_key:
        env_file_path = ROOT_DIR / ".env"
        error_msg = f"""
SECRET_KEY must be set in .env file or environment variable for security.

Current value: {os.getenv('SECRET_KEY', 'NOT_SET')}
.env file location: {env_file_path}

To generate a secure SECRET_KEY, run:
  python -c "import secrets; print(secrets.token_urlsafe(32))"

Then add it to your .env file:
  SECRET_KEY=<generated-key>
"""
        raise ValueError(error_msg)


def get_settings() -> BackendSettings:
    """获取配置实例"""
    return settings
