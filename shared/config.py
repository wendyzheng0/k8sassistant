"""
Shared configuration management
Provides unified configuration access for all modules
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class SharedSettings(BaseSettings):
    """Shared configuration settings"""
    
    # LLM Configuration
    LLM_PROVIDER: str = "openai"  # openai, anthropic, ollama
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://api.deepseek.com/v1"
    LLM_MODEL: str = "deepseek-chat"
    MAX_TOKENS: int = 4096
    TEMPERATURE: float = 0.7
    # LLM request timeout (seconds). Increase if prompts are large / model is slow.
    LLM_TIMEOUT: int = 180

    # RAG Generation prompt size controls
    # How many documents to include in the prompt for generation.
    RAG_GENERATION_MAX_DOCS: int = 10
    # Max characters to include from a single document chunk (after code block restore).
    RAG_GENERATION_MAX_CHARS_PER_DOC: int = 2000
    # Max total characters of the constructed context (best-effort cap).
    RAG_GENERATION_MAX_TOTAL_CHARS: int = 20000
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_BACKEND: str = "onnx"  # torch, onnx
    EMBEDDING_LOCAL_DIR: str = ""
    EMBEDDING_CACHE_DIR: str = str(PROJECT_ROOT / "hf_cache")
    
    # HuggingFace Configuration
    HF_MIRROR_BASE_URL: str = "https://hf-mirror.com"
    HF_OFFLINE: bool = False
    
    # Milvus Configuration
    MILVUS_MODE: str = "embedded"
    MILVUS_URI: str = "http://localhost:19530"
    COLLECTION_NAME: str = "k8s_docs"
    VECTOR_DIM: int = 512
    
    # Elasticsearch Configuration
    ELASTICSEARCH_HOST: str = "http://localhost:9200"
    ELASTICSEARCH_INDEX: str = "k8s-docs"
    ELASTICSEARCH_USER: str = "elastic"
    ELASTICSEARCH_PASSWORD: str = "password"
    ELASTICSEARCH_CA_CERTS: str = ""
    
    # RRF Reranking Configuration
    RRF_K: int = 60
    
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        case_sensitive=True,
        env_prefix="",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Global settings instance
_settings: Optional[SharedSettings] = None


def get_settings() -> SharedSettings:
    """Get the shared settings instance (singleton)"""
    global _settings
    if _settings is None:
        _settings = SharedSettings()
    return _settings


def reload_settings() -> SharedSettings:
    """Reload settings (useful for testing)"""
    global _settings
    _settings = SharedSettings()
    return _settings

