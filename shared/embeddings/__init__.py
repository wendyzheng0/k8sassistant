"""
Shared Embedding service
Provides unified embedding capabilities for both backend and data_processing
"""

from .base import BaseEmbedding, EmbeddingConfig
from .service import EmbeddingService, create_embedding_service

__all__ = [
    "BaseEmbedding",
    "EmbeddingConfig",
    "EmbeddingService",
    "create_embedding_service",
]

