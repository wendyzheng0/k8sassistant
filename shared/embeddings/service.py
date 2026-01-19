"""
Unified Embedding Service
Provides a single interface for embedding operations across all modules
"""

from typing import List, Union, Optional, Dict, Any
import logging

from .base import BaseEmbedding, EmbeddingConfig
from .bge_onnx_embedding import BGEOnnxEmbedding


logger = logging.getLogger(__name__)


# Singleton instance
_embedding_service: Optional["EmbeddingService"] = None


class EmbeddingService:
    """
    Unified embedding service
    
    This service provides a consistent interface for embedding operations
    and can be shared between backend and data_processing modules.
    """
    
    def __init__(self, embedding_model: Optional[BaseEmbedding] = None):
        """
        Initialize the embedding service
        
        Args:
            embedding_model: Optional pre-configured embedding model.
                           If None, will create default BGEOnnxEmbedding.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self._model = embedding_model or self._create_default_model()
    
    def _create_default_model(self) -> BaseEmbedding:
        """Create default embedding model based on settings"""
        from shared.config import get_settings
        settings = get_settings()
        
        self.logger.info(f"🔄 Creating embedding model: {settings.EMBEDDING_MODEL}")
        
        config = EmbeddingConfig(
            model_name=settings.EMBEDDING_MODEL,
            backend=settings.EMBEDDING_BACKEND,
            cache_dir=settings.EMBEDDING_CACHE_DIR,
            local_dir=settings.EMBEDDING_LOCAL_DIR,
            hf_mirror_url=settings.HF_MIRROR_BASE_URL
        )
        
        # Currently only BGE ONNX is implemented
        return BGEOnnxEmbedding(config)
    
    @property
    def model(self) -> BaseEmbedding:
        """Get the underlying embedding model"""
        return self._model
    
    def encode(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Encode text(s) to embedding vector(s)
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            List of embedding vectors
        """
        return self._model.encode(texts)
    
    def encode_batch(
        self, 
        texts: List[str], 
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Encode texts in batches
        
        Args:
            texts: List of texts to encode
            batch_size: Override default batch size
            show_progress: Whether to show progress
            
        Returns:
            List of embedding vectors
        """
        return self._model.encode_batch(texts, batch_size, show_progress)
    
    def get_text_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text (llama-index compatible)
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector
        """
        return self._model.get_text_embedding(text)
    
    def get_text_embedding_batch(
        self, 
        texts: List[str], 
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Get embeddings for a batch of texts (llama-index compatible)
        
        Args:
            texts: List of texts
            show_progress: Whether to show progress
            
        Returns:
            List of embedding vectors
        """
        return self._model.get_text_embedding_batch(texts, show_progress)
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension"""
        return self._model.get_embedding_dimension()
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        return self._model.similarity(text1, text2)
    
    def find_most_similar(
        self, 
        query: str, 
        candidates: List[str], 
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar texts from candidates
        
        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of top results to return
            
        Returns:
            List of (text, score) tuples
        """
        import numpy as np
        
        if not candidates:
            return []
        
        # Encode all texts
        all_texts = [query] + candidates
        embeddings = self.encode(all_texts)
        
        query_embedding = embeddings[0]
        candidate_embeddings = embeddings[1:]
        
        # Calculate similarities
        similarities = []
        for i, candidate_emb in enumerate(candidate_embeddings):
            sim = np.dot(query_embedding, candidate_emb)
            similarities.append((candidates[i], float(sim)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model"""
        return self._model.get_model_info()


def create_embedding_service(
    config: Optional[EmbeddingConfig] = None,
    use_singleton: bool = True
) -> EmbeddingService:
    """
    Create or get an embedding service instance
    
    Args:
        config: Optional custom configuration
        use_singleton: If True, returns cached instance
        
    Returns:
        EmbeddingService instance
    """
    global _embedding_service
    
    if use_singleton and _embedding_service is not None and config is None:
        return _embedding_service

    if config:
        model = BGEOnnxEmbedding(config)
        service = EmbeddingService(model)
    else:
        service = EmbeddingService()
    
    if use_singleton and config is None:
        _embedding_service = service
    
    return service


def clear_embedding_service_cache() -> None:
    """Clear the cached embedding service instance"""
    global _embedding_service
    _embedding_service = None
    logger.info("✅ Cleared embedding service cache")

