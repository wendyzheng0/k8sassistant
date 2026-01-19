"""
文本嵌入服务
Wraps the shared embedding service for backward compatibility
"""

from typing import List, Union
import logging

# Import from shared module
from shared.embeddings import EmbeddingService as SharedEmbeddingService, create_embedding_service


logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    文本嵌入服务类
    
    This class wraps the shared embedding service for backward compatibility
    with existing code that uses EmbeddingService directly.
    """
    
    def __init__(self):
        """Initialize the embedding service"""
        self.logger = logging.getLogger("EmbeddingService")
        
        # Use the shared singleton embedding service
        self._service = create_embedding_service(use_singleton=True)
        
        self.logger.info("✅ Embedding service initialized (using shared service)")
    
    @property
    def model(self):
        """Get the underlying embedding model"""
        return self._service.model
    
    def encode(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Encode text(s) to embedding vector(s)
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            List of embedding vectors
        """
        return self._service.encode(texts)
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Batch encode texts
        
        Args:
            texts: List of texts
            batch_size: Batch size
            
        Returns:
            List of embedding vectors
        """
        return self._service.encode_batch(texts, batch_size)
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self._service.get_embedding_dimension()
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        return self._service.similarity(text1, text2)
    
    def find_most_similar(self, query: str, candidates: List[str], top_k: int = 5) -> List[tuple]:
        """
        Find most similar texts
        
        Args:
            query: Query text
            candidates: Candidate texts
            top_k: Number of top results
            
        Returns:
            List of (text, score) tuples
        """
        return self._service.find_most_similar(query, candidates, top_k)
    
    def __del__(self):
        """Clean up resources"""
        pass
