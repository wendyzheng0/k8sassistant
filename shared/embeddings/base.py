"""
Base Embedding abstraction
Defines the interface that all embedding providers must implement
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Union, Optional, Dict, Any
import logging


logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models"""
    model_name: str = "BAAI/bge-small-zh-v1.5"
    backend: str = "onnx"  # torch, onnx
    cache_dir: Optional[str] = None
    local_dir: Optional[str] = None
    batch_size: int = 128
    hf_mirror_url: str = "https://hf-mirror.com"
    extra_params: Dict[str, Any] = field(default_factory=dict)


class BaseEmbedding(ABC):
    """
    Abstract base class for embedding models
    
    All embedding implementations should implement this interface.
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize the embedding model
        
        Args:
            config: Embedding configuration. If None, will load from environment/settings.
        """
        self.config = config or self._load_default_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialized = False
    
    @abstractmethod
    def _load_default_config(self) -> EmbeddingConfig:
        """Load default configuration from environment or settings"""
        pass
    
    @abstractmethod
    def encode(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Encode text(s) to embedding vector(s)
        
        Args:
            texts: Single text or list of texts to encode
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
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
            show_progress: Whether to show progress bar
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors
        
        Returns:
            Integer dimension of embeddings
        """
        pass
    
    def get_text_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text (llama-index compatible interface)
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector
        """
        embeddings = self.encode(text)
        return embeddings[0] if embeddings else []
    
    def get_text_embedding_batch(
        self, 
        texts: List[str], 
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Get embeddings for a batch of texts (llama-index compatible interface)
        
        Args:
            texts: List of texts to encode
            show_progress: Whether to show progress bar
            
        Returns:
            List of embedding vectors
        """
        return self.encode_batch(texts, show_progress=show_progress)
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        import numpy as np
        
        embeddings = self.encode([text1, text2])
        vec1 = np.array(embeddings[0])
        vec2 = np.array(embeddings[1])
        
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return float(similarity)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "model_name": self.config.model_name,
            "backend": self.config.backend,
            "batch_size": self.config.batch_size,
            "dimension": self.get_embedding_dimension() if self._initialized else None
        }
    
    def __del__(self):
        """Clean up resources"""
        pass

