"""
BGE ONNX Embedding implementation
Provides high-performance embedding using ONNX Runtime with optional GPU acceleration
"""

import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from typing import List, Union, Optional
import logging
import torch
import psutil

from .base import BaseEmbedding, EmbeddingConfig


logger = logging.getLogger(__name__)


def check_gpu_availability() -> bool:
    """
    Check if GPU is available for ONNX Runtime
    
    Returns:
        bool: True if GPU is available, False otherwise
    """
    try:
        available_providers = ort.get_available_providers()
        gpu_available = "CUDAExecutionProvider" in available_providers
        print(f"🔍 Available ONNX providers: {available_providers}")
        print(f"🚀 GPU (CUDA) available: {gpu_available}")
        return gpu_available
    except Exception as e:
        print(f"❌ Error checking GPU availability: {e}")
        return False


class BGEOnnxEmbedding(BaseEmbedding):
    """
    BGE ONNX embedding model
    
    This implementation provides the same interface as HuggingFaceEmbedding but uses
    ONNX Runtime with GPU acceleration for better performance.
    
    Batch size recommendations based on GPU memory:
    - 2GB GPU: batch_size = 32
    - 4GB GPU: batch_size = 64
    - 8GB+ GPU: batch_size = 128
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        super().__init__(config)
        self.session: Optional[ort.InferenceSession] = None
        self.tokenizer = None
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self.embedding_dim: int = 0
        self._initialize_model()
    
    def _load_default_config(self) -> EmbeddingConfig:
        """Load configuration from shared settings"""
        from shared.config import get_settings
        settings = get_settings()
        
        return EmbeddingConfig(
            model_name=settings.EMBEDDING_MODEL,
            backend=settings.EMBEDDING_BACKEND,
            cache_dir=settings.EMBEDDING_CACHE_DIR,
            local_dir=settings.EMBEDDING_LOCAL_DIR,
            hf_mirror_url=settings.HF_MIRROR_BASE_URL
        )
    
    def _get_optimal_gpu_memory_limit(self) -> int:
        """Get optimal GPU memory limit based on actual GPU memory"""
        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_gb = gpu_memory / (1024**3)
                
                print(f"🔍 Detected GPU memory: {gpu_memory_gb:.2f} GB")
                
                # Use 80% of available GPU memory
                optimal_limit = int(gpu_memory * 0.8)
                optimal_limit_gb = optimal_limit / (1024**3)
                
                print(f"🎯 Setting GPU memory limit to: {optimal_limit_gb:.2f} GB (80% of total)")
                return optimal_limit
            else:
                print("⚠️ CUDA not available, using default 2GB limit")
                return 2 * 1024 * 1024 * 1024
        except Exception as e:
            print(f"⚠️ Could not detect GPU memory: {e}, using default 2GB limit")
            return 2 * 1024 * 1024 * 1024

    def _adjust_batch_size_for_memory(self):
        """Dynamically adjust batch size based on available memory"""
        try:
            available_memory = psutil.virtual_memory().available
            gpu_memory = 0
            
            # Try to get GPU memory info if available
            try:
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
            except:
                pass
            
            # Adjust batch size based on available memory
            if available_memory < 2 * 1024 * 1024 * 1024:  # Less than 2GB
                self.config.batch_size = min(16, self.config.batch_size)
                print(f"⚠️ Low memory detected, reducing batch size to {self.config.batch_size}")
            elif available_memory < 4 * 1024 * 1024 * 1024:  # Less than 4GB
                self.config.batch_size = min(32, self.config.batch_size)
                print(f"⚠️ Moderate memory detected, reducing batch size to {self.config.batch_size}")
            
            # Adjust batch size based on GPU memory
            if gpu_memory > 0:
                gpu_memory_gb = gpu_memory / (1024**3)
                if gpu_memory_gb >= 20:  # 20GB+ GPU
                    self.config.batch_size = max(128, self.config.batch_size)
                    print(f"🚀 Large GPU detected ({gpu_memory_gb:.1f}GB), using batch size: {self.config.batch_size}")
                elif gpu_memory_gb >= 8:  # 8GB+ GPU
                    self.config.batch_size = max(64, self.config.batch_size)
                    print(f"✅ Good GPU detected ({gpu_memory_gb:.1f}GB), using batch size: {self.config.batch_size}")
                elif gpu_memory_gb >= 4:  # 4GB+ GPU
                    self.config.batch_size = min(32, self.config.batch_size)
                    print(f"⚠️ Moderate GPU detected ({gpu_memory_gb:.1f}GB), using batch size: {self.config.batch_size}")
                else:  # Less than 4GB GPU
                    self.config.batch_size = min(16, self.config.batch_size)
                    print(f"⚠️ Small GPU detected ({gpu_memory_gb:.1f}GB), using batch size: {self.config.batch_size}")
                
        except Exception as e:
            print(f"⚠️ Could not adjust batch size: {e}")

    def _resolve_model_path(self) -> str:
        """Resolve the actual model path from config"""
        # Check for local model directory first
        if self.config.local_dir and os.path.isdir(self.config.local_dir):
            print(f"Using local model: {self.config.local_dir}")
            return self.config.local_dir
        
        # Set up HuggingFace mirror if configured
        if self.config.hf_mirror_url:
            os.environ["HF_ENDPOINT"] = self.config.hf_mirror_url
            os.environ["HUGGINGFACE_HUB_BASE_URL"] = self.config.hf_mirror_url
        
        if self.config.cache_dir:
            os.environ["TRANSFORMERS_CACHE"] = self.config.cache_dir
            os.environ["HF_HOME"] = self.config.cache_dir
            
            if not os.path.exists(self.config.cache_dir):
                os.makedirs(self.config.cache_dir, exist_ok=True)
        
        # Download from HuggingFace
        from huggingface_hub import snapshot_download
        
        print(f"Downloading {self.config.model_name} to {self.config.cache_dir}")
        model_path = snapshot_download(
            self.config.model_name,
            endpoint=self.config.hf_mirror_url,
            cache_dir=self.config.cache_dir
        )
        print(f"Model downloaded to {model_path}")
        return model_path

    def _initialize_model(self):
        """Initialize the ONNX model and tokenizer"""
        try:
            print(f"🔄 Loading BGE ONNX model: {self.config.model_name}")
            
            # Adjust batch size based on available memory
            self._adjust_batch_size_for_memory()
            
            # Resolve model path
            model_path = self._resolve_model_path()
            
            # Check if ONNX model exists
            onnx_path = os.path.join(model_path, "onnx", "model.onnx")
            if not os.path.exists(onnx_path):
                raise FileNotFoundError(f"ONNX model not found at: {onnx_path}")
            
            # Set up providers based on device
            gpu_available = False
            try:
                available_providers = ort.get_available_providers()
                print(f"🔍 Available ONNX providers: {available_providers}")
                
                if "CUDAExecutionProvider" in available_providers:
                    gpu_available = True
                    print("🚀 CUDA provider is available")
                else:
                    print("⚠️ CUDA provider not available, falling back to CPU")
            except Exception as e:
                print(f"⚠️ Error checking CUDA availability: {e}")
                print("🔄 Falling back to CPU execution")
            
            # Configure providers
            if gpu_available:
                gpu_memory_limit = self._get_optimal_gpu_memory_limit()
                providers = [
                    ("CUDAExecutionProvider", {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "gpu_mem_limit": gpu_memory_limit,
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                        "enable_cuda_graph": False,
                    }),
                    "CPUExecutionProvider"
                ]
                print("🚀 Using GPU acceleration with CUDA")
            else:
                providers = ["CPUExecutionProvider"]
                print("💻 Using CPU execution")
            
            # Create ONNX session
            session_options = ort.SessionOptions()
            
            if gpu_available:
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
                session_options.enable_mem_pattern = True
                session_options.enable_cpu_mem_arena = True
            else:
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                session_options.enable_mem_pattern = False
                session_options.enable_cpu_mem_arena = False
            
            try:
                self.session = ort.InferenceSession(
                    onnx_path,
                    sess_options=session_options,
                    providers=providers
                )
                
                actual_providers = self.session.get_providers()
                print(f"📋 Actual providers used: {actual_providers}")
                
            except Exception as e:
                print(f"❌ Failed to create ONNX session: {e}")
                # Fallback to CPU only
                providers = ["CPUExecutionProvider"]
                self.session = ort.InferenceSession(
                    onnx_path,
                    sess_options=session_options,
                    providers=providers
                )
                print("✅ Created CPU-only ONNX session")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                cache_dir=self.config.cache_dir
            )
            
            # Get model info
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]
            
            print(f"✅ BGE ONNX model loaded successfully")
            print(f"📋 Inputs: {self.input_names}")
            print(f"📋 Outputs: {self.output_names}")
            
            # Test and get embedding dimension
            test_embedding = self._encode_single("test")
            self.embedding_dim = len(test_embedding)
            print(f"📏 Embedding dimension: {self.embedding_dim}")
            
            self._initialized = True
            
        except Exception as e:
            print(f"❌ Failed to initialize BGE ONNX model: {e}")
            raise
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings (required for BGE models)"""
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    def _encode_single(self, text: str) -> List[float]:
        """Encode a single text"""
        return self._encode_texts([text])[0]
    
    def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        """Encode a list of texts"""
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np"
        )
        
        # Prepare ONNX inputs
        onnx_inputs = {}
        for input_name in self.input_names:
            if input_name in inputs:
                onnx_inputs[input_name] = inputs[input_name].astype(np.int64)
        
        # Run inference
        outputs = self.session.run(self.output_names, onnx_inputs)
        embeddings = outputs[0]
        
        # Handle different output shapes
        if len(embeddings.shape) == 3:
            # Shape: (batch_size, sequence_length, hidden_size)
            embeddings = embeddings[:, 0, :]  # Take [CLS] token
        
        # Normalize
        normalized = self._normalize_embeddings(embeddings)
        return normalized.tolist()
    
    def encode(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Encode text(s) to embedding vector(s)"""
        if isinstance(texts, str):
            texts = [texts]
        
        return self.encode_batch(texts)
    
    def encode_batch(
        self, 
        texts: List[str], 
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> List[List[float]]:
        """Encode texts in batches"""
        if not texts:
            return []
        
        bs = batch_size or self.config.batch_size
        all_embeddings = []
        
        for i in range(0, len(texts), bs):
            batch_texts = texts[i:i + bs]
            batch_embeddings = self._encode_texts(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
            if show_progress:
                print(f"📦 Processed batch {i//bs + 1}/{(len(texts) + bs - 1)//bs}")
        
        return all_embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors"""
        return self.embedding_dim
    
    # llama-index compatibility methods
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a query (required by llama-index BaseEmbedding)"""
        return self._encode_single(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text (required by llama-index BaseEmbedding)"""
        return self._encode_single(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts (required by llama-index BaseEmbedding)"""
        return self.encode_batch(texts)


def create_bge_onnx_embedding(
    model_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
    batch_size: int = 128
) -> BGEOnnxEmbedding:
    """
    Factory function to create BGE ONNX embedding model
    
    Args:
        model_name: Model name or path
        cache_dir: Optional cache directory
        batch_size: Batch size for processing
        
    Returns:
        BGEOnnxEmbedding instance
    """
    config = EmbeddingConfig(
        model_name=model_name or "BAAI/bge-small-zh-v1.5",
        cache_dir=cache_dir,
        batch_size=batch_size
    )
    return BGEOnnxEmbedding(config)

