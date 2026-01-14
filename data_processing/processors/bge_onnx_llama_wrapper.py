"""
BGE ONNX Embedding wrapper for llama-index compatibility

This module provides a wrapper that makes BGEOpenXEmbedding compatible with
llama-index's embedding interface, allowing it to replace HuggingFaceEmbedding
in dataloader.py and other llama-index components.
"""

import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from typing import List, Union, Optional
import logging
import torch
import psutil

# Import llama-index base classes
try:
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.bridge.pydantic import Field
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    # Fallback for older versions or if llama-index is not available
    LLAMA_INDEX_AVAILABLE = False
    BaseEmbedding = object
    Field = lambda default=None, **kwargs: default

logger = logging.getLogger(__name__)


def check_gpu_availability():
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


class BGEOpenXEmbedding(BaseEmbedding):
    """
    BGE ONNX embedding model compatible with llama-index
    
    This class provides the same interface as HuggingFaceEmbedding but uses
    ONNX Runtime with GPU acceleration for better performance.

    adjust batch_size according to GPU size
    batch_size = 32  # 2GB GPU
    batch_size = 64  # 4GB GPU
    batch_size = 128 # 8GB+ GPU

    GPU size limitation
    # 在包装器中调整 GPU 内存限制
    gpu_mem_limit = 24 * 1024 * 1024 * 1024  # 24GB
    """
    
    model_path: str = Field(description="Path to the BGE model directory")
    device: str = Field(default="gpu", description="Device to use (gpu or cpu)")
    cache_dir: Optional[str] = Field(default=None, description="Cache directory for tokenizer")
    batch_size: int = Field(default=128, description="Batch size for processing")
    
    # Runtime attributes (not validated by Pydantic)
    session: Optional[object] = Field(default=None, exclude=True)
    tokenizer: Optional[object] = Field(default=None, exclude=True)
    input_names: List[str] = Field(default_factory=list, exclude=True)
    output_names: List[str] = Field(default_factory=list, exclude=True)
    embedding_dim: int = Field(default=0, exclude=True)
    
    def __init__(
        self,
        model_path: str,
        device: str = "gpu",
        cache_dir: Optional[str] = None,
        batch_size: int = 128,
        **kwargs
    ):
        """
        Initialize BGE ONNX embedding model
        
        Args:
            model_path: Path to the BGE model directory
            device: Device to use ("gpu" or "cpu")
            cache_dir: Optional cache directory for tokenizer
            batch_size: Batch size for processing
        """
        # Initialize base class
        if LLAMA_INDEX_AVAILABLE:
            super().__init__(
                model_path=model_path,
                device=device,
                cache_dir=cache_dir,
                batch_size=batch_size,
                **kwargs
            )
        else:
            # Fallback initialization
            self.model_path = model_path
            self.device = device
            self.cache_dir = cache_dir
            self.batch_size = batch_size
        
        # Initialize runtime attributes
        self.session = None
        self.tokenizer = None
        self.input_names = []
        self.output_names = []
        self.embedding_dim = 0
        
        self._initialize_model()
    
    def _get_optimal_gpu_memory_limit(self):
        """Get optimal GPU memory limit based on actual GPU memory"""
        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_gb = gpu_memory / (1024**3)
                
                print(f"🔍 Detected GPU memory: {gpu_memory_gb:.2f} GB")
                
                # Use 80% of available GPU memory to leave room for other processes
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
                self.batch_size = min(16, self.batch_size)
                print(f"⚠️ Low memory detected, reducing batch size to {self.batch_size}")
            elif available_memory < 4 * 1024 * 1024 * 1024:  # Less than 4GB
                self.batch_size = min(32, self.batch_size)
                print(f"⚠️ Moderate memory detected, reducing batch size to {self.batch_size}")
            
            # Adjust batch size based on GPU memory
            if gpu_memory > 0:
                gpu_memory_gb = gpu_memory / (1024**3)
                if gpu_memory_gb >= 20:  # 20GB+ GPU
                    self.batch_size = max(128, self.batch_size)  # Use larger batch size
                    print(f"🚀 Large GPU detected ({gpu_memory_gb:.1f}GB), using batch size: {self.batch_size}")
                elif gpu_memory_gb >= 8:  # 8GB+ GPU
                    self.batch_size = max(64, self.batch_size)
                    print(f"✅ Good GPU detected ({gpu_memory_gb:.1f}GB), using batch size: {self.batch_size}")
                elif gpu_memory_gb >= 4:  # 4GB+ GPU
                    self.batch_size = min(32, self.batch_size)
                    print(f"⚠️ Moderate GPU detected ({gpu_memory_gb:.1f}GB), using batch size: {self.batch_size}")
                else:  # Less than 4GB GPU
                    self.batch_size = min(16, self.batch_size)
                    print(f"⚠️ Small GPU detected ({gpu_memory_gb:.1f}GB), using batch size: {self.batch_size}")
                
        except Exception as e:
            print(f"⚠️ Could not adjust batch size: {e}")

    def _initialize_model(self):
        """Initialize the ONNX model and tokenizer"""
        try:
            print(f"🔄 Loading BGE ONNX model from: {self.model_path}")
            
            # Adjust batch size based on available memory
            self._adjust_batch_size_for_memory()
            
            # Check if ONNX model exists
            onnx_path = os.path.join(self.model_path, "onnx", "model.onnx")
            if not os.path.exists(onnx_path):
                raise FileNotFoundError(f"ONNX model not found at: {onnx_path}")
            
            # Set up providers based on device
            gpu_available = False
            if self.device.lower() in ["gpu", "cuda"]:
                # Check if CUDA is available
                try:
                    available_providers = ort.get_available_providers()
                    print(f"🔍 Available ONNX providers: {available_providers}")
                    
                    if "CUDAExecutionProvider" in available_providers:
                        gpu_available = True
                        print("🚀 CUDA provider is available")
                    else:
                        print("⚠️ CUDA provider not available")
                        print("💡 This usually means:")
                        print("   - NVIDIA GPU drivers are not installed")
                        print("   - CUDA toolkit is not installed")
                        print("   - ONNX Runtime was not built with CUDA support")
                        print("🔄 Falling back to CPU execution")
                except Exception as e:
                    print(f"⚠️ Error checking CUDA availability: {e}")
                    print("🔄 Falling back to CPU execution")
            
            if gpu_available:
                # Get actual GPU memory to set appropriate limit
                gpu_memory_limit = self._get_optimal_gpu_memory_limit()
                
                providers = [
                    ("CUDAExecutionProvider", {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",  # Better for large memory allocations
                        "gpu_mem_limit": gpu_memory_limit,  # Use optimal memory limit
                        "cudnn_conv_algo_search": "EXHAUSTIVE",  # Better performance for large GPUs
                        "do_copy_in_default_stream": True,
                        "enable_cuda_graph": False,  # Since this is embedding model, the input sequence length is not fixed, so enable_cuda_graph is not needed
                    }),
                    "CPUExecutionProvider"
                ]
                print("🚀 Using GPU acceleration with CUDA")
            else:
                providers = ["CPUExecutionProvider"]
                print("💻 Using CPU execution")
                print("💡 For better performance, consider:")
                print("   - Installing NVIDIA GPU drivers")
                print("   - Installing CUDA toolkit")
                print("   - Using ONNX Runtime with CUDA support")
            
            # Create ONNX session with optimized settings
            session_options = ort.SessionOptions()
            # session_options.log_severity_level = 1
            
            if gpu_available:
                # Conservative GPU settings to avoid compatibility issues
                # session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                # session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
                session_options.enable_mem_pattern = True
                session_options.enable_cpu_mem_arena = True
                print("🚀 Using conservative GPU session settings")
            else:
                # CPU-optimized settings to prevent memory issues
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                session_options.enable_mem_pattern = False  # Disable to reduce memory usage
                session_options.enable_cpu_mem_arena = False  # Disable to reduce memory usage
                print("💻 Using CPU-optimized session settings to prevent memory issues")
            
            try:
                self.session = ort.InferenceSession(
                    onnx_path,
                    sess_options=session_options,
                    providers=providers
                )
                
                # Verify which provider is actually being used
                actual_providers = self.session.get_providers()
                print(f"📋 Actual providers used: {actual_providers}")
                
                if "CUDAExecutionProvider" in actual_providers:
                    print("✅ Successfully using GPU acceleration")
                else:
                    print("⚠️ Using CPU execution (GPU not available or failed to initialize)")
                    
            except Exception as e:
                print(f"❌ Failed to create ONNX session with providers {providers}: {e}")
                if gpu_available and "CUDAExecutionProvider" in [p[0] if isinstance(p, tuple) else p for p in providers]:
                    print("🔄 CUDA initialization failed, trying fallback options...")
                    
                    # Try with simplified CUDA settings first
                    try:
                        print("🔄 Trying simplified CUDA settings...")
                        simplified_cuda_providers = [
                            ("CUDAExecutionProvider", {
                                "device_id": 0,
                            }),
                            "CPUExecutionProvider"
                        ]
                        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                        self.session = ort.InferenceSession(
                            onnx_path,
                            sess_options=session_options,
                            providers=simplified_cuda_providers
                        )
                        print("✅ Successfully created ONNX session with simplified CUDA settings")
                    except Exception as cuda_e:
                        print(f"⚠️ Simplified CUDA settings also failed: {cuda_e}")
                        print("🔄 Falling back to CPU-only execution...")
                        
                        # Update session options for CPU-only execution
                        session_options.enable_cpu_mem_arena = True  # Re-enable for CPU
                        providers = ["CPUExecutionProvider"]
                        try:
                            self.session = ort.InferenceSession(
                                onnx_path,
                                sess_options=session_options,
                                providers=providers
                            )
                            print("✅ Successfully created CPU-only ONNX session")
                        except Exception as cpu_e:
                            print(f"❌ Failed to create CPU-only session: {cpu_e}")
                            raise cpu_e
                else:
                    raise e
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                cache_dir=self.cache_dir
            )
            
            # Get model info
            self.input_names = [input.name for input in self.session.get_inputs()]
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            print(f"✅ BGE ONNX model loaded successfully")
            print(f"📋 Inputs: {self.input_names}")
            print(f"📋 Outputs: {self.output_names}")
            
            # Test the model and get embedding dimension
            test_embedding = self._get_text_embedding_single("test")
            self.embedding_dim = len(test_embedding)
            print(f"📏 Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            print(f"❌ Failed to initialize BGE ONNX model: {e}")
            raise
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings (required for BGE models)"""
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    def _get_text_embedding_single(self, text: str) -> List[float]:
        """
        Get embedding for a single text (internal method)
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as list of floats
        """
        embeddings = self._get_text_embedding_batch([text])
        return embeddings[0]
    
    def _get_text_embedding_batch(self, texts: List[str], show_progress: bool = False) -> List[List[float]]:
        """
        Get embeddings for a batch of texts (llama-index interface)
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self._encode_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def _encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode a batch of texts"""
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
        embeddings = outputs[0]  # First output
        
        # Handle different output shapes
        if len(embeddings.shape) == 3:
            # Shape: (batch_size, sequence_length, hidden_size)
            # Extract [CLS] token embedding (first token) for each sequence
            embeddings = embeddings[:, 0, :]  # Take first token ([CLS]) for each sequence
        elif len(embeddings.shape) == 2:
            # Shape: (batch_size, hidden_size) - already sentence embeddings
            pass  # Use as-is
        else:
            raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")
        
        # Normalize
        normalized_embeddings = self._normalize_embeddings(embeddings)
        
        return normalized_embeddings.tolist()
    
    # llama-index compatibility methods
    def get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text (llama-index interface)"""
        return self._get_text_embedding_single(text)
    
    def get_text_embedding_batch(self, texts: List[str], show_progress: bool = False) -> List[List[float]]:
        """Get embeddings for a batch of texts (llama-index interface)"""
        return self._get_text_embedding_batch(texts)
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts"""
        embeddings = self._get_text_embedding_batch([text1, text2])
        vec1 = np.array(embeddings[0])
        vec2 = np.array(embeddings[1])
        
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return float(similarity)
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self.embedding_dim
    
    # Required abstract methods for BaseEmbedding
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a query (required by BaseEmbedding)"""
        return self._get_text_embedding_single(query)
    
    def _aget_query_embedding(self, query: str) -> List[float]:
        """Async version of _get_query_embedding (required by BaseEmbedding)"""
        # For now, just return the sync version
        # In a real implementation, you might want to use async/await
        return self._get_text_embedding_single(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text (required by BaseEmbedding)"""
        return self._get_text_embedding_single(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts (required by BaseEmbedding)"""
        return self._get_text_embedding_batch(texts)


# Factory function for easy integration
def create_bge_onnx_embedding(
    model_path: str,
    device: str = "gpu",
    cache_dir: Optional[str] = None,
    batch_size: int = 32
) -> BGEOpenXEmbedding:
    """
    Factory function to create BGE ONNX embedding model
    
    Args:
        model_path: Path to BGE model directory
        device: Device to use ("gpu" or "cpu")
        cache_dir: Optional cache directory
        batch_size: Batch size for processing
        
    Returns:
        BGEOpenXEmbedding instance
    """
    return BGEOpenXEmbedding(
        model_path=model_path,
        device=device,
        cache_dir=cache_dir,
        batch_size=batch_size
    )


# Example usage
if __name__ == "__main__":
    # Test the wrapper
    model_path = "../../hf_cache/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    model_path = '../../hf_cache/models--BAAI--bge-small-zh-v1.5/snapshots/7999e1d3359715c523056ef9478215996d62a620'
    
    try:
        # Create embedding model
        embedding_model = create_bge_onnx_embedding(model_path, device="gpu")
        
        # Test single text embedding
        text = "Kubernetes is a container orchestration platform"
        embedding = embedding_model.get_text_embedding(text)
        print(f"✅ Single embedding generated, dimension: {len(embedding)}")
        
        # Test batch embedding
        texts = [
            "Kubernetes is a container orchestration platform",
            "Docker is a containerization technology"
        ]
        embeddings = embedding_model.get_text_embedding_batch(texts)
        print(f"✅ Batch embeddings generated: {len(embeddings)} embeddings")
        
        # Test similarity
        similarity = embedding_model.similarity(texts[0], texts[1])
        print(f"📊 Similarity: {similarity:.4f}")
        
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please update the model_path in the test code")
