"""
文本嵌入服务
"""

import torch
from typing import List, Union
import os
from app.core.config import settings
from app.core.logging import get_logger
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from huggingface_hub import snapshot_download


class EmbeddingService:
    """文本嵌入服务类"""
    
    def __init__(self):
        self.logger = get_logger("EmbeddingService")
        self.model_name = settings.EMBEDDING_MODEL
        self.device = settings.EMBEDDING_DEVICE
        self.model: HuggingFaceEmbedding = None
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化嵌入模型"""
        try:
            self.logger.info(f"🔄 Loading embedding model: {self.model_name}")
            
            # 检查设备可用性
            if self.device == "cuda" and not torch.cuda.is_available():
                self.logger.warning("⚠️ CUDA not available, switching to CPU")
                self.device = "cpu"

            # 配置 HF 镜像与离线模式
            if settings.HF_MIRROR_BASE_URL:
                os.environ["HF_ENDPOINT"] = settings.HF_MIRROR_BASE_URL
                os.environ["HUGGINGFACE_HUB_BASE_URL"] = settings.HF_MIRROR_BASE_URL
            if settings.EMBEDDING_CACHE_DIR:
                os.environ["TRANSFORMERS_CACHE"] = settings.EMBEDDING_CACHE_DIR
                os.environ["HF_HOME"] = settings.EMBEDDING_CACHE_DIR            
            # 用 HuggingFaceEmbedding 包装
            model_path = settings.EMBEDDING_MODEL
            
            # 检查是否有本地模型目录
            local_dir = os.getenv("EMBEDDING_LOCAL_DIR", "").strip()
            if local_dir and os.path.isdir(local_dir):
                model_path = local_dir
                self.logger.info(f"Using local model: {model_path}")
            else:
                # 如果没有本地模型，尝试从缓存或下载
                if not os.path.exists(settings.EMBEDDING_CACHE_DIR):
                    os.mkdir(settings.EMBEDDING_CACHE_DIR)
                self.logger.info(f"Trying to download {settings.EMBEDDING_MODEL} to {settings.EMBEDDING_CACHE_DIR} from {settings.HF_MIRROR_BASE_URL}")
                model_path = snapshot_download(
                    settings.EMBEDDING_MODEL,
                    endpoint=settings.HF_MIRROR_BASE_URL,
                    cache_dir=settings.EMBEDDING_CACHE_DIR
                )
                self.logger.info(f"model downloaded to {model_path}")

            self.logger.info(f"create embedding model")
            self.logger.info(f"model_path: {model_path}")
            self.logger.info(f"device: {settings.EMBEDDING_DEVICE}")
            self.model = HuggingFaceEmbedding(
                model_name=model_path,
                device=self.device
            )

            self.logger.info(f"✅ Embedding model loaded successfully, device: {self.device}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load embedding model: {e}")
            raise
    
    def encode(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        将文本编码为向量
        
        Args:
            texts: 单个文本或文本列表
            
        Returns:
            向量列表
        """
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            # 使用 HuggingFaceEmbedding 的 get_text_embedding_batch 方法
            embeddings = self.model.get_text_embedding_batch(texts)
            
            # 确保返回格式一致
            if len(texts) == 1:
                return [embeddings[0]]
            else:
                return embeddings
                
        except Exception as e:
            self.logger.error(f"❌ Text encoding failed: {e}")
            raise
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        批量编码文本
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            向量列表
        """
        try:
            if not texts:
                return []
            
            all_embeddings = []
            
            # 分批处理
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.encode(batch_texts)
                all_embeddings.extend(batch_embeddings)
                
                self.logger.debug(f"📦 Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            self.logger.info(f"✅ Batch encoding completed, processed {len(texts)} texts")
            return all_embeddings
            
        except Exception as e:
            self.logger.error(f"❌ Batch encoding failed: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """获取嵌入向量维度"""
        try:
            # 使用一个简单的测试文本获取维度
            test_embedding = self.model.get_text_embedding("test")
            return len(test_embedding)
        except Exception as e:
            self.logger.error(f"❌ Failed to get embedding dimension: {e}")
            raise
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            相似度分数 (0-1)
        """
        try:
            embeddings = self.encode([text1, text2])
            
            # 计算余弦相似度
            import numpy as np
            vec1 = np.array(embeddings[0])
            vec2 = np.array(embeddings[1])
            
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate similarity: {e}")
            raise
    
    def find_most_similar(self, query: str, candidates: List[str], top_k: int = 5) -> List[tuple]:
        """
        找到最相似的候选文本
        
        Args:
            query: 查询文本
            candidates: 候选文本列表
            top_k: 返回前k个最相似的结果
            
        Returns:
            (文本, 相似度分数) 的列表，按相似度降序排列
        """
        try:
            if not candidates:
                return []
            
            # 编码所有文本
            all_texts = [query] + candidates
            embeddings = self.encode(all_texts)
            
            query_embedding = embeddings[0]
            candidate_embeddings = embeddings[1:]
            
            # 计算相似度
            import numpy as np
            similarities = []
            
            for i, candidate_embedding in enumerate(candidate_embeddings):
                similarity = np.dot(query_embedding, candidate_embedding)
                similarities.append((candidates[i], similarity))
            
            # 按相似度排序并返回前k个
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            self.logger.error(f"❌ Failed to find most similar text: {e}")
            raise
    
    def __del__(self):
        """清理资源"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
        except:
            pass
