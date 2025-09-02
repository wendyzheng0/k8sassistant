"""
文本嵌入服务
"""

import torch
from typing import List, Union
from sentence_transformers import SentenceTransformer
import os
from app.core.config import settings
from app.core.logging import get_logger


class EmbeddingService:
    """文本嵌入服务类"""
    
    def __init__(self):
        self.logger = get_logger("EmbeddingService")
        self.model_name = settings.EMBEDDING_MODEL
        self.device = settings.EMBEDDING_DEVICE
        self.model: SentenceTransformer = None
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化嵌入模型"""
        try:
            self.logger.info(f"🔄 正在加载嵌入模型: {self.model_name}")
            
            # 检查设备可用性
            if self.device == "cuda" and not torch.cuda.is_available():
                self.logger.warning("⚠️ CUDA 不可用，切换到 CPU")
                self.device = "cpu"
            
            # 配置 HF 镜像与离线模式
            if settings.HF_MIRROR_BASE_URL:
                os.environ["HF_ENDPOINT"] = settings.HF_MIRROR_BASE_URL
                os.environ["HUGGINGFACE_HUB_BASE_URL"] = settings.HF_MIRROR_BASE_URL
            if settings.EMBEDDING_CACHE_DIR:
                os.environ["TRANSFORMERS_CACHE"] = settings.EMBEDDING_CACHE_DIR
                os.environ["HF_HOME"] = settings.EMBEDDING_CACHE_DIR

            # 设置离线模式
            if settings.HF_OFFLINE:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"

            local_dir = settings.EMBEDDING_LOCAL_DIR.strip() if isinstance(settings.EMBEDDING_LOCAL_DIR, str) else ""
            use_local = bool(local_dir) and os.path.isdir(local_dir)
            
            model_source = local_dir if use_local else self.model_name

            # 加载模型（优先本地目录，其次网络/镜像）
            if settings.EMBEDDING_CACHE_DIR:
                self.model = SentenceTransformer(
                    model_source,
                    device=self.device,
                    cache_folder=settings.EMBEDDING_CACHE_DIR
                )
            else:
                self.model = SentenceTransformer(
                    model_source,
                    device=self.device
                )
            
            self.logger.info(f"✅ 嵌入模型加载成功，设备: {self.device}")
            
        except Exception as e:
            self.logger.error(f"❌ 嵌入模型加载失败: {e}")
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
            
            # 编码文本
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=False,
                normalize_embeddings=True
            )
            
            # 转换为列表格式
            if len(texts) == 1:
                return [embeddings.tolist()]
            else:
                return embeddings.tolist()
                
        except Exception as e:
            self.logger.error(f"❌ 文本编码失败: {e}")
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
                
                self.logger.debug(f"📦 处理批次 {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            self.logger.info(f"✅ 批量编码完成，共处理 {len(texts)} 个文本")
            return all_embeddings
            
        except Exception as e:
            self.logger.error(f"❌ 批量编码失败: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """获取嵌入向量维度"""
        try:
            # 使用一个简单的测试文本获取维度
            test_embedding = self.encode("test")
            return len(test_embedding[0])
        except Exception as e:
            self.logger.error(f"❌ 获取嵌入维度失败: {e}")
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
            self.logger.error(f"❌ 计算相似度失败: {e}")
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
            self.logger.error(f"❌ 查找最相似文本失败: {e}")
            raise
    
    def __del__(self):
        """清理资源"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
        except:
            pass
