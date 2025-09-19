"""
复杂检索服务 - 结合 Milvus 向量数据库、Elasticsearch 和 CrossEncoder 重排序
"""

import asyncio
import time
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError as ESConnectionError
from sentence_transformers import CrossEncoder
from huggingface_hub import snapshot_download
import torch

from app.core.config import settings
from app.core.logging import get_logger
from app.services.milvus_service import MilvusService
from app.services.embedding_service import EmbeddingService


@dataclass
class RetrievalRequest:
    """检索请求"""
    query: str
    top_k: int = 10
    milvus_weight: float = 0.6
    elasticsearch_weight: float = 0.4
    rerank_top_k: int = 20
    use_reranking: bool = True


@dataclass
class RetrievalResult:
    """检索结果"""
    query: str
    documents: List[Dict[str, Any]]
    milvus_results: List[Dict[str, Any]]
    elasticsearch_results: List[Dict[str, Any]]
    reranked_results: List[Dict[str, Any]]
    execution_time: float
    metadata: Dict[str, Any]


class RRFReranker:
    """RRF (Reciprocal Rank Fusion) 重排序器"""
    
    def __init__(self, k: int = 60):
        """
        初始化 RRF 重排序器
        
        Args:
            k: RRF 算法中的常数，通常设置为 60
        """
        self.logger = get_logger("RRFReranker")
        self.k = k
    
    def rerank(
        self, 
        milvus_results: List[Dict[str, Any]], 
        elasticsearch_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        使用 RRF 算法重排序结果
        
        Args:
            milvus_results: Milvus 搜索结果
            elasticsearch_results: Elasticsearch 搜索结果
            
        Returns:
            重排序后的结果列表
        """
        try:
            self.logger.info(f"🔄 Starting RRF reranking with k={self.k}")
            
            # 创建文档ID到RRF分数的映射
            doc_scores = {}
            
            # 处理 Milvus 结果
            for rank, doc in enumerate(milvus_results):
                doc_id = doc.get('id')
                if doc_id:
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = {
                            'id': doc_id,
                            'content': doc.get('content', ''),
                            'metadata': doc.get('metadata', {}),
                            'file_path': doc.get('file_path', ''),
                            'rrf_score': 0.0,
                            'milvus_rank': rank + 1,
                            'elasticsearch_rank': None,
                            'sources': []
                        }
                    
                    # 计算 RRF 分数：1 / (k + rank)
                    rrf_score = 1.0 / (self.k + rank + 1)
                    doc_scores[doc_id]['rrf_score'] += rrf_score
                    doc_scores[doc_id]['sources'].append('milvus')
            
            # 处理 Elasticsearch 结果
            for rank, doc in enumerate(elasticsearch_results):
                doc_id = doc.get('id')
                if doc_id:
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = {
                            'id': doc_id,
                            'content': doc.get('content', ''),
                            'metadata': {},
                            'file_path': doc.get('file_path', ''),
                            'rrf_score': 0.0,
                            'milvus_rank': None,
                            'elasticsearch_rank': rank + 1,
                            'sources': []
                        }
                    else:
                        doc_scores[doc_id]['elasticsearch_rank'] = rank + 1
                    
                    # 计算 RRF 分数：1 / (k + rank)
                    rrf_score = 1.0 / (self.k + rank + 1)
                    doc_scores[doc_id]['rrf_score'] += rrf_score
                    doc_scores[doc_id]['sources'].append('elasticsearch')
                    
                    # 添加高亮信息
                    if 'highlights' in doc:
                        doc_scores[doc_id]['highlights'] = doc['highlights']
            
            # 按 RRF 分数排序
            reranked_results = list(doc_scores.values())
            reranked_results.sort(key=lambda x: x['rrf_score'], reverse=True)
            
            self.logger.info(f"✅ RRF reranking completed, processed {len(reranked_results)} documents")
            return reranked_results
            
        except Exception as e:
            self.logger.error(f"❌ RRF reranking failed: {e}")
            # 如果 RRF 失败，返回合并的结果
            combined = milvus_results + elasticsearch_results
            return combined


class CrossEncoderReranker:
    """CrossEncoder 重排序器"""
    
    def __init__(self, model_path: str = "/gemini/pretrain/BAAI/bge-reranker-base"):
        self.logger = get_logger("CrossEncoderReranker")
        self.model_path = model_path
        self.model_name = "BAAI/bge-reranker-base"
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """初始化嵌入模型"""
        try:
            self.logger.info(f"🔄 Loading reranker model: {self.model_name}")
            
            # 检查设备可用性
            self.device = settings.EMBEDDING_DEVICE
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
            
            # 尝试从缓存或下载
            if not os.path.exists(settings.EMBEDDING_CACHE_DIR):
                os.mkdir(settings.EMBEDDING_CACHE_DIR)
            self.logger.info(f"Trying to download {self.model_name} to {settings.EMBEDDING_CACHE_DIR} from {settings.HF_MIRROR_BASE_URL}")
            model_path = snapshot_download(
                self.model_name,
                endpoint=settings.HF_MIRROR_BASE_URL,
                cache_dir=settings.EMBEDDING_CACHE_DIR
            )
            self.logger.info(f"model downloaded to {model_path}")

            self.logger.info(f"create reranker model")
            self.logger.info(f"model_path: {model_path}")
            self.logger.info(f"device: {self.device}")
            
            self.model = CrossEncoder(model_path)

            self.logger.info(f"✅ Reranker model loaded successfully, device: {self.device}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load reranker model: {e}")
            raise

    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        重排序文档
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回前k个结果
            
        Returns:
            重排序后的文档列表
        """
        if not self.model or not documents:
            return documents[:top_k]
        
        try:
            # 准备查询-文档对
            pairs = []
            for doc in documents:
                content = doc.get('content', '')
                if content:
                    pairs.append([query, content])
            
            if not pairs:
                return documents[:top_k]
            
            # 计算相关性分数
            scores = self.model.predict(pairs)
            
            # 将分数添加到文档中
            for i, doc in enumerate(documents):
                doc['rerank_score'] = float(scores[i])
            
            # 按重排序分数排序
            reranked_docs = sorted(documents, key=lambda x: x.get('rerank_score', 0), reverse=True)
            
            self.logger.info(f"✅ Reranked {len(documents)} documents, returning top {top_k}")
            return reranked_docs[:top_k]
            
        except Exception as e:
            self.logger.error(f"❌ Reranking failed: {e}")
            return documents[:top_k]


class ElasticsearchService:
    """Elasticsearch 服务"""
    
    def __init__(self):
        self.logger = get_logger("ElasticsearchService")
        self.client: Optional[Elasticsearch] = None
        self.index_name = getattr(settings, 'ELASTICSEARCH_INDEX', 'k8s-docs')
        self.host = getattr(settings, 'ELASTICSEARCH_HOST', 'http://localhost:9200')
        self.username = getattr(settings, 'ELASTICSEARCH_USER', 'elastic')
        self.password = getattr(settings, 'ELASTICSEARCH_PASSWORD', 'password')
        self.ca_certs = getattr(settings, 'ELASTICSEARCH_CA_CERTS', None)
    
    async def initialize(self):
        """初始化 Elasticsearch 连接"""
        try:
            self.logger.info(f"🔄 Connecting to Elasticsearch: {self.host}")
            
            # 构建连接参数
            connection_params = {
                'hosts': [self.host],
                'basic_auth': (self.username, self.password)
            }
            
            if self.ca_certs and os.path.exists(self.ca_certs):
                connection_params['ca_certs'] = self.ca_certs
            
            self.client = Elasticsearch(**connection_params)
            
            # 测试连接
            info = self.client.info()
            self.logger.info(f"✅ Connected to Elasticsearch: {info.get('version', {}).get('number', 'unknown')}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Elasticsearch: {e}")
            self.client = None
    
    async def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        在 Elasticsearch 中搜索
        
        Args:
            query: 搜索查询
            top_k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        if not self.client:
            self.logger.warning("⚠️ Elasticsearch client not initialized")
            return []
        
        try:
            # 构建搜索查询
            search_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["text^2", "file_path"],
                        "type": "best_fields",
                        "fuzziness": "AUTO"
                    }
                },
                "size": top_k,
                "_source": ["text", "file_path", "chunk_index"],
                "highlight": {
                    "fields": {
                        "text": {
                            "fragment_size": 150,
                            "number_of_fragments": 3
                        }
                    }
                }
            }
            
            # 执行搜索
            response = self.client.search(
                index=self.index_name,
                body=search_body
            )
            
            # 处理结果
            results = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                highlight = hit.get('highlight', {})
                
                result = {
                    'id': hit['_id'],
                    'content': source.get('text', ''),
                    'file_path': source.get('file_path', ''),
                    'chunk_index': source.get('chunk_index', 0),
                    'score': hit['_score'],
                    'source': 'elasticsearch',
                    'highlights': highlight.get('text', [])
                }
                results.append(result)
            
            self.logger.info(f"✅ Elasticsearch search completed, found {len(results)} results, requested {top_k}")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Elasticsearch search failed: {e}")
            return []
    
    async def close(self):
        """关闭连接"""
        try:
            if self.client:
                self.client.close()
            self.logger.info("✅ Elasticsearch connection closed")
        except Exception as e:
            self.logger.error(f"❌ Failed to close Elasticsearch connection: {e}")


class ContextualCompressionRetriever:
    """上下文压缩检索器"""
    
    def __init__(self):
        self.logger = get_logger("ContextualCompressionRetriever")
        self.milvus_service = MilvusService()
        self.elasticsearch_service = ElasticsearchService()
        self.embedding_service = EmbeddingService()
        self.rrf_reranker = RRFReranker()
        self.cross_encoder_reranker = CrossEncoderReranker()
    
    async def initialize(self):
        """初始化所有服务"""
        try:
            await self.milvus_service.initialize()
            await self.elasticsearch_service.initialize()
            self.logger.info("✅ ContextualCompressionRetriever initialized")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize ContextualCompressionRetriever: {e}")
            raise
    
    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """
        执行复杂检索
        
        Args:
            request: 检索请求
            
        Returns:
            检索结果
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"🔍 Starting complex retrieval for query: {request.query[:50]}...")
            
            # 1. 并行执行 Milvus 和 Elasticsearch 检索
            milvus_task = asyncio.create_task(self._milvus_search(request))
            elasticsearch_task = asyncio.create_task(self._elasticsearch_search(request))
            
            milvus_results, elasticsearch_results = await asyncio.gather(
                milvus_task, elasticsearch_task
            )
            
            # rrf_results = await self._combine_results(
            #     milvus_results,
            #     elasticsearch_results,
            #     request.milvus_weight,
            #     request.elasticsearch_weight
            # )

            # 2. 使用 RRF 算法进行第一次重排序
            rrf_results = self.rrf_reranker.rerank(milvus_results, elasticsearch_results)
            
            # 3. 使用 CrossEncoder 进行第二次重排序（如果启用）
            reranked_results = rrf_results
            if request.use_reranking and len(rrf_results) > 1:
                # 只对前 top_k*2 个 RRF 结果进行 CrossEncoder 重排序，提高效率
                candidates_for_rerank = rrf_results[:request.top_k * 2]
                reranked_results = self.cross_encoder_reranker.rerank(
                    request.query, 
                    candidates_for_rerank, 
                    request.rerank_top_k
                )
            
            # 4. 返回最终结果
            final_results = reranked_results[:request.top_k]
            
            execution_time = time.time() - start_time
            
            result = RetrievalResult(
                query=request.query,
                documents=final_results,
                milvus_results=milvus_results,
                elasticsearch_results=elasticsearch_results,
                reranked_results=reranked_results,
                execution_time=execution_time,
                metadata={
                    'milvus_weight': request.milvus_weight,
                    'elasticsearch_weight': request.elasticsearch_weight,
                    'use_reranking': request.use_reranking,
                    'total_milvus_results': len(milvus_results),
                    'total_elasticsearch_results': len(elasticsearch_results),
                    'total_rrf_results': len(rrf_results),
                    'final_results_count': len(final_results),
                    'reranking_method': 'RRF + CrossEncoder' if request.use_reranking else 'RRF only'
                }
            )
            
            self.logger.info(f"✅ Complex retrieval completed in {execution_time:.2f}s, found {len(final_results)} results")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Complex retrieval failed: {e}")
            raise
    
    async def _milvus_search(self, request: RetrievalRequest) -> List[Dict[str, Any]]:
        """执行 Milvus 向量搜索"""
        try:
            # 生成查询向量
            query_embedding = self.embedding_service.encode(request.query)[0]
            
            # 在 Milvus 中搜索
            similar_docs = await self.milvus_service.search_similar(
                query_embedding=query_embedding,
                top_k=request.top_k * 4  # 获取更多结果用于后续处理
            )
            
            # 格式化结果
            milvus_results = []
            for doc in similar_docs:
                milvus_results.append({
                    'id': doc.get('id'),
                    'content': doc.get('content'),
                    'metadata': doc.get('metadata', {}),
                    'score': doc.get('score', 0.0),
                    'source': 'milvus',
                    'file_path': doc.get('file_path', ''),
                    'distance': doc.get('distance', 0.0)
                })
            
            return milvus_results
            
        except Exception as e:
            self.logger.error(f"❌ Milvus search failed: {e}")
            return []
    
    async def _elasticsearch_search(self, request: RetrievalRequest) -> List[Dict[str, Any]]:
        """执行 Elasticsearch 文本搜索"""
        try:
            # 在 Elasticsearch 中搜索
            es_results = await self.elasticsearch_service.search(
                query=request.query,
                top_k=request.top_k * 4  # 获取更多结果用于后续处理
            )
            
            return es_results
            
        except Exception as e:
            self.logger.error(f"❌ Elasticsearch search failed: {e}")
            return []
    
    async def _combine_results(
        self, 
        milvus_results: List[Dict[str, Any]], 
        elasticsearch_results: List[Dict[str, Any]],
        milvus_weight: float,
        elasticsearch_weight: float
    ) -> List[Dict[str, Any]]:
        """合并和去重结果"""
        try:
            # 创建文档ID到结果的映射
            doc_map = {}
            
            # 处理 Milvus 结果
            for result in milvus_results:
                doc_id = result.get('id')
                if doc_id:
                    if doc_id not in doc_map:
                        doc_map[doc_id] = {
                            'id': doc_id,
                            'content': result.get('content', ''),
                            'metadata': result.get('metadata', {}),
                            'file_path': result.get('file_path', ''),
                            'milvus_score': 0.0,
                            'elasticsearch_score': 0.0,
                            'combined_score': 0.0,
                            'sources': []
                        }
                    
                    doc_map[doc_id]['milvus_score'] = result.get('score', 0.0)
                    doc_map[doc_id]['sources'].append('milvus')
            
            # 处理 Elasticsearch 结果
            for result in elasticsearch_results:
                doc_id = result.get('id')
                if doc_id:
                    if doc_id not in doc_map:
                        doc_map[doc_id] = {
                            'id': doc_id,
                            'content': result.get('content', ''),
                            'metadata': {},
                            'file_path': result.get('file_path', ''),
                            'milvus_score': 0.0,
                            'elasticsearch_score': 0.0,
                            'combined_score': 0.0,
                            'sources': []
                        }
                    
                    doc_map[doc_id]['elasticsearch_score'] = result.get('score', 0.0)
                    doc_map[doc_id]['sources'].append('elasticsearch')
                    
                    # 添加高亮信息
                    if 'highlights' in result:
                        doc_map[doc_id]['highlights'] = result['highlights']
            
            # 计算组合分数并排序
            combined_results = []
            for doc_id, doc in doc_map.items():
                # 归一化分数
                milvus_score = doc['milvus_score']
                elasticsearch_score = doc['elasticsearch_score']
                
                # 计算组合分数
                combined_score = (
                    milvus_score * milvus_weight + 
                    elasticsearch_score * elasticsearch_weight
                )
                
                doc['combined_score'] = combined_score
                combined_results.append(doc)
            
            # 按组合分数排序
            combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
            
            self.logger.info(f"✅ Combined {len(milvus_results)} Milvus and {len(elasticsearch_results)} Elasticsearch results into {len(combined_results)} unique documents")
            return combined_results
            
        except Exception as e:
            self.logger.error(f"❌ Result combination failed: {e}")
            return []
    
    async def close(self):
        """关闭所有服务"""
        try:
            await self.milvus_service.close()
            await self.elasticsearch_service.close()
            self.logger.info("✅ ContextualCompressionRetriever closed")
        except Exception as e:
            self.logger.error(f"❌ Failed to close ContextualCompressionRetriever: {e}")


class ComplexRetrievalService:
    """复杂检索服务主类"""
    
    def __init__(self):
        self.logger = get_logger("ComplexRetrievalService")
        self.retriever = ContextualCompressionRetriever()
        self._initialized = False
    
    async def initialize(self):
        """初始化服务"""
        if not self._initialized:
            await self.retriever.initialize()
            self._initialized = True
            self.logger.info("✅ ComplexRetrievalService initialized")
    
    async def search(
        self, 
        query: str, 
        top_k: int = 10,
        milvus_weight: float = 0.6,
        elasticsearch_weight: float = 0.4,
        use_reranking: bool = True
    ) -> RetrievalResult:
        """
        执行复杂检索
        
        Args:
            query: 搜索查询
            top_k: 返回结果数量
            milvus_weight: Milvus 结果权重
            elasticsearch_weight: Elasticsearch 结果权重
            use_reranking: 是否使用重排序
            
        Returns:
            检索结果
        """
        if not self._initialized:
            await self.initialize()
        
        request = RetrievalRequest(
            query=query,
            top_k=top_k,
            milvus_weight=milvus_weight,
            elasticsearch_weight=elasticsearch_weight,
            use_reranking=use_reranking,
            rerank_top_k=top_k * 2
        )
        
        return await self.retriever.retrieve(request)
    
    async def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取文档"""
        try:
            # 首先尝试从 Milvus 获取
            # 这里需要实现根据ID查询的功能
            # 暂时返回 None
            return None
        except Exception as e:
            self.logger.error(f"❌ Failed to get document by ID: {e}")
            return None
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        try:
            milvus_stats = await self.retriever.milvus_service.get_collection_stats()
            
            return {
                'milvus_stats': milvus_stats,
                'elasticsearch_connected': self.retriever.elasticsearch_service.client is not None,
                'rrf_reranker_available': True,  # RRF 重排序器总是可用的
                'cross_encoder_reranker_available': self.retriever.cross_encoder_reranker.model is not None,
                'service_initialized': self._initialized
            }
        except Exception as e:
            self.logger.error(f"❌ Failed to get stats: {e}")
            return {}
    
    async def close(self):
        """关闭服务"""
        try:
            await self.retriever.close()
            self._initialized = False
            self.logger.info("✅ ComplexRetrievalService closed")
        except Exception as e:
            self.logger.error(f"❌ Failed to close ComplexRetrievalService: {e}")

