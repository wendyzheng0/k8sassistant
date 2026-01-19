"""
Elasticsearch Data Access Client
统一的 Elasticsearch 数据访问层，支持存储和关键字检索

同时供 data_processing (存储) 和 backend (关键字检索) 使用
"""

import asyncio
import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from elasticsearch import Elasticsearch
from llama_index.core.schema import TextNode
from llama_index.vector_stores.elasticsearch import ElasticsearchStore


@dataclass
class ElasticsearchConfig:
    """Elasticsearch 配置"""
    es_url: str = ""
    index_name: str = ""
    username: str = ""
    password: str = ""
    batch_size: int = 200
    
    # 搜索配置
    request_timeout: float = 10.0
    max_retries: int = 2
    retry_on_timeout: bool = True
    enable_highlight: bool = True
    enable_fuzziness: bool = True
    
    def __post_init__(self):
        """从 shared config 加载默认值"""
        from shared.config import get_settings
        settings = get_settings()
        
        if not self.es_url:
            self.es_url = settings.ELASTICSEARCH_HOST
        if not self.index_name:
            self.index_name = settings.ELASTICSEARCH_INDEX
        if not self.username:
            self.username = settings.ELASTICSEARCH_USER
        if not self.password:
            self.password = settings.ELASTICSEARCH_PASSWORD


@dataclass
class StorageResult:
    """存储操作结果"""
    success: bool = True
    stored_count: int = 0
    error_count: int = 0
    errors: List[str] = field(default_factory=list)
    
    def add_error(self, error: str):
        self.errors.append(error)
        self.error_count += 1
        self.success = False


class ElasticsearchClient:
    """
    Elasticsearch 统一客户端
    
    提供两种使用方式:
    1. 使用 LlamaIndex ElasticsearchStore 进行文档存储（适合 data_processing）
    2. 使用 elasticsearch-py 进行关键字检索（适合 backend）
    """
    
    def __init__(self, config: Optional[ElasticsearchConfig] = None):
        self.config = config or ElasticsearchConfig()
        self.logger = logging.getLogger("shared.elasticsearch")
        
        # elasticsearch-py 客户端（用于检索）
        self._client: Optional[Elasticsearch] = None
        
        # LlamaIndex ElasticsearchStore（用于存储）
        self._vector_store: Optional[ElasticsearchStore] = None
        
        self._initialized = False
    
    async def initialize(self, for_storage: bool = False) -> None:
        """
        初始化 Elasticsearch 连接
        
        Args:
            for_storage: 是否用于存储（会初始化 LlamaIndex Store）
        """
        if self._initialized:
            return
        
        try:
            self.logger.info(f"🔗 Connecting to Elasticsearch: {self.config.es_url}")
            
            # 初始化 elasticsearch-py 客户端
            connection_params = {
                'hosts': [self.config.es_url],
                'basic_auth': (self.config.username, self.config.password),
                'request_timeout': self.config.request_timeout,
                'max_retries': self.config.max_retries,
                'retry_on_timeout': self.config.retry_on_timeout,
            }
            
            self._client = Elasticsearch(**connection_params)
            
            # 测试连接
            info = await asyncio.to_thread(self._client.info)
            self.logger.info(f"✅ Connected to Elasticsearch: {info.get('version', {}).get('number', 'unknown')}")
            
            if for_storage:
                # 初始化 LlamaIndex Store 用于存储
                self._vector_store = ElasticsearchStore(
                    es_url=self.config.es_url,
                    index_name=self.config.index_name,
                    es_user=self.config.username,
                    es_password=self.config.password,
                )
            
            self._initialized = True
            self.logger.info(f"✅ Elasticsearch initialized: {self.config.index_name}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Elasticsearch: {e}")
            self.logger.info("💡 Troubleshooting tips:")
            self.logger.info("   1. Check if Elasticsearch is running")
            self.logger.info(f"   2. Verify the host URL ({self.config.es_url})")
            self.logger.info("   3. Check username/password credentials")
            # 不抛出异常，允许系统在没有 ES 的情况下继续运行
            self._client = None
    
    # ==================== 存储操作 ====================
    
    async def store_documents(self, documents: List[Dict[str, Any]]) -> StorageResult:
        """
        存储文档到 Elasticsearch
        
        支持两种模式（自动检测）:
        1. 有 embedding 时：使用 LlamaIndex ElasticsearchStore（向量+关键字混合检索）
        2. 无 embedding 时：使用 elasticsearch-py 直接存储（纯关键字/BM25 检索）
        
        注意：LlamaIndex 的 ElasticsearchStore.add() 即使设置 BM25Strategy，
        存储时仍然要求 embedding，因此纯关键字模式必须用 elasticsearch-py。
        
        Args:
            documents: 文档列表，每个文档包含 id, content, metadata（embedding 可选）
            
        Returns:
            StorageResult: 存储结果
        """
        result = StorageResult()
        
        if not documents:
            self.logger.warning("⚠️ No documents to store")
            return result
        
        # 检查是否有 embedding，决定使用哪种存储方式
        has_embeddings = any(doc.get("embedding") for doc in documents)
        
        if has_embeddings:
            # 有 embedding：使用 LlamaIndex ElasticsearchStore
            return await self._store_with_llamaindex(documents, result)
        else:
            # 无 embedding：直接用 elasticsearch-py 存储（BM25 关键字检索）
            return await self._store_text_only(documents, result)
    
    async def _store_with_llamaindex(self, documents: List[Dict[str, Any]], result: StorageResult) -> StorageResult:
        """使用 LlamaIndex ElasticsearchStore 存储带向量的文档"""
        if not self._vector_store:
            self._vector_store = ElasticsearchStore(
                es_url=self.config.es_url,
                index_name=self.config.index_name,
                es_user=self.config.username,
                es_password=self.config.password,
            )
        
        nodes: List[TextNode] = []
        for doc in documents:
            doc_id = doc.get("id")
            content = doc.get("content")
            embedding = doc.get("embedding")
            
            if not doc_id or not content or not embedding:
                result.add_error("Invalid document: missing id, content, or embedding")
                continue
            
            nodes.append(TextNode(
                id_=doc_id,
                text=content,
                embedding=embedding,
                metadata=doc.get("metadata", {}),
            ))
        
        if not nodes:
            result.add_error("No valid documents to store")
            return result
        
        try:
            total_stored = 0
            for i in range(0, len(nodes), self.config.batch_size):
                batch = nodes[i:i + self.config.batch_size]
                self._vector_store.add(batch)
                total_stored += len(batch)
                self.logger.debug(f"📦 Stored batch {i // self.config.batch_size + 1}")
            
            result.stored_count = total_stored
            self.logger.info(f"✅ Stored {total_stored} documents with embeddings to Elasticsearch")
        except Exception as e:
            result.add_error(f"Failed to store: {str(e)}")
        
        return result
    
    async def _store_text_only(self, documents: List[Dict[str, Any]], result: StorageResult) -> StorageResult:
        """
        使用 elasticsearch-py 直接存储纯文本文档（BM25 关键字检索）
        
        LlamaIndex 的 ElasticsearchStore.add() 即使用 BM25Strategy，
        仍然会检查 embedding，因此纯文本必须用原生 ES 客户端。
        
        注意：使用与 Milvus 相同的 doc_id 格式（relative_path#chunk_index），
        以便在 RRF reranker 中正确融合两边的结果。
        """
        if not self._client:
            result.add_error("Elasticsearch client not initialized")
            return result
        
        # 确保索引存在，使用适合关键字检索的 mapping
        await self._ensure_text_index_exists()
        
        try:
            total_stored = 0
            for i in range(0, len(documents), self.config.batch_size):
                batch = documents[i:i + self.config.batch_size]
                actions = []
                
                for doc in batch:
                    doc_id = doc.get("id")
                    content = doc.get("content", "")
                    metadata = doc.get("metadata", {})
                    
                    if not doc_id or not content:
                        result.add_error("Invalid document: missing id or content")
                        continue
                    
                    # 获取文件路径（优先使用相对路径，确保一致性）
                    file_path = metadata.get("relative_path", "") or metadata.get("file_path", "")
                    chunk_index = metadata.get("chunk_index", 0)
                    
                    # 使用统一的 doc_id 格式（与 Milvus 保持一致）
                    # 优先使用 metadata 中的 doc_id，否则使用传入的 id
                    unified_doc_id = metadata.get("doc_id", doc_id)
                    
                    # 构建 ES 文档
                    es_doc = {
                        "text": content,
                        "file_path": file_path,  # 使用相对路径
                        "chunk_index": chunk_index,
                        "source": metadata.get("source", ""),
                        "doc_id": unified_doc_id,  # 统一的 doc_id 用于 reranker 融合
                    }
                    
                    # 使用 unified_doc_id 作为 ES 文档 ID
                    actions.append({"index": {"_index": self.config.index_name, "_id": unified_doc_id}})
                    actions.append(es_doc)
                
                if actions:
                    await asyncio.to_thread(
                        self._client.bulk,
                        body=actions,
                        refresh=True
                    )
                    total_stored += len(batch)
                    self.logger.debug(f"📦 Stored batch {i // self.config.batch_size + 1}")
            
            result.stored_count = total_stored
            self.logger.info(f"✅ Stored {total_stored} text-only documents to Elasticsearch (BM25)")
        except Exception as e:
            result.add_error(f"Failed to store: {str(e)}")
        
        return result
    
    async def _ensure_text_index_exists(self) -> None:
        """确保纯文本索引存在，创建适合 BM25 关键字检索的 mapping"""
        try:
            index_exists = await asyncio.to_thread(
                self._client.indices.exists,
                index=self.config.index_name
            )
            if not index_exists:
                mapping = {
                    "mappings": {
                        "properties": {
                            "text": {"type": "text", "analyzer": "standard"},
                            "file_path": {"type": "keyword"},
                            "chunk_index": {"type": "integer"},
                            "source": {"type": "keyword"},
                            "doc_id": {"type": "keyword"},
                        }
                    }
                }
                await asyncio.to_thread(
                    self._client.indices.create,
                    index=self.config.index_name,
                    body=mapping
                )
                self.logger.info(f"✅ Created text index: {self.config.index_name}")
        except Exception as e:
            self.logger.warning(f"⚠️ Index check/create warning: {e}")
    
    # ==================== 关键字检索操作 ====================
    
    async def text_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        关键字/文本检索（BM25）
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        if not self._client:
            self.logger.warning("⚠️ Elasticsearch client not initialized")
            return []
        
        try:
            # 根据 query 长度决定是否启用模糊匹配
            token_count = len(query.split())
            use_fuzziness = self.config.enable_fuzziness and token_count <= 8
            use_highlight = self.config.enable_highlight and top_k <= 20
            
            # 构建搜索查询
            multi_match: Dict[str, Any] = {
                "query": query,
                "fields": ["text^2", "file_path"],
                "type": "best_fields",
            }
            
            if use_fuzziness:
                multi_match.update({
                    "fuzziness": "AUTO",
                    "max_expansions": 50,
                    "prefix_length": 1,
                })
            
            search_body = {
                "query": {
                    "multi_match": multi_match
                },
                "size": top_k,
                "_source": ["text", "file_path", "chunk_index", "doc_id"],  # 添加 doc_id 用于 reranker 融合
                "track_total_hits": False,
            }
            
            if use_highlight:
                search_body["highlight"] = {
                    "fields": {
                        "text": {
                            "fragment_size": 150,
                            "number_of_fragments": 3,
                        }
                    }
                }
            
            # 执行搜索
            response = await asyncio.to_thread(
                self._client.search,
                index=self.config.index_name,
                body=search_body,
                request_timeout=self.config.request_timeout,
            )
            
            # 处理结果
            results = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                highlight = hit.get('highlight', {})
                
                # 获取统一的 doc_id（用于 reranker 融合）
                # 优先使用存储时保存的 doc_id，否则使用 ES 文档 ID
                doc_id = source.get('doc_id') or hit['_id']
                
                results.append({
                    'id': hit['_id'],
                    'doc_id': doc_id,  # 添加 doc_id 用于 reranker 融合
                    'content': source.get('text', ''),
                    'file_path': source.get('file_path', ''),
                    'chunk_index': source.get('chunk_index', 0),
                    'score': hit['_score'],
                    'source': 'elasticsearch',
                    'highlights': highlight.get('text', [])
                })
            
            self.logger.info(f"🔍 Text search completed, found {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Text search failed: {e}")
            return []
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        if not self._client:
            return {"status": "not_initialized"}
        
        try:
            index_stats = await asyncio.to_thread(
                self._client.indices.stats,
                index=self.config.index_name
            )
            doc_count = index_stats["_all"]["primaries"]["docs"]["count"]
            
            return {
                "index_name": self.config.index_name,
                "doc_count": doc_count,
                "status": "exists",
            }
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to get index stats: {e}")
            return {"status": "error", "error": str(e)}
    
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._client is not None
    
    async def close(self) -> None:
        """关闭连接"""
        try:
            if self._client:
                self._client.close()
            self._initialized = False
            self.logger.info("✅ Elasticsearch connection closed")
        except Exception as e:
            self.logger.error(f"❌ Failed to close: {e}")

