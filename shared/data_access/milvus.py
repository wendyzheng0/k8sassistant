"""
Milvus Data Access Client
统一的 Milvus 数据访问层，支持存储和向量检索

同时供 data_processing (存储) 和 backend (检索) 使用
"""

import asyncio
import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pymilvus import MilvusClient as PyMilvusClient, connections, Collection, CollectionSchema, FieldSchema, DataType
from llama_index.core.schema import TextNode
from llama_index.vector_stores.milvus import MilvusVectorStore


@dataclass
class MilvusConfig:
    """Milvus 配置"""
    uri: str = ""
    collection_name: str = ""
    vector_dim: int = 0
    similarity_metric: str = "COSINE"
    overwrite: bool = False  # 是否覆盖已存在的集合（仅用于存储）
    
    def __post_init__(self):
        """从 shared config 加载默认值"""
        if not self.uri or not self.collection_name or not self.vector_dim:
            from shared.config import get_settings
            settings = get_settings()
            
            if not self.uri:
                self.uri = settings.MILVUS_URI
            if not self.collection_name:
                self.collection_name = settings.COLLECTION_NAME
            if not self.vector_dim:
                self.vector_dim = settings.VECTOR_DIM


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


class MilvusClient:
    """
    Milvus 统一客户端
    
    提供两种使用方式:
    1. 使用 LlamaIndex MilvusVectorStore 进行批量存储（适合 data_processing）
    2. 使用 pymilvus 进行灵活的向量检索（适合 backend）
    """
    
    def __init__(self, config: Optional[MilvusConfig] = None):
        self.config = config or MilvusConfig()
        self.logger = logging.getLogger("shared.milvus")
        
        # pymilvus 客户端（用于检索）
        self._client: Optional[PyMilvusClient] = None
        
        # LlamaIndex VectorStore（用于存储）
        self._vector_store: Optional[MilvusVectorStore] = None
        
        self._initialized = False
    
    async def initialize(self, for_storage: bool = False) -> None:
        """
        初始化 Milvus 连接
        
        Args:
            for_storage: 是否用于存储（会初始化 LlamaIndex VectorStore）
        """
        if self._initialized:
            return
        
        try:
            self.logger.info(f"🔗 Connecting to Milvus: {self.config.uri}")
            
            # 解析 URI
            raw_uri = self.config.uri.strip()
            if "://" in raw_uri:
                parsed = urlparse(raw_uri)
                host = parsed.hostname
                port = parsed.port
                client_uri = f"{parsed.scheme}://{host}:{port}"
            else:
                if ":" in raw_uri:
                    host, port_str = raw_uri.rsplit(":", 1)
                    port = int(port_str)
                else:
                    host = raw_uri
                    port = 19530
                client_uri = f"http://{host}:{port}"
            
            # 连接 pymilvus
            connections.connect(alias="default", host=host, port=port)
            self._client = PyMilvusClient(uri=client_uri, token="")
            
            if for_storage:
                # 初始化 LlamaIndex VectorStore 用于存储
                self._vector_store = MilvusVectorStore(
                    uri=self.config.uri,
                    collection_name=self.config.collection_name,
                    dim=self.config.vector_dim,
                    overwrite=self.config.overwrite,
                    similarity_metric=self.config.similarity_metric,
                )
            else:
                # 确保集合存在（用于检索场景）
                await self._ensure_collection_exists()
            
            self._initialized = True
            self.logger.info(f"✅ Milvus initialized: {self.config.collection_name}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Milvus: {e}")
            raise
    
    async def _ensure_collection_exists(self) -> None:
        """确保集合存在"""
        try:
            collections = self._client.list_collections()
            if self.config.collection_name not in collections:
                self.logger.warning(f"⚠️ Collection {self.config.collection_name} does not exist")
                # 创建集合
                fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=65535, is_primary=True),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="metadata", dtype=DataType.JSON),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.config.vector_dim)
                ]
                schema = CollectionSchema(fields=fields, description="Vector store")
                self._client.create_collection(
                    collection_name=self.config.collection_name,
                    schema=schema
                )
                
                # 创建索引
                collection = Collection(self.config.collection_name)
                collection.create_index(
                    field_name="embedding",
                    index_params={
                        "index_type": "IVF_FLAT",
                        "metric_type": self.config.similarity_metric,
                        "params": {"nlist": 1024}
                    }
                )
                collection.load()
                self.logger.info(f"✅ Created collection: {self.config.collection_name}")
            else:
                # 加载已存在的集合
                collection = Collection(self.config.collection_name)
                collection.load()
                self.logger.info(f"✅ Collection exists: {self.config.collection_name}")
        except Exception as e:
            self.logger.warning(f"⚠️ Error ensuring collection exists: {e}")
    
    # ==================== 存储操作 ====================
    
    async def store_documents(self, documents: List[Dict[str, Any]]) -> StorageResult:
        """
        存储文档到 Milvus（使用 LlamaIndex）
        
        Args:
            documents: 文档列表，每个文档包含 id, content, embedding, metadata
                      metadata 中应包含 doc_id (格式: relative_path#chunk_index) 用于 reranker 融合
            
        Returns:
            StorageResult: 存储结果
        """
        result = StorageResult()
        
        if not documents:
            self.logger.warning("⚠️ No documents to store")
            return result
        
        if not self._vector_store:
            # 如果没有初始化 vector_store，初始化它
            self._vector_store = MilvusVectorStore(
                uri=self.config.uri,
                collection_name=self.config.collection_name,
                dim=self.config.vector_dim,
                overwrite=self.config.overwrite,
                similarity_metric=self.config.similarity_metric,
            )
        
        # 转换为 LlamaIndex TextNode
        nodes: List[TextNode] = []
        for doc in documents:
            if not doc.get("id") or not doc.get("content") or not doc.get("embedding"):
                result.add_error("Invalid document: missing required fields")
                continue
            
            content = doc["content"]
            if len(content) > 65000:
                content = content[:65000]
            
            metadata = doc.get("metadata", {}).copy()
            
            # 确保 doc_id 存在于 metadata 中（用于 reranker 融合）
            # 优先使用 metadata 中的 doc_id，否则使用文档 id
            if "doc_id" not in metadata:
                metadata["doc_id"] = doc["id"]
            
            node = TextNode(
                id_=doc["id"],  # 使用统一的 doc_id 作为主键
                text=content,
                embedding=doc["embedding"],
                metadata=metadata,
            )
            nodes.append(node)
        
        if not nodes:
            result.add_error("No valid documents to store")
            return result
        
        try:
            self._vector_store.add(nodes)
            result.stored_count = len(nodes)
            self.logger.info(f"✅ Stored {len(nodes)} documents to Milvus")
        except Exception as e:
            result.add_error(f"Failed to store: {str(e)}")
        
        return result
    
    # ==================== 检索操作 ====================
    
    async def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        向量相似度检索
        
        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        if not self._client:
            self.logger.error("Milvus client not initialized")
            return []
        
        try:
            # 动态计算 nprobe
            try:
                collection_info = self._client.describe_collection(self.config.collection_name)
                row_count = collection_info.get("num_rows", 0) or collection_info.get("row_count", 0)
                
                if row_count < 1000:
                    nprobe = 10
                elif row_count < 10000:
                    nprobe = 32
                elif row_count < 100000:
                    nprobe = 64
                else:
                    nprobe = 128
            except Exception:
                nprobe = 32
            
            # 执行搜索
            results = self._client.search(
                collection_name=self.config.collection_name,
                data=[query_embedding],
                search_params={
                    "metric_type": self.config.similarity_metric,
                    "params": {"nprobe": nprobe}
                },
                limit=top_k,
                output_fields=["*"]
            )
            
            # 格式化结果
            search_results = []
            if results and len(results) > 0:
                for result in results[0]:
                    entity = result.get("entity", {})
                    
                    # 处理分数
                    score = result.get("score") or result.get("distance")
                    if result.get("distance") is not None and result.get("score") is None:
                        distance = result.get("distance")
                        score = 1.0 / (1.0 + distance) if distance > 0 else 1.0
                    
                    # 提取内容
                    content = entity.get("content", entity.get("text", ""))
                    
                    # 获取 metadata（LlamaIndex 存储时可能放在 entity.metadata 或直接在 entity 中）
                    metadata = entity.get("metadata", {})
                    
                    # 提取 file_path（优先使用相对路径，与 doc_id 保持一致）
                    file_path = (
                        metadata.get("relative_path", "") or 
                        metadata.get("file_path", "") or 
                        entity.get("file_path", "unknown")
                    )
                    
                    # 获取统一的 doc_id（用于 reranker 融合）
                    # 优先从 metadata 中获取，否则使用 entity 中的 doc_id，最后使用 result id
                    doc_id = metadata.get("doc_id") or entity.get("doc_id") or result.get("id", "unknown")
                    
                    search_results.append({
                        "id": result.get("id", "unknown"),
                        "doc_id": doc_id,  # 添加 doc_id 用于 reranker 融合
                        "content": content,
                        "file_path": file_path,
                        "chunk_index": metadata.get("chunk_index", 0),
                        "metadata": metadata,
                        "score": score,
                        "entity": entity,
                    })
            
            self.logger.info(f"🔍 Search completed, returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            self.logger.error(f"❌ Search failed: {e}")
            return []
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        if not self._client:
            return {"status": "not_initialized"}
        
        try:
            collections = self._client.list_collections()
            if self.config.collection_name not in collections:
                return {
                    "collection_name": self.config.collection_name,
                    "row_count": 0,
                    "status": "not_exists"
                }
            
            collection_info = self._client.describe_collection(self.config.collection_name)
            row_count = collection_info.get("num_rows", 0) or collection_info.get("row_count", 0)
            
            return {
                "collection_name": self.config.collection_name,
                "row_count": row_count,
                "vector_dim": self.config.vector_dim,
                "status": "exists",
            }
        except Exception as e:
            self.logger.error(f"❌ Failed to get stats: {e}")
            return {"status": "error", "error": str(e)}
    
    async def delete_documents(self, document_ids: List[str]) -> int:
        """删除文档"""
        if not self._client or not document_ids:
            return 0
        
        try:
            self._client.delete(
                collection_name=self.config.collection_name,
                pks=document_ids
            )
            self.logger.info(f"✅ Deleted {len(document_ids)} documents")
            return len(document_ids)
        except Exception as e:
            self.logger.error(f"❌ Failed to delete: {e}")
            return 0
    
    async def get_chunks_by_file_path(self, file_path: str) -> List[Dict[str, Any]]:
        """
        根据文件路径获取所有相关的分块
        
        Args:
            file_path: 文件的相对路径
            
        Returns:
            分块列表，每个分块包含 id, content, metadata 等信息
        """
        if not self._client:
            self.logger.error("Milvus client not initialized")
            return []
        
        results = []
        
        # 尝试多种查询方式，因为数据可能以不同方式存储
        # 1. 首先尝试使用动态字段查询（LlamaIndex 存储时 metadata 字段会被提升到顶层）
        try:
            filter_expr = f'relative_path == "{file_path}" or file_path == "{file_path}"'
            results = self._client.query(
                collection_name=self.config.collection_name,
                filter=filter_expr,
                output_fields=["*"],
                limit=1000
            )
            self.logger.debug(f"Dynamic field query returned {len(results)} results")
        except Exception as e:
            self.logger.debug(f"Dynamic field query failed: {e}")
        
        # 2. 如果没有结果，尝试使用 id 前缀匹配（doc_id 格式: relative_path#chunk_index）
        if not results:
            try:
                # 使用 like 操作符匹配以 file_path# 开头的 id
                filter_expr = f'id like "{file_path}#%"'
                results = self._client.query(
                    collection_name=self.config.collection_name,
                    filter=filter_expr,
                    output_fields=["*"],
                    limit=1000
                )
                self.logger.debug(f"ID prefix query returned {len(results)} results")
            except Exception as e:
                self.logger.debug(f"ID prefix query failed: {e}")
        
        # 3. 如果还是没有结果，尝试使用 pymilvus Collection API 进行更灵活的查询
        if not results:
            try:
                from pymilvus import Collection
                collection = Collection(self.config.collection_name)
                collection.load()
                
                # 尝试不同的表达式
                expressions = [
                    f'relative_path == "{file_path}"',
                    f'file_path == "{file_path}"',
                    f'id like "{file_path}#%"',
                ]
                
                for expr in expressions:
                    try:
                        query_results = collection.query(
                            expr=expr,
                            output_fields=["*"],
                            limit=1000
                        )
                        if query_results:
                            results = query_results
                            self.logger.debug(f"Collection query with '{expr}' returned {len(results)} results")
                            break
                    except Exception as e:
                        self.logger.debug(f"Query with '{expr}' failed: {e}")
                        continue
            except Exception as e:
                self.logger.debug(f"Collection API query failed: {e}")
        
        # 格式化结果
        chunks = []
        for result in results:
            # 动态字段存储在顶层，不在 metadata 嵌套字段中
            # 构建 metadata 字典从顶层字段
            metadata = {}
            metadata_fields = [
                'file_path', 'file_name', 'file_type', 'file_size',
                'creation_date', 'last_modified_date', 'absolute_path',
                'relative_path', 'chunk_index', 'chunk_id', 'doc_id'
            ]
            for field in metadata_fields:
                if field in result:
                    metadata[field] = result[field]
            
            # 从多个可能的位置提取字段
            content = result.get("content") or result.get("text", "")
            doc_id = result.get("doc_id") or result.get("id", "")
            result_file_path = result.get("relative_path") or result.get("file_path", "")
            chunk_index = result.get("chunk_index", 0)
            
            chunks.append({
                "id": result.get("id", "unknown"),
                "content": content,
                "doc_id": doc_id,
                "file_path": result_file_path,
                "chunk_index": chunk_index,
                "metadata": metadata,
                "entity": result,
            })
        
        self.logger.info(f"🔍 Found {len(chunks)} chunks for file: {file_path}")
        return chunks
    
    async def close(self) -> None:
        """关闭连接"""
        try:
            if self._client:
                self._client.close()
            connections.disconnect("default")
            self._initialized = False
            self.logger.info("✅ Milvus connection closed")
        except Exception as e:
            self.logger.error(f"❌ Failed to close: {e}")

