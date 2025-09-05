"""
Milvus vector database service
"""

import asyncio
import os
from typing import List, Dict, Any, Optional
from pymilvus import MilvusClient, connections, Collection
from urllib.parse import urlparse
from app.core.config import settings
from app.core.logging import get_logger


class MilvusService:
    """Milvus vector database service class"""
    
    def __init__(self):
        self.client: Optional[MilvusClient] = None
        self.logger = get_logger("MilvusService")
        self.collection_name = settings.COLLECTION_NAME
        self.vector_dim = settings.VECTOR_DIM
        self.mode = getattr(settings, 'MILVUS_MODE', 'embedded')
        
    async def initialize(self):
        """Initialize Milvus connection"""
        try:
            if self.mode == "embedded":
                await self._initialize_embedded()
            else:
                await self._initialize_standalone()
                
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Milvus connection: {e}")
            raise
    
    async def _initialize_embedded(self):
        """Initialize embedded Milvus mode"""
        try:
            self.logger.info("🚀 Initializing embedded Milvus mode...")
            
            # 设置 embedded 模式的环境变量
            os.environ["MILVUS_MODE"] = "embedded"
            
            # 创建数据目录
            data_dir = "/app/milvus_data"
            os.makedirs(data_dir, exist_ok=True)
            
            # 使用 localhost 连接 embedded Milvus
            host = "localhost"
            port = 19530
            
            # 连接到 embedded Milvus
            self.logger.info(f"Connecting to embedded Milvus: {host}:{port}")
            connections.connect(alias="default", host=host, port=port)
            
            # 创建 MilvusClient
            client_uri = f"http://{host}:{port}"
            self.logger.info(f"Creating MilvusClient: {client_uri}")
            self.client = MilvusClient(uri=client_uri, token="")
            
            # 检查集合是否存在，如果不存在则创建
            await self._ensure_collection_exists()
            
            self.logger.info(f"✅ Embedded Milvus connection initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize embedded Milvus: {e}")
            raise
    
    async def _initialize_standalone(self):
        """Initialize standalone Milvus mode"""
        try:
            # 解析并规范化 URI，支持 "host:port" 与 "http(s)://host:port"
            raw_uri = settings.MILVUS_URI.strip()
            if "://" in raw_uri:
                parsed = urlparse(raw_uri)
                if not parsed.hostname or not parsed.port:
                    raise ValueError(f"Invalid MILVUS_URI: {raw_uri}")
                host = parsed.hostname
                port = parsed.port
                client_uri = f"{parsed.scheme}://{host}:{port}"
            else:
                # 纯 host:port 形式
                if ":" not in raw_uri:
                    raise ValueError(f"Invalid MILVUS_URI (missing port): {raw_uri}")
                host, port_str = raw_uri.rsplit(":", 1)
                port = int(port_str)
                client_uri = f"http://{host}:{port}"

            # 连接到 Milvus（gRPC 连接，供部分 SDK API 使用）
            self.logger.info(f"Connecting to Milvus: {host}:{port}")
            connections.connect(alias="default", host=host, port=port)

            # 创建 MilvusClient（HTTP 接口）
            self.logger.info(f"Creating MilvusClient: {client_uri}")
            self.client = MilvusClient(uri=client_uri, token="")  # 如有鉴权请配置 token
            
            # 检查集合是否存在，如果不存在则创建
            await self._ensure_collection_exists()
            
            self.logger.info(f"✅ Milvus connection initialized successfully: {settings.MILVUS_URI}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize standalone Milvus: {e}")
            raise
    
    async def _ensure_collection_exists(self):
        """Ensure collection exists, create if not"""
        try:
            # 检查集合是否存在
            collections = self.client.list_collections()
            
            if self.collection_name not in collections:
                # 创建集合
                schema = {
                    "fields": [
                        {
                            "name": "id",
                            "dtype": "VARCHAR",
                            "max_length": 65535,
                            "is_primary": True
                        },
                        {
                            "name": "content",
                            "dtype": "VARCHAR",
                            "max_length": 65535
                        },
                        {
                            "name": "metadata",
                            "dtype": "JSON"
                        },
                        {
                            "name": "embedding",
                            "dtype": "FLOAT_VECTOR",
                            "dim": self.vector_dim
                        }
                    ],
                    "description": "Kubernetes 文档向量存储"
                }
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    schema=schema,
                    properties={"collection.ttl.seconds": 0},
                    dimension=self.vector_dim
                )
                
                # 创建索引（使用 ORM 接口，因为 MilvusClient 2.3.4 无 create_index 方法）
                collection = Collection(self.collection_name)
                collection.create_index(
                    field_name="embedding",
                    index_params={
                        "index_type": "IVF_FLAT",
                        "metric_type": "IP",
                        "params": {"nlist": 1024}
                    }
                )
                collection.load()
                
                self.logger.info(f"✅ Created collection and index: {self.collection_name}")
            else:
                # 集合已存在，确保索引存在
                collection = Collection(self.collection_name)
                try:
                    has_indexes = getattr(collection, "indexes", None)
                    if not has_indexes:
                        collection.create_index(
                            field_name="embedding",
                            index_params={
                                "index_type": "IVF_FLAT",
                                "metric_type": "IP",
                                "params": {"nlist": 1024}
                            }
                        )
                    collection.load()
                except Exception:
                    # 忽略检查索引过程中的非致命错误，后续操作若失败再上抛
                    pass
                
                self.logger.info(f"✅ Collection already exists: {self.collection_name}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to ensure collection exists: {e}")
            raise
    
    async def insert_documents(self, documents: List[Dict[str, Any]]):
        """Insert documents into vector database"""
        try:
            if not documents:
                self.logger.warning("⚠️ No documents to insert")
                return
            
            # 准备插入数据
            data = []
            for doc in documents:
                # 验证必要的字段
                if not doc.get("id") or not doc.get("content") or not doc.get("embedding"):
                    self.logger.warning(f"⚠️ Skipping invalid document: {doc.get('id', 'unknown')}")
                    continue
                    
                data.append({
                    "id": doc["id"],
                    "content": doc["content"],
                    "metadata": doc.get("metadata", {}),
                    "embedding": doc["embedding"]
                })
            
            if not data:
                self.logger.warning("⚠️ No valid documents to insert")
                return
            
            # 插入数据
            self.client.insert(
                collection_name=self.collection_name,
                data=data
            )
            
            self.logger.info(f"✅ Successfully inserted {len(data)} documents")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to insert documents: {e}")
            # Don't throw exception, just log error
            self.logger.error(f"Insertion failed, but application will continue running")
    
    async def search_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            # 首先检查集合是否有数据
            try:
                # 使用正确的 API 获取集合信息
                collections = self.client.list_collections()
                if self.collection_name not in collections:
                    self.logger.warning("⚠️ Collection does not exist, returning empty results")
                    return []
                
                # 尝试获取集合统计信息（如果可用）
                try:
                    # 使用 describe_collection 获取集合信息
                    collection_info = self.client.describe_collection(self.collection_name)
                    self.logger.info(f"Collection info: {collection_info}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Unable to get collection details: {e}")
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Unable to check collection status: {e}")
                # 继续尝试搜索，如果失败再处理
            
            # 执行向量搜索 - 修复参数问题
            try:
                # 使用正确的参数格式，避免参数冲突
                # 在 pymilvus 2.6.1 中，search_params 包含所有搜索参数
                results = self.client.search(
                    collection_name=self.collection_name,
                    data=[query_embedding],
                    search_params={
                        "metric_type": "IP",
                        "params": {"nprobe": 10}
                    },
                    limit=top_k,
                    output_fields=["*"]  # 使用 "*" 获取所有字段，避免字段名不匹配问题
                )
                self.logger.info("✅ Search successful with IP metric type")
            except Exception as e:
                self.logger.warning(f"⚠️ IP metric search failed: {e}")
                try:
                    # 尝试使用 COSINE 度量类型
                    self.logger.info("Trying COSINE metric type...")
                    results = self.client.search(
                        collection_name=self.collection_name,
                        data=[query_embedding],
                        search_params={
                            "metric_type": "COSINE",
                            "params": {"nprobe": 10}
                        },
                        limit=top_k,
                        output_fields=["*"]
                    )
                    self.logger.info("✅ Search successful with COSINE metric type")
                except Exception as e2:
                    self.logger.warning(f"⚠️ COSINE metric search failed: {e2}")
                    # 尝试简化的参数格式作为最后的回退
                    self.logger.info("Trying simplified search parameter format...")
                    results = self.client.search(
                        collection_name=self.collection_name,
                        data=[query_embedding],
                        limit=top_k,
                        output_fields=["*"]
                    )
                    self.logger.info("✅ Search successful with simplified parameters")
            
            # 格式化结果
            search_results = []
            self.logger.info(f"Search results: {results}")
            
            # 检查结果是否为空或无效
            if not results or len(results) == 0:
                self.logger.info("🔍 Search completed, no relevant documents found")
                return []
                
            for result in results[0]:  # results[0] 包含第一个查询的结果
                # 使用与 test_milvus_dump.py 相同的数据提取方式
                entity = result.get("entity", {})
                
                # 处理相似度分数
                similarity_score = result.get("score")
                distance = result.get("distance")
                
                # 转换距离为相似度分数（如果需要）
                if distance is not None and similarity_score is None:
                    similarity_score = 1.0 / (1.0 + distance) if distance > 0 else 1.0
                
                # 安全地提取内容，处理不同的字段名
                content = ""
                if isinstance(entity, dict):
                    # 尝试不同的可能字段名
                    content = entity.get("content", entity.get("text", entity.get("data", "")))
                elif isinstance(result, dict):
                    # 如果 entity 不是字典，直接从 result 中获取
                    content = result.get("content", result.get("text", result.get("data", "")))
                
                # 安全地提取元数据
                metadata = {}
                if isinstance(entity, dict):
                    metadata = entity.get("metadata", {})
                elif isinstance(result, dict):
                    metadata = result.get("metadata", {})
                
                search_results.append({
                    "id": result.get("id", "unknown"),
                    "content": content,
                    "metadata": metadata,
                    "score": similarity_score,
                    "distance": distance,
                    "entity": entity  # 保留原始 entity 用于调试
                })
            
            self.logger.info(f"🔍 Search completed, returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            self.logger.error(f"❌ Failed to search similar documents: {e}")
            # Return empty results instead of throwing exception to prevent app crash
            return []
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            # 检查集合是否存在
            collections = self.client.list_collections()
            if self.collection_name not in collections:
                return {
                    "collection_name": self.collection_name,
                    "row_count": 0,
                    "vector_dim": self.vector_dim,
                    "status": "not_exists"
                }
            
            # 尝试获取集合信息
            try:
                collection_info = self.client.describe_collection(self.collection_name)
                # 从集合信息中提取行数（如果可用）
                row_count = 0
                if "num_rows" in collection_info:
                    row_count = collection_info["num_rows"]
                elif "row_count" in collection_info:
                    row_count = collection_info["row_count"]
                
                return {
                    "collection_name": self.collection_name,
                    "row_count": row_count,
                    "vector_dim": self.vector_dim,
                    "status": "exists",
                    "collection_info": collection_info
                }
            except Exception as e:
                self.logger.warning(f"⚠️ Unable to get collection details: {e}")
                return {
                    "collection_name": self.collection_name,
                    "row_count": 0,
                    "vector_dim": self.vector_dim,
                    "status": "exists_but_no_details"
                }
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get collection statistics: {e}")
            # Return default values instead of throwing exception
            return {
                "collection_name": self.collection_name,
                "row_count": 0,
                "vector_dim": self.vector_dim,
                "status": "error"
            }
    
    async def delete_documents(self, document_ids: List[str]):
        """Delete specified documents"""
        try:
            if not document_ids:
                self.logger.warning("⚠️ No document IDs to delete")
                return
                
            self.client.delete(
                collection_name=self.collection_name,
                pks=document_ids
            )
            self.logger.info(f"✅ Successfully deleted {len(document_ids)} documents")
        except Exception as e:
            self.logger.error(f"❌ Failed to delete documents: {e}")
            # Don't throw exception, just log error
            self.logger.error(f"Deletion failed, but application will continue running")
    
    async def close(self):
        """Close connection"""
        try:
            if self.client:
                self.client.close()
            connections.disconnect("default")
            self.logger.info("✅ Milvus connection closed")
        except Exception as e:
            self.logger.error(f"❌ Failed to close Milvus connection: {e}")
    
    def __del__(self):
        """Destructor, ensure connection is closed"""
        try:
            asyncio.create_task(self.close())
        except:
            pass
