"""
Milvus 向量数据库服务
"""

import asyncio
import os
from typing import List, Dict, Any, Optional
from pymilvus import MilvusClient, connections, Collection
from urllib.parse import urlparse
from app.core.config import settings
from app.core.logging import get_logger


class MilvusService:
    """Milvus 向量数据库服务类"""
    
    def __init__(self):
        self.client: Optional[MilvusClient] = None
        self.logger = get_logger("MilvusService")
        self.collection_name = settings.COLLECTION_NAME
        self.vector_dim = settings.VECTOR_DIM
        self.mode = getattr(settings, 'MILVUS_MODE', 'embedded')
        
    async def initialize(self):
        """初始化 Milvus 连接"""
        try:
            if self.mode == "embedded":
                await self._initialize_embedded()
            else:
                await self._initialize_standalone()
                
        except Exception as e:
            self.logger.error(f"❌ Milvus 连接初始化失败: {e}")
            raise
    
    async def _initialize_embedded(self):
        """初始化 embedded Milvus 模式"""
        try:
            self.logger.info("🚀 初始化 embedded Milvus 模式...")
            
            # 设置 embedded 模式的环境变量
            os.environ["MILVUS_MODE"] = "embedded"
            
            # 创建数据目录
            data_dir = "/app/milvus_data"
            os.makedirs(data_dir, exist_ok=True)
            
            # 使用 localhost 连接 embedded Milvus
            host = "localhost"
            port = 19530
            
            # 连接到 embedded Milvus
            self.logger.info(f"连接到 embedded Milvus: {host}:{port}")
            connections.connect(alias="default", host=host, port=port)
            
            # 创建 MilvusClient
            client_uri = f"http://{host}:{port}"
            self.logger.info(f"创建 MilvusClient: {client_uri}")
            self.client = MilvusClient(uri=client_uri, token="")
            
            # 检查集合是否存在，如果不存在则创建
            await self._ensure_collection_exists()
            
            self.logger.info(f"✅ Embedded Milvus 连接初始化成功")
            
        except Exception as e:
            self.logger.error(f"❌ Embedded Milvus 初始化失败: {e}")
            raise
    
    async def _initialize_standalone(self):
        """初始化 standalone Milvus 模式"""
        try:
            # 解析并规范化 URI，支持 "host:port" 与 "http(s)://host:port"
            raw_uri = settings.MILVUS_URI.strip()
            if "://" in raw_uri:
                parsed = urlparse(raw_uri)
                if not parsed.hostname or not parsed.port:
                    raise ValueError(f"非法的 MILVUS_URI: {raw_uri}")
                host = parsed.hostname
                port = parsed.port
                client_uri = f"{parsed.scheme}://{host}:{port}"
            else:
                # 纯 host:port 形式
                if ":" not in raw_uri:
                    raise ValueError(f"非法的 MILVUS_URI（缺少端口）: {raw_uri}")
                host, port_str = raw_uri.rsplit(":", 1)
                port = int(port_str)
                client_uri = f"http://{host}:{port}"

            # 连接到 Milvus（gRPC 连接，供部分 SDK API 使用）
            self.logger.info(f"连接到 Milvus: {host}:{port}")
            connections.connect(alias="default", host=host, port=port)

            # 创建 MilvusClient（HTTP 接口）
            self.logger.info(f"创建 MilvusClient: {client_uri}")
            self.client = MilvusClient(uri=client_uri, token="")  # 如有鉴权请配置 token
            
            # 检查集合是否存在，如果不存在则创建
            await self._ensure_collection_exists()
            
            self.logger.info(f"✅ Milvus 连接初始化成功: {settings.MILVUS_URI}")
            
        except Exception as e:
            self.logger.error(f"❌ Standalone Milvus 初始化失败: {e}")
            raise
    
    async def _ensure_collection_exists(self):
        """确保集合存在，如果不存在则创建"""
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
                        "metric_type": "COSINE",
                        "params": {"nlist": 1024}
                    }
                )
                collection.load()
                
                self.logger.info(f"✅ 创建集合和索引: {self.collection_name}")
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
                                "metric_type": "COSINE",
                                "params": {"nlist": 1024}
                            }
                        )
                    collection.load()
                except Exception:
                    # 忽略检查索引过程中的非致命错误，后续操作若失败再上抛
                    pass
                
                self.logger.info(f"✅ 集合已存在: {self.collection_name}")
                
        except Exception as e:
            self.logger.error(f"❌ 确保集合存在失败: {e}")
            raise
    
    async def insert_documents(self, documents: List[Dict[str, Any]]):
        """插入文档到向量数据库"""
        try:
            if not documents:
                self.logger.warning("⚠️ 没有文档需要插入")
                return
            
            # 准备插入数据
            data = []
            for doc in documents:
                # 验证必要的字段
                if not doc.get("id") or not doc.get("content") or not doc.get("embedding"):
                    self.logger.warning(f"⚠️ 跳过无效文档: {doc.get('id', 'unknown')}")
                    continue
                    
                data.append({
                    "id": doc["id"],
                    "content": doc["content"],
                    "metadata": doc.get("metadata", {}),
                    "embedding": doc["embedding"]
                })
            
            if not data:
                self.logger.warning("⚠️ 没有有效的文档需要插入")
                return
            
            # 插入数据
            self.client.insert(
                collection_name=self.collection_name,
                data=data
            )
            
            self.logger.info(f"✅ 成功插入 {len(data)} 个文档")
            
        except Exception as e:
            self.logger.error(f"❌ 插入文档失败: {e}")
            # 不抛出异常，只记录错误
            self.logger.error(f"插入失败，但应用将继续运行")
    
    async def search_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索相似文档"""
        try:
            # 首先检查集合是否有数据
            try:
                # 使用正确的 API 获取集合信息
                collections = self.client.list_collections()
                if self.collection_name not in collections:
                    self.logger.warning("⚠️ 集合不存在，返回空结果")
                    return []
                
                # 尝试获取集合统计信息（如果可用）
                try:
                    # 使用 describe_collection 获取集合信息
                    collection_info = self.client.describe_collection(self.collection_name)
                    self.logger.info(f"集合信息: {collection_info}")
                except Exception as e:
                    self.logger.warning(f"⚠️ 无法获取集合详细信息: {e}")
                    
            except Exception as e:
                self.logger.warning(f"⚠️ 无法检查集合状态: {e}")
                # 继续尝试搜索，如果失败再处理
            
            # 执行向量搜索 - 修复参数问题
            try:
                results = self.client.search(
                    collection_name=self.collection_name,
                    data=[query_embedding],
                    anns_field="embedding",
                    param={
                        "metric_type": "COSINE",
                        "params": {"nprobe": 10}
                    },
                    limit=top_k,
                    output_fields=["id", "content", "metadata"]
                )
            except TypeError as e:
                if "multiple values for argument 'anns_field'" in str(e):
                    # 尝试不同的参数格式
                    self.logger.info("尝试使用替代的搜索参数格式...")
                    results = self.client.search(
                        collection_name=self.collection_name,
                        data=[query_embedding],
                        param={
                            "metric_type": "COSINE",
                            "params": {"nprobe": 10}
                        },
                        limit=top_k,
                        output_fields=["id", "content", "metadata"]
                    )
                else:
                    raise
            
            # 格式化结果
            search_results = []
            self.logger.info(f"搜索结果: {results}")
            
            # 检查结果是否为空或无效
            if not results or len(results) == 0:
                self.logger.info("🔍 搜索完成，没有找到相关文档")
                return []
                
            for result in results[0]:  # results[0] 包含第一个查询的结果
                search_results.append({
                    "id": result["id"],
                    "content": result["content"],
                    "metadata": result["metadata"],
                    "score": result["score"]
                })
            
            self.logger.info(f"🔍 搜索完成，返回 {len(search_results)} 个结果")
            return search_results
            
        except Exception as e:
            self.logger.error(f"❌ 搜索相似文档失败: {e}")
            # 返回空结果而不是抛出异常，这样应用不会崩溃
            return []
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
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
                self.logger.warning(f"⚠️ 无法获取集合详细信息: {e}")
                return {
                    "collection_name": self.collection_name,
                    "row_count": 0,
                    "vector_dim": self.vector_dim,
                    "status": "exists_but_no_details"
                }
                
        except Exception as e:
            self.logger.error(f"❌ 获取集合统计信息失败: {e}")
            # 返回默认值而不是抛出异常
            return {
                "collection_name": self.collection_name,
                "row_count": 0,
                "vector_dim": self.vector_dim,
                "status": "error"
            }
    
    async def delete_documents(self, document_ids: List[str]):
        """删除指定文档"""
        try:
            if not document_ids:
                self.logger.warning("⚠️ 没有文档ID需要删除")
                return
                
            self.client.delete(
                collection_name=self.collection_name,
                pks=document_ids
            )
            self.logger.info(f"✅ 成功删除 {len(document_ids)} 个文档")
        except Exception as e:
            self.logger.error(f"❌ 删除文档失败: {e}")
            # 不抛出异常，只记录错误
            self.logger.error(f"删除失败，但应用将继续运行")
    
    async def close(self):
        """关闭连接"""
        try:
            if self.client:
                self.client.close()
            connections.disconnect("default")
            self.logger.info("✅ Milvus 连接已关闭")
        except Exception as e:
            self.logger.error(f"❌ 关闭 Milvus 连接失败: {e}")
    
    def __del__(self):
        """析构函数，确保连接被关闭"""
        try:
            asyncio.create_task(self.close())
        except:
            pass
