"""
文档处理器 - 文本分割和向量化
Uses shared modules for embedding and configuration

@deprecated: 此模块已弃用，请使用新的流水线架构

新的使用方式:
    from data_processing.processors import PipelineRunner
    
    runner = PipelineRunner()
    result = await runner.run(data_dir="./data/zh-cn", storage_backend="milvus")

或使用命令行:
    python -m data_processing.processors.cli --data-dir ./data/zh-cn --backend milvus
"""

import warnings
warnings.warn(
    "document_processor.py is deprecated. Use 'from data_processing.processors import PipelineRunner' instead.",
    DeprecationWarning,
    stacklevel=2
)

import os
import sys
import uuid
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path for shared module imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv

# Load environment variables
env_path = os.path.join(project_root, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Import from shared modules
from shared.config import get_settings
from shared.embeddings import create_embedding_service, EmbeddingService


# Simple logger for data processing
import logging

def get_logger(name: str) -> logging.Logger:
    """Get a configured logger"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger("DocumentProcessor")


class MilvusServiceWrapper:
    """
    Wrapper for Milvus operations
    Uses pymilvus directly instead of backend service
    """
    
    def __init__(self, uri: str, collection_name: str, vector_dim: int):
        self.uri = uri
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.client = None
        self.logger = get_logger("MilvusServiceWrapper")
    
    async def initialize(self):
        """Initialize Milvus connection"""
        from pymilvus import MilvusClient, connections, Collection, CollectionSchema, FieldSchema, DataType
        from urllib.parse import urlparse
        
        try:
            # Parse URI
            raw_uri = self.uri.strip()
            if "://" in raw_uri:
                parsed = urlparse(raw_uri)
                host = parsed.hostname
                port = parsed.port
                client_uri = f"{parsed.scheme}://{host}:{port}"
            else:
                if ":" not in raw_uri:
                    raise ValueError(f"Invalid MILVUS_URI (missing port): {raw_uri}")
                host, port_str = raw_uri.rsplit(":", 1)
                port = int(port_str)
                client_uri = f"http://{host}:{port}"
            
            # Connect to Milvus
            self.logger.info(f"Connecting to Milvus: {host}:{port}")
            connections.connect(alias="default", host=host, port=port)
            
            # Create MilvusClient
            self.client = MilvusClient(uri=client_uri, token="")
            
            # Ensure collection exists
            await self._ensure_collection_exists()
            
            self.logger.info(f"✅ Milvus connection initialized: {self.uri}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Milvus: {e}")
            raise
    
    async def _ensure_collection_exists(self):
        """Ensure collection exists"""
        from pymilvus import Collection, CollectionSchema, FieldSchema, DataType
        
        collections = self.client.list_collections()
        
        if self.collection_name not in collections:
            # Create collection
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=65535, is_primary=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata", dtype=DataType.JSON),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim)
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="K8s 文档向量存储"
            )
            
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema
            )
            
            # Create index
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
            
            self.logger.info(f"✅ Created collection: {self.collection_name}")
        else:
            collection = Collection(self.collection_name)
            collection.load()
            self.logger.info(f"✅ Collection already exists: {self.collection_name}")
    
    async def insert_documents(self, documents: List[Dict[str, Any]]):
        """Insert documents into vector database"""
        if not documents:
            self.logger.warning("⚠️ No documents to insert")
            return
        
        data = []
        for doc in documents:
            if not doc.get("id") or not doc.get("content") or not doc.get("embedding"):
                continue
            data.append({
                "id": doc["id"],
                "content": doc["content"],
                "metadata": doc.get("metadata", {}),
                "embedding": doc["embedding"]
            })
        
        if data:
            self.client.insert(
                collection_name=self.collection_name,
                data=data
            )
            self.logger.info(f"✅ Successfully inserted {len(data)} documents")
    
    async def close(self):
        """Close connection"""
        from pymilvus import connections
        try:
            if self.client:
                self.client.close()
            connections.disconnect("default")
            self.logger.info("✅ Milvus connection closed")
        except Exception as e:
            self.logger.error(f"❌ Failed to close Milvus connection: {e}")


class DocumentProcessor:
    """文档处理器 - 使用shared模块"""
    
    def __init__(
        self,
        milvus_uri: str = None,
        collection_name: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        # Get settings
        settings = get_settings()
        
        # Use provided values or defaults from settings
        self.milvus_uri = milvus_uri or os.getenv("MILVUS_URI", settings.MILVUS_URI)
        self.collection_name = collection_name or os.getenv("COLLECTION_NAME", settings.COLLECTION_NAME)
        self.chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", str(settings.CHUNK_SIZE)))
        self.chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", str(settings.CHUNK_OVERLAP)))
        
        # Initialize embedding service using shared module
        self.embedding_service = create_embedding_service(use_singleton=True)
        
        # Get vector dimension from embedding service
        self.vector_dim = self.embedding_service.get_embedding_dimension()
        
        # Initialize Milvus wrapper
        self.milvus_service = MilvusServiceWrapper(
            uri=self.milvus_uri,
            collection_name=self.collection_name,
            vector_dim=self.vector_dim
        )
        
        # Create text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
        
        print(f'🔧 当前配置:')
        print(f'   - MILVUS_URI: {self.milvus_uri}')
        print(f'   - COLLECTION_NAME: {self.collection_name}')
        print(f'   - CHUNK_SIZE: {self.chunk_size}')
        print(f'   - CHUNK_OVERLAP: {self.chunk_overlap}')
        print(f'   - VECTOR_DIM: {self.vector_dim}')
    
    async def initialize(self):
        """初始化服务"""
        await self.milvus_service.initialize()
        logger.info("✅ 文档处理器初始化完成")
    
    async def process_documents(self, docs_dir: str = None):
        """处理文档目录"""
        docs_path = Path(docs_dir or "docs")
        if not docs_path.exists():
            logger.error(f"❌ 文档目录不存在: {docs_dir}")
            return
        
        logger.info(f"📁 开始处理文档目录: {docs_dir}")
        
        # 收集所有文档文件
        doc_files = []
        for ext in ['.txt', '.md', '.html']:
            doc_files.extend(docs_path.rglob(f"*{ext}"))
        
        logger.info(f"📋 找到 {len(doc_files)} 个文档文件")
        
        # 处理每个文档
        all_chunks = []
        for doc_file in doc_files:
            try:
                chunks = await self._process_single_document(doc_file)
                all_chunks.extend(chunks)
                logger.info(f"✅ 处理文档: {doc_file.name} -> {len(chunks)} 个块")
            except Exception as e:
                logger.error(f"❌ 处理文档失败 {doc_file.name}: {e}")
        
        logger.info(f"📊 总共生成 {len(all_chunks)} 个文本块")
        
        # 批量向量化
        if all_chunks:
            await self._vectorize_and_store(all_chunks)
        
        logger.info("✅ 文档处理完成")
    
    async def _process_single_document(self, file_path: Path) -> List[Dict[str, Any]]:
        """处理单个文档"""
        # 读取文件内容
        content = await self._read_file(file_path)
        if not content:
            return []
        
        # 提取元数据
        metadata = self._extract_metadata(file_path, content)
        
        # 文本分割
        chunks = self._split_text(content, metadata)
        
        return chunks
    
    async def _read_file(self, file_path: Path) -> str:
        """读取文件内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            logger.error(f"❌ 读取文件失败 {file_path}: {e}")
            return ""
    
    def _extract_metadata(self, file_path: Path, content: str) -> Dict[str, Any]:
        """提取文档元数据"""
        metadata = {
            "filename": file_path.name,
            "file_path": str(file_path),
            "file_size": len(content),
            "file_type": file_path.suffix,
            "title": file_path.stem
        }
        
        # 尝试从内容中提取标题
        lines = content.split('\n')
        for line in lines[:10]:
            line = line.strip()
            if line.startswith('# '):
                metadata["title"] = line[2:].strip()
                break
            elif line.startswith('title:'):
                metadata["title"] = line[6:].strip()
                break
        
        return metadata
    
    def _split_text(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """分割文本"""
        # 创建 LangChain Document
        doc = Document(page_content=content, metadata=metadata)
        
        # 分割文本
        chunks = self.text_splitter.split_documents([doc])
        
        # 转换为字典格式
        result = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "id": str(uuid.uuid4()),
                "content": chunk.page_content,
                "metadata": {
                    **chunk.metadata,
                    "chunk_index": i,
                    "chunk_id": str(uuid.uuid4())
                }
            }
            result.append(chunk_data)
        
        return result
    
    async def _vectorize_and_store(self, chunks: List[Dict[str, Any]]):
        """向量化并存储文本块"""
        logger.info("🔄 开始向量化文本块...")
        
        # 批量编码 using shared embedding service
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embedding_service.encode_batch(texts, batch_size=32)
        
        # 准备存储数据
        documents = []
        for chunk, embedding in zip(chunks, embeddings):
            document = {
                "id": chunk["id"],
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "embedding": embedding
            }
            documents.append(document)
        
        # 存储到 Milvus
        await self.milvus_service.insert_documents(documents)
        
        logger.info(f"✅ 成功存储 {len(documents)} 个文档到向量数据库")
    
    async def process_single_file(self, file_path: str) -> bool:
        """处理单个文件"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"❌ 文件不存在: {file_path}")
                return False
            
            chunks = await self._process_single_document(file_path)
            if chunks:
                await self._vectorize_and_store(chunks)
                logger.info(f"✅ 文件处理成功: {file_path.name}")
                return True
            else:
                logger.warning(f"⚠️ 文件内容为空: {file_path.name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 处理文件失败 {file_path}: {e}")
            return False
    
    async def close(self):
        """关闭服务"""
        await self.milvus_service.close()


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="K8s Assistant 文档处理器")
    parser.add_argument(
        "--milvus-uri", 
        default="http://localhost:19530",
        help="Milvus 服务地址 (默认: http://localhost:19530)"
    )
    parser.add_argument(
        "--collection-name", 
        default="k8s_docs",
        help="集合名称 (默认: k8s_docs)"
    )
    parser.add_argument(
        "--docs-dir", 
        default="docs",
        help="文档目录路径 (默认: docs)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="文本块大小 (默认: 512)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="文本块重叠大小 (默认: 50)"
    )
    parser.add_argument(
        "--single-file",
        help="处理单个文件"
    )
    return parser.parse_args()


async def main():
    """主函数"""
    args = parse_arguments()
    
    processor = DocumentProcessor(
        milvus_uri=args.milvus_uri,
        collection_name=args.collection_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    try:
        await processor.initialize()
        
        if args.single_file:
            # 处理单个文件
            success = await processor.process_single_file(args.single_file)
            if success:
                logger.info("✅ 单文件处理完成")
            else:
                logger.error("❌ 单文件处理失败")
        else:
            # 处理文档目录
            await processor.process_documents(args.docs_dir)
            
    finally:
        await processor.close()


if __name__ == "__main__":
    asyncio.run(main())
