"""
文档处理器 - 文本分割和向量化
"""

import os
import sys
import uuid
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# 在导入任何模块之前设置环境变量
def set_environment_variables(milvus_uri: str = None, collection_name: str = None, chunk_size: int = None, chunk_overlap: int = None):
    """设置环境变量，必须在导入任何模块之前调用"""
    if milvus_uri:
        os.environ["MILVUS_URI"] = milvus_uri
        print(f'✅ 设置 MILVUS_URI: {milvus_uri}')
    if collection_name:
        os.environ["COLLECTION_NAME"] = collection_name
        print(f'✅ 设置 COLLECTION_NAME: {collection_name}')
    if chunk_size:
        os.environ["CHUNK_SIZE"] = str(chunk_size)
        print(f'✅ 设置 CHUNK_SIZE: {chunk_size}')
    if chunk_overlap:
        os.environ["CHUNK_OVERLAP"] = str(chunk_overlap)
        print(f'✅ 设置 CHUNK_OVERLAP: {chunk_overlap}')

# 解析命令行参数
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

# 获取命令行参数并设置环境变量
args = parse_arguments()
set_environment_variables(
    milvus_uri=args.milvus_uri,
    collection_name=args.collection_name,
    chunk_size=args.chunk_size,
    chunk_overlap=args.chunk_overlap
)

# 现在导入项目模块（环境变量已经设置）
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from app.services.embedding_service import EmbeddingService
from app.services.milvus_service import MilvusService
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("DocumentProcessor")


class DocumentProcessor:
    """文档处理器"""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.milvus_service = MilvusService()
        
        # 创建文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
        
        print(f'🔧 当前配置:')
        print(f'   - MILVUS_URI: {settings.MILVUS_URI}')
        print(f'   - COLLECTION_NAME: {settings.COLLECTION_NAME}')
        print(f'   - CHUNK_SIZE: {settings.CHUNK_SIZE}')
        print(f'   - CHUNK_OVERLAP: {settings.CHUNK_OVERLAP}')
    
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
        for line in lines[:10]:  # 只检查前10行
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
        
        # 批量编码
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


async def main():
    """主函数"""
    processor = DocumentProcessor()
    
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
