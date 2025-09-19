import os
import sys
import tempfile
import json
import time
import shutil
import traceback
import html2text
import argparse
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core import Settings
from llama_index.core.readers.base import BaseReader
from llama_index.core.node_parser import HTMLNodeParser, SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import TitleExtractor, KeywordExtractor
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext,
    Document
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.milvus import MilvusVectorStore
# from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from pymilvus import MilvusClient
from huggingface_hub import snapshot_download
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from milvus_lite.server import Server
import data_cleaner
from code_extractor import CodeExtractor
from bge_onnx_llama_wrapper import BGEOpenXEmbedding


if os.path.exists('../../.env'):
    load_dotenv('../../.env')

# Default values
DEFAULT_DB_URI = 'http://localhost:19530'
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'zh-cn-orig', 'docs')
DEFAULT_MILVUS_DATA = "milvus_data"
DEFAULT_START_MILVUS = False

milvus_server = None


def init_embed_model():
    # 从环境变量或配置读取设置
    hf_endpoint = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
    hf_base_url = os.getenv("HUGGINGFACE_HUB_BASE_URL", "https://hf-mirror.com")
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    device = os.getenv("EMBEDDING_DEVICE", "cpu")
    backend = os.getenv("EMBEDDING_BACKEND", "torch")  # torch, onnx, openvino
    cache_dir = os.getenv("EMBEDDING_CACHE_DIR", "hf_cache")
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'hf_cache')
    
    # 初始化嵌入模型
    model_path = model_name
    # 检查是否有本地模型目录
    local_dir = os.getenv("EMBEDDING_LOCAL_DIR", "").strip()
    if local_dir and os.path.isdir(local_dir):
        model_path = local_dir
        print(f"Using local model: {model_path}")
    else:
        # 如果没有本地模型，尝试HuggingFace的缓存或下载
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        print(f"Trying to download {model_name} to {cache_dir} from {hf_endpoint}")
        model_path = snapshot_download(
            model_name,
            endpoint=hf_endpoint,
            cache_dir=cache_dir
        )
        print(f"model downloaded to {model_path}")
    
    print(f"Create embedding model")
    print(f"model_path: {model_path}")
    print(f"device: {device}")
    print(f"backend: {backend}")
    
    # 根据后端类型初始化模型
    if backend == "onnx":
        # 检查ONNX模型路径
        onnx_path = os.path.join(model_path, "onnx", "model.onnx")
        if not os.path.exists(onnx_path):
            print(f"ONNX model path not found: {onnx_path}")
            print("Falling back to PyTorch backend")
            # 回退到PyTorch后端
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=model_path,
                device=device,
                backend="torch"
            )
        else:
            # 使用我们的BGE ONNX包装器
            print("Using BGE ONNX embedding model")
            Settings.embed_model = BGEOpenXEmbedding(
                model_path=model_path,
                device=device,
                cache_dir=cache_dir,
                batch_size=128
            )
    else:
        # PyTorch后端（默认）
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=model_path,
            device=device,
            backend="torch"
        )


def init_llm():
    # 使用 OpenAI 兼容接口，可通过环境变量配置自定义网关
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    model_name = os.getenv("LLM_MODEL", "qwen-plus")
    Settings.llm = OpenAI(model=model_name, base_url=base_url, api_key=api_key)


def start_milvus(milvus_data_path=DEFAULT_MILVUS_DATA, port=19530):
    global milvus_server
    try:
        print("🚀 Starting Milvus server...")
        print(f"📁 Milvus data path: {milvus_data_path}")
        milvus_server = Server(db_file=milvus_data_path, address=f'localhost:{port}')
        milvus_server.start()
        print("✅ Milvus server started successfully")
        
        # 等待服务器完全启动
        import time
        print("⏳ Waiting for server to fully start...")
        time.sleep(3)
        
        # 测试连接
        try:
            from pymilvus import connections
            connections.connect("default", host="localhost", port=port)
            print("✅ Milvus connection test successful")
            connections.disconnect("default")
        except Exception as e:
            print(f"⚠️ Milvus connection test failed: {e}")
    except Exception as e:
        print(f"🔴 Milvus server startup failed: {e}")
        milvus_server = None
        raise


def init_vector_store(db_uri):
    # 创建 MilvusVectorStore 实例
    # 动态推断向量维度以匹配当前嵌入模型
    try:
        inferred_dim = len(Settings.embed_model.get_text_embedding("__dim_probe__"))
    except Exception:
        inferred_dim = 512  # BGE-small-zh-v1.5 512维版本

    vector_store = MilvusVectorStore(
        uri=db_uri,                        # 连接地址
        dim=inferred_dim,                  # 向量维度
        collection_name=os.environ['COLLECTION_NAME'],   # 集合名称
        # enable_sparse=True,  # 启用稀疏向量（BM25）
        overwrite=True,                    # 如果集合存在，是否覆盖
        consistency_level="Strong"         # 一致性级别：Strong, Session, Bounded, Eventually
    )

    print("✅ MilvusVectorStore initialized successfully!")
    print(f"📊 Collection name: {vector_store.collection_name}")
    print(f"🔗 Connection URI: {db_uri}")
    print(f"📏 Vector dimension: {inferred_dim}")
    return vector_store


def process_with_html_parser(data_dir, db_uri):
    # Initialize
    vector_store = init_vector_store(db_uri)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 从环境变量读取chunk配置
    chunk_size = int(os.getenv("CHUNK_SIZE", "1024"))  # 默认1024，比512大一些
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100"))  # 默认100

    print(f"📏 使用chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}")

    # Define transformations
    # 使用SentenceSplitter，在文档预处理阶段已经清理了HTML
    transformations = [
        # 直接使用SentenceSplitter，按chunk_size切分
        SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n\n"  # 使用双换行符作为分隔符
        ),
        # These extractors depends on LLM, so we disable them
        # TitleExtractor(),
        # KeywordExtractor(keywords=3),
    ]
    
    # Create ingestion pipeline
    pipeline = IngestionPipeline(transformations=transformations)

    print(f"Going to load data from {data_dir}")
    # 使用更灵活的过滤方式，过滤掉路径中包含 _print/index.html 的文件
    exclude_files = []
    reader = SimpleDirectoryReader(
        input_dir=data_dir, 
        required_exts=[".html"],
        exclude=exclude_files,
        recursive=True
    )

    batch_nodes = []
    processed_files = 0
    for docs in reader.iter_data():
        for doc in docs:
            # 过滤掉路径中包含 _print/index.html 的文件
            if doc.metadata and 'file_path' in doc.metadata:
                file_path = doc.metadata['file_path']
                if '_print' in file_path:
                    print(f"🚫 Skip file: {file_path} (including _print)")
                    continue
                print(f"Processing file: {file_path}")
            
            # processed_doc = preprocess_html_document(doc)
            processed_doc = preprocess_html_document_safe(doc)
            nodes = pipeline.run(documents=[processed_doc])
            batch_nodes.extend(nodes)
            processed_files += 1
            
            # 调试信息：显示前几个节点的长度
            if processed_files <= 3:  # 只显示前3个文件的调试信息
                print(f"📊 File {processed_files} is processed into {len(nodes)} 个节点")
                for i, node in enumerate(nodes[:5]):  # 只显示前5个节点
                    text_len = len(node.text.strip())
                    print(f"  node {i+1}: length={text_len}, preview: {node.text.strip()[:100]}...")
                if len(nodes) > 5:
                    print(f"  ... There are {len(nodes)-5} more nodes")

    print(f"Process {processed_files} files, cleaning and validating {len(batch_nodes)} nodes...")
    # Clean and validate nodes before creating index
    valid_nodes = data_cleaner.clean_and_validate_nodes(batch_nodes, min_text_length=10)
    
    if not valid_nodes:
        raise ValueError("No valid nodes found for index creation")
    
    # Create index from the processed nodes
    print(f"\n📥 Starting to build vector index from {len(valid_nodes)} nodes...")
    index = VectorStoreIndex(
        nodes=valid_nodes,
        storage_context=storage_context
    )

    print("✅ Vector index construction completed!")
    print(f"📖 Indexed {len(batch_nodes)} nodes")
    return index


def preprocess_html_document_safe(doc: Document) -> Document:
    """
    安全地预处理HTML文档，提取代码块到metadata中
    
    Args:
        doc: 原始文档对象
        
    Returns:
        Document: 预处理后的文档对象
    """
    try:
        # 确保文档文本是字符串
        if not isinstance(doc.text, str):
            doc.text = str(doc.text)
        
        # 清理HTML内容
        cleaned_html = data_cleaner.clean_html_content(doc.text)
        
        # 提取代码块
        code_extractor = CodeExtractor()
        processed_html, extracted_codes = code_extractor.extract_code_blocks(cleaned_html)
        
        # 重新解析处理后的HTML
        processed_soup = BeautifulSoup(processed_html, "html.parser")
        
        # 尝试找到主要内容区域
        content_div = processed_soup.find("div", class_="td-content")
        if content_div:
            # 提取纯文本，移除HTML标签
            processed_text = content_div.get_text(separator="\n", strip=True)
        else:
            # 如果没有找到特定区域，提取整个文档的纯文本
            processed_text = processed_soup.get_text(separator="\n", strip=True)
        
        # 清理处理后的文本
        processed_text = data_cleaner.clean_text(processed_text)
        
        # 清理元数据
        cleaned_metadata = data_cleaner.clean_metadata(doc.metadata)
        
        # 将提取的代码块添加到元数据中
        if extracted_codes:
            cleaned_metadata['extracted_codes'] = extracted_codes
            cleaned_metadata['code_blocks_count'] = len(extracted_codes)
            print(f"Extracted {len(extracted_codes)} code blocks")
        
        # 创建新的文档对象
        return Document(text=processed_text, metadata=cleaned_metadata)
        
    except Exception as e:
        print(f"Exception occurs when preprocessing HTML document: {e}")
        # 如果出错，返回清理后的原始文档
        return data_cleaner.clean_document(doc)


def main():
    """Main function to handle command line arguments and execute data processing"""
    parser = argparse.ArgumentParser(description='Process HTML documents and create vector index')
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR,
                        help=f'Path to the data directory containing HTML files (default: {DEFAULT_DATA_DIR})')
    parser.add_argument('--start-milvus', action='store_true', default=DEFAULT_START_MILVUS,
                        help='Whether to start Milvus server (default: False)')
    parser.add_argument('--milvus-data', type=str, default=DEFAULT_MILVUS_DATA,
                        help=f'Path to Milvus database files (default: {DEFAULT_MILVUS_DATA})')
    parser.add_argument('--db-uri', type=str, default=DEFAULT_DB_URI,
                        help=f'Milvus database URI (default: {DEFAULT_DB_URI})')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.data_dir):
        print(f"🔴 Error: Data directory does not exist: {args.data_dir}")
        return 1
    
    try:
        print("🔧 Initializing embedding model...")
        init_embed_model()
        print("🔧 Initializing LLM...")
        init_llm()
        
        if args.start_milvus:
            print("🔧 Starting Milvus server...")
            start_milvus(args.milvus_data)
        
        print("🔧 Starting data processing...")
        print(f"📂 Data directory: {args.data_dir}")
        print(f"🔗 Database URI: {args.db_uri}")
        
        index = process_with_html_parser(args.data_dir, args.db_uri)
        
        # 确保资源正确清理
        if index is not None:
            if hasattr(index, 'storage_context'):
                # 可选：持久化保存
                # index.storage_context.persist()
                
                # 显式关闭存储上下文
                if hasattr(index.storage_context, 'vector_store'):
                    vector_store = index.storage_context.vector_store
                    if hasattr(vector_store, 'client'):
                        vector_store.client.close()
            
            # 清理索引
            del index
            
        return 0
            
    except Exception as e:
        print(f"🔴 Error occurred during execution: {str(e)}")
        traceback.print_exc()
        return 1
    finally:
        if milvus_server is not None:
            try:
                milvus_server.stop()
                print("✅ Milvus server stopped")
            except Exception as e:
                print(f"⚠️ Error occurred while stopping Milvus server: {e}")
        print("Program execution completed")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
