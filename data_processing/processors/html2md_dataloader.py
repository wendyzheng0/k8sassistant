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
from llama_index.core.storage.docstore.simple_docstore import SimpleDocumentStore
from llama_index.core.storage.index_store.simple_index_store import SimpleIndexStore
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


if os.path.exists('../../.env'):
    load_dotenv('../../.env')

# Default values
DEFAULT_DB_URI = 'http://localhost:19530'
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'zh-cn')
DEFAULT_MD_CACHE = "./md_cache"
DEFAULT_MILVUS_DATA = "milvus_data"
DEFAULT_START_MILVUS = False

milvus_server = None


def init_embed_model():
    # 从环境变量或配置读取设置
    hf_endpoint = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
    hf_base_url = os.getenv("HUGGINGFACE_HUB_BASE_URL", "https://hf-mirror.com")
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    device = os.getenv("EMBEDDING_DEVICE", "cpu")
    backend = os.getenv("EMBEDDING_BACKEND", "torch")  # torch, onnx, openvino
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'hf_cache')
    cache_dir = os.getenv("EMBEDDING_CACHE_DIR", cache_dir)
    
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
    
    print(f"Load embedding model")
    print(f"model_path: {model_path}")
    print(f"device: {device}")
    print(f"backend: {backend}")
    
    # 根据后端类型初始化模型
    if backend == "onnx":
        # ONNX后端配置
        model_kwargs = {}
        
        # 根据设备选择执行提供者
        if device == "cuda":
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' in providers:
                    model_kwargs["provider"] = "CUDAExecutionProvider"
                    print("Using CUDA ONNX execution provider")
                else:
                    print("CUDA execution provider not available, falling back to CPU")
                    model_kwargs["provider"] = "CPUExecutionProvider"
            except Exception as e:
                print(f"Error checking CUDA providers: {e}, falling back to CPU")
                model_kwargs["provider"] = "CPUExecutionProvider"
        else:
            # CPU模式，强制使用CPU执行提供者
            model_kwargs["provider"] = "CPUExecutionProvider"
            print("Using CPU ONNX execution provider")
        
        # 检查ONNX模型路径
        onnx_path = os.path.join(model_path, "onnx")
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
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=onnx_path,
                device=device,
                backend="onnx",
                model_kwargs=model_kwargs
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
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
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


def cache_md(original_filename, md_content, data_dir, md_cache):
    rel_path = os.path.relpath(original_filename, data_dir)
    cache_path = os.path.join(md_cache, rel_path)
    base, ext = os.path.splitext(cache_path)
    cache_path = base + ".md"
    if not os.path.exists(os.path.dirname(cache_path)):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as file:
        file.write(md_content)

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

# process single document with html2text
def process_doc_html2text(doc, data_dir, md_cache):
    h = html2text.HTML2Text()
    h.ignore_links = True
    soup = BeautifulSoup(doc.text, "html.parser")
    content = soup.find("div", class_="td-content")
    processed_content = h.handle(str(content))
    cache_md(doc.metadata['file_path'], processed_content, data_dir, md_cache)
    print(f"Processed file: {doc.metadata['file_path']}")
    # Document 的 text 属性是只读的，需新建对象并保留元数据
    processed_doc = Document(text=processed_content, metadata=doc.metadata or {})
    return processed_doc

def process(data_dir, md_cache, db_uri=DEFAULT_DB_URI):
    vector_store = init_vector_store(db_uri)

    # 使用更灵活的过滤方式，过滤掉路径中包含 _print/index.html 的文件
    exclude_files = []

    # 迭代式加载和处理
    print(f"Going to load data from {data_dir}")
    reader_iter = SimpleDirectoryReader(
        input_dir=data_dir, 
        required_exts=[".html"],
        exclude=exclude_files,
        recursive=True
    )

    md_docs = []
    processed_files = 0

    print("Starting iterative loading:")
    for docs in reader_iter.iter_data():
        # 处理每个文件的文档
        processed_docs = []
        for doc in docs:
            # 过滤掉路径中包含 _print/index.html 的文件
            if doc.metadata and 'file_path' in doc.metadata:
                file_path = doc.metadata['file_path']
                if '_print/index.html' in file_path:
                    print(f"🚫 跳过文件: {file_path} (包含 _print/index.html)")
                    continue
            
            # 在这里可以进行实时处理，比如数据清洗、分析等
            processed_doc = process_doc_html2text(doc, data_dir, md_cache)
            processed_docs.append(processed_doc)
            
        md_docs.extend(processed_docs)
        processed_files += 1
        print(f"Processed file {processed_files} documents")

    print(f"\nLoading completed, total {len(md_docs)} documents")

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 创建存储上下文，使用显式组件避免文件加载问题
    # docstore = SimpleDocumentStore()
    # index_store = SimpleIndexStore()
    
    # storage_context = StorageContext.from_defaults(
    #     docstore=docstore,
    #     index_store=index_store,
    #     vector_store=vector_store
    # )
    
    # 从文档创建向量索引
    print("\n📥 Starting to build vector index...")
    print(f"📁 Vector store: {vector_store.collection_name}")
    
    index = VectorStoreIndex.from_documents(
        md_docs,
        storage_context=storage_context,
        show_progress=True
    )

    print("✅ Vector index construction completed!")
    print(f"📖 Indexed {len(md_docs)} documents")
    return index


def main():
    """Main function to handle command line arguments and execute data processing"""
    parser = argparse.ArgumentParser(description='Process HTML documents and create vector index')
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR,
                        help=f'Path to the data directory containing HTML files (default: {DEFAULT_DATA_DIR})')
    parser.add_argument('--md-cache', type=str, default=DEFAULT_MD_CACHE,
                        help=f'Path to store converted markdown files (default: {DEFAULT_MD_CACHE})')
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
    
    # Create md_cache directory if it doesn't exist
    if not os.path.exists(args.md_cache):
        os.makedirs(args.md_cache, exist_ok=True)
        print(f"📁 Created md_cache directory: {args.md_cache}")
    
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
        print(f"📁 MD cache directory: {args.md_cache}")
        print(f"🗄️ Milvus data path: {args.milvus_data}")
        print(f"🔗 Database URI: {args.db_uri}")
        
        index = process(args.data_dir, args.md_cache, args.db_uri)
        
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
