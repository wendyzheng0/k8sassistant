"""
HTML to Markdown Dataloader
Uses shared modules for embedding service

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
    "html2md_dataloader.py is deprecated. Use 'from data_processing.processors import PipelineRunner' instead.",
    DeprecationWarning,
    stacklevel=2
)

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
from bs4 import BeautifulSoup
from pymilvus import MilvusClient
from milvus_lite.server import Server
import data_cleaner

# Add project root to path for shared module imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


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
    """
    Initialize embedding model using shared embedding module
    The shared module handles model downloading automatically
    """
    from shared.embeddings import create_embedding_service
    
    print("🔄 Initializing embedding model using shared module...")
    
    # Create embedding service using shared module
    # It automatically handles model downloading, caching, and initialization
    embedding_service = create_embedding_service(use_singleton=True)
    
    # Set the underlying model as llama-index's embed_model
    Settings.embed_model = embedding_service.model
    
    print(f"✅ Embedding model initialized: {embedding_service.get_model_info()}")


def init_llm():
    """Initialize LLM using OpenAI-compatible interface"""
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    model_name = os.getenv("LLM_MODEL", "qwen-plus")
    
    if base_url and api_key:
        Settings.llm = OpenAI(model=model_name, base_url=base_url, api_key=api_key)
        print(f"✅ LLM initialized: {model_name}")
    else:
        print("⚠️ LLM not configured (missing LLM_BASE_URL or LLM_API_KEY)")
        Settings.llm = None


def start_milvus(milvus_data_path=DEFAULT_MILVUS_DATA, port=19530):
    global milvus_server
    
    try:
        print("🚀 Starting Milvus server...")
        print(f"📁 Milvus data path: {milvus_data_path}")
        milvus_server = Server(db_file=milvus_data_path, address=f'localhost:{port}')
        milvus_server.start()
        print("✅ Milvus server started successfully")
        
        # Wait for server to fully start
        import time
        print("⏳ Waiting for server to fully start...")
        time.sleep(3)
        
        # Test connection
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
    """Initialize vector store with dynamic dimension detection"""
    # Dynamically infer vector dimension
    try:
        inferred_dim = len(Settings.embed_model.get_text_embedding("__dim_probe__"))
    except Exception:
        inferred_dim = 512  # BGE-small-zh-v1.5 default

    vector_store = MilvusVectorStore(
        uri=db_uri,
        dim=inferred_dim,
        collection_name=os.environ['COLLECTION_NAME'],
        overwrite=True,
        consistency_level="Strong"
    )

    print("✅ MilvusVectorStore initialized successfully!")
    print(f"📊 Collection name: {vector_store.collection_name}")
    print(f"🔗 Connection URI: {db_uri}")
    print(f"📏 Vector dimension: {inferred_dim}")
    return vector_store


def process_doc_html2text(doc, data_dir, md_cache):
    """Process single document with html2text"""
    h = html2text.HTML2Text()
    h.ignore_links = True
    soup = BeautifulSoup(doc.text, "html.parser")
    content = soup.find("div", class_="td-content")
    processed_content = h.handle(str(content))
    cache_md(doc.metadata['file_path'], processed_content, data_dir, md_cache)
    print(f"Processed file: {doc.metadata['file_path']}")
    processed_doc = Document(text=processed_content, metadata=doc.metadata or {})
    return processed_doc


def process(data_dir, md_cache, db_uri=DEFAULT_DB_URI):
    """Main processing function"""
    vector_store = init_vector_store(db_uri)

    exclude_files = []

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
        processed_docs = []
        for doc in docs:
            if doc.metadata and 'file_path' in doc.metadata:
                file_path = doc.metadata['file_path']
                if '_print/index.html' in file_path:
                    print(f"🚫 Skip file: {file_path} (includes _print/index.html)")
                    continue
            
            processed_doc = process_doc_html2text(doc, data_dir, md_cache)
            processed_docs.append(processed_doc)
            
        md_docs.extend(processed_docs)
        processed_files += 1
        print(f"Processed file {processed_files} documents")

    print(f"\nLoading completed, total {len(md_docs)} documents")

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
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
    """Main function"""
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
    
    # Create md_cache directory if needed
    if not os.path.exists(args.md_cache):
        os.makedirs(args.md_cache, exist_ok=True)
        print(f"📁 Created md_cache directory: {args.md_cache}")
    
    try:
        print("🔧 Initializing embedding model (using shared module)...")
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
        
        # Cleanup
        if index is not None:
            if hasattr(index, 'storage_context'):
                if hasattr(index.storage_context, 'vector_store'):
                    vector_store = index.storage_context.vector_store
                    if hasattr(vector_store, 'client'):
                        vector_store.client.close()
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
