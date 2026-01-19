"""
@deprecated: 此模块已弃用，请使用新的流水线架构

新的使用方式:
    from data_processing.processors import PipelineRunner
    
    runner = PipelineRunner()
    result = await runner.run(data_dir="./data/zh-cn", storage_backend="elasticsearch")

或使用命令行:
    python -m data_processing.processors.cli --data-dir ./data/zh-cn --backend elasticsearch
"""

import warnings
warnings.warn(
    "elastic_dataloader.py is deprecated. Use 'from data_processing.processors import PipelineRunner' instead.",
    DeprecationWarning,
    stacklevel=2
)

import os
import sys
import argparse
import traceback
import json
from typing import Dict, Generator, Iterable, List, Tuple

from dotenv import load_dotenv
from bs4 import BeautifulSoup
from elasticsearch import Elasticsearch
from elasticsearch import helpers as es_helpers

# Local utilities
import data_cleaner


# Load .env if present (project root two levels up from this file)
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env")
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)


# Defaults
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "zh-cn"
)
DEFAULT_ES_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
DEFAULT_ES_INDEX = os.getenv("ELASTICSEARCH_INDEX", "k8s-docs")
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
DEFAULT_BATCH_BYTES = int(os.getenv("ES_BATCH_BYTES", str(2 * 1024 * 1024)))  # ~2MB target


def preprocess_html_document_safe(html_text: str, metadata: Dict) -> Tuple[str, Dict]:
    """
    Preprocess HTML: clean HTML and extract readable text without separating code blocks.

    Returns a tuple of (processed_text, cleaned_metadata).
    """
    try:
        if not isinstance(html_text, str):
            html_text = str(html_text)

        cleaned_html = data_cleaner.clean_html_content(html_text)
        processed_soup = BeautifulSoup(cleaned_html, "html.parser")

        content_div = processed_soup.find("div", class_="td-content")
        if content_div:
            processed_text = content_div.get_text(separator="\n", strip=True)
        else:
            processed_text = processed_soup.get_text(separator="\n", strip=True)

        processed_text = data_cleaner.clean_text(processed_text)
        cleaned_metadata = data_cleaner.clean_metadata(metadata or {})

        return processed_text, cleaned_metadata
    except Exception as e:
        print(f"Exception occurs when preprocessing HTML document: {e}")
        cleaned_doc = data_cleaner.clean_document(type("Doc", (), {"text": html_text, "metadata": metadata})())
        return cleaned_doc.text, cleaned_doc.metadata


def iter_html_files(root_dir: str) -> Generator[Tuple[str, Dict], None, None]:
    """
    Yield (html_content, metadata) for .html files under root_dir, recursively.
    Skips paths containing '_print'.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.lower().endswith(".html"):
                continue
            file_path = os.path.join(dirpath, fname)
            if "_print" in file_path.replace("\\", "/"):
                print(f"🚫 Skip file: {file_path} (including _print)")
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    html = f.read()
                metadata = {"file_path": file_path}
                yield html, metadata
            except Exception as e:
                print(f"⚠️ Failed to read {file_path}: {e}")


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into chunks with simple paragraph/newline awareness and overlap.
    """
    if not text:
        return []

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buffer: List[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal buffer, current_len
        if not buffer:
            return
        combined = "\n\n".join(buffer).strip()
        if combined:
            chunks.append(combined[:chunk_size])
        if overlap > 0 and combined:
            # create overlap seed for next chunk
            seed = combined[-overlap:]
            buffer = [seed]
            current_len = len(seed)
        else:
            buffer = []
            current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        to_add = ("\n\n" if buffer else "") + para
        if current_len + len(to_add) > chunk_size and current_len > 0:
            flush()
            to_add = para
        buffer.append(to_add if not buffer else para)
        current_len = len("\n\n".join(buffer))
        if current_len >= chunk_size:
            flush()

    flush()
    return chunks


def ensure_index(es: Elasticsearch, index_name: str) -> None:
    """
    Create index with a minimal mapping if it doesn't exist.
    """
    if es.indices.exists(index=index_name):
        return
    body = {
        "settings": {
            "number_of_shards": int(os.getenv("ES_NUM_SHARDS", "1")),
            "number_of_replicas": int(os.getenv("ES_NUM_REPLICAS", "0")),
            "analysis": {
                "analyzer": {
                    "default": {"type": "standard"}
                }
            }
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "file_path": {"type": "keyword"},
                "chunk_index": {"type": "integer"}
            }
        }
    }
    es.indices.create(index=index_name, body=body)
    print(f"✅ Created index '{index_name}'")


def connect_es(host: str) -> Elasticsearch:
    username = os.getenv("ELASTICSEARCH_USER", "elastic")
    password = os.getenv("ELASTICSEARCH_PASSWORD", "password")
    ca_certs = os.getenv("ELASTICSEARCH_CA_CERTS")
    
    # Determine if we should use SSL based on the host URL
    use_ssl = host.startswith("https://")
    
    # Build connection parameters
    connection_params = {
        "hosts": [host],
        "basic_auth": (username, password),
    }
    
    if use_ssl and ca_certs and os.path.exists(ca_certs):
        connection_params["ca_certs"] = ca_certs
        print(f"🔒 Using SSL with certificate: {ca_certs}")
    elif use_ssl:
        print("⚠️ SSL enabled but certificate not found, using verify_certs=False")
        connection_params["verify_certs"] = False
        connection_params["ssl_show_warn"] = False
    
    es = Elasticsearch(**connection_params)
    
    try:
        info = es.info()
        print(f"🔗 Connected to Elasticsearch: {info.get('version', {}).get('number', 'unknown')}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("💡 Troubleshooting tips:")
        print("   1. Check if Elasticsearch is running")
        print("   2. Verify the host URL (http://localhost:9200 or https://localhost:9200)")
        print("   3. Check username/password credentials")
        print("   4. For HTTPS, ensure certificate path is correct")
        raise RuntimeError(f"Cannot connect to Elasticsearch at {host}: {e}")
    
    return es


def generate_actions(index: str, docs: Iterable[Dict]) -> Iterable[Dict]:
    for d in docs:
        yield {
            "_op_type": "index",
            "_index": index,
            "_source": d,
        }


def bulk_index_chunks(
    es: Elasticsearch,
    index_name: str,
    items: Iterable[Dict],
    batch_bytes_target: int = DEFAULT_BATCH_BYTES,
) -> Tuple[int, int]:
    """
    Stream items to Elasticsearch using a byte-size-aware batcher.
    Returns (success_count, error_count).
    """
    batch: List[Dict] = []
    current_bytes = 0
    success = 0
    errors = 0

    def flush_batch():
        nonlocal batch, current_bytes, success, errors
        if not batch:
            return
        resp = es_helpers.bulk(es, generate_actions(index_name, batch), refresh=False)
        ok, detail = resp
        # elasticsearch-py returns a tuple (successes, errors list) in some versions,
        # and (ok, items) in others via helpers.bulk. We'll compute conservatively.
        if isinstance(ok, int):
            success += ok
        else:
            # Fallback: count items with status < 300
            succeeded = 0
            failed = 0
            for item in detail:
                action = next(iter(item.values()))
                status = action.get("status", 500)
                if 200 <= status < 300:
                    succeeded += 1
                else:
                    failed += 1
            success += succeeded
            errors += failed
        batch = []
        current_bytes = 0

    for it in items:
        as_bytes = json.dumps(it, ensure_ascii=False).encode("utf-8")
        size = len(as_bytes)
        if current_bytes + size > batch_bytes_target and current_bytes > 0:
            flush_batch()
        batch.append(it)
        current_bytes += size
    flush_batch()
    return success, errors


def process_and_index(
    data_dir: str,
    es_host: str,
    index_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Tuple[int, int, int]:
    """
    Process HTML files, chunk text, and index chunks to Elasticsearch.
    Returns (files_processed, chunks_indexed, errors).
    """
    es = connect_es(es_host)
    ensure_index(es, index_name)

    files = 0
    all_items: List[Dict] = []

    print(f"📂 Loading HTML from: {data_dir}")
    for html, metadata in iter_html_files(data_dir):
        files += 1
        processed_text, cleaned_meta = preprocess_html_document_safe(html, metadata)
        chunks = chunk_text(processed_text, chunk_size, chunk_overlap)
        for i, chunk in enumerate(chunks):
            doc = {
                "text": chunk,
                "file_path": cleaned_meta.get("file_path"),
                "chunk_index": i,
            }
            all_items.append(doc)

    print(f"📦 Prepared {len(all_items)} chunks from {files} files. Indexing to '{index_name}'...")
    success, err = bulk_index_chunks(es, index_name, all_items)
    print(f"✅ Indexed: {success}, ❌ Errors: {err}")
    return files, success, err


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse HTML docs and index chunks into Elasticsearch")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help=f"Directory with HTML files (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--es-host", type=str, default=DEFAULT_ES_HOST, help=f"Elasticsearch host (default: {DEFAULT_ES_HOST})")
    parser.add_argument("--index", type=str, default=DEFAULT_ES_INDEX, help=f"Elasticsearch index name (default: {DEFAULT_ES_INDEX})")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Chunk overlap in characters")

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"🔴 Error: Data directory does not exist: {args.data_dir}")
        return 1

    try:
        files, success, err = process_and_index(
            data_dir=args.data_dir,
            es_host=args.es_host,
            index_name=args.index,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        print(f"Done. Files: {files}, Indexed chunks: {success}, Errors: {err}")
        return 0 if err == 0 else 2
    except Exception as e:
        print(f"🔴 Error occurred during execution: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())


