"""
功能：
- 使用 LlamaParse 将文件/目录解析为文档
- 构建摄取流水线（文本切分、标题与关键词提取）
- 初始化嵌入模型与 LLM（基于环境变量配置）
- 在 Milvus 中创建向量索引
"""

import os
from typing import List


def create_parser():
    """创建并返回 LlamaParse 解析器（输出为 markdown）"""
    from llama_parse import LlamaParse

    # ensure_env_var("LLAMA_CLOUD_API_KEY")

    parser = LlamaParse(result_type="markdown")
    return parser


def load_documents(
    parser,
    source_path: str,
) -> List:
    """
    使用解析器从文件或目录加载文档
    若提供目录，则递归解析目录下的文件；若提供文件路径，则仅解析该文件
    """
    documents: List = []

    if os.path.isdir(source_path):
        for root, _dirs, files in os.walk(source_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    docs = parser.load_data(file_path)
                    documents.extend(docs)
                except Exception as exc:  # noqa: BLE001
                    print(f"Error while parsing '{file_path}': {exc}")
    else:
        documents = parser.load_data(source_path)

    return documents


def setup_models():
    """初始化嵌入模型与 LLM，并写入全局 Settings"""
    # Lazy imports
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from langchain_openai import ChatOpenAI
    from llama_index.llms.langchain import LangChainLLM
    from llama_index.core import Settings

    embed_model = HuggingFaceEmbedding(
        model_name="C:/Users/15040/.cache/modelscope/hub/models/BAAI/bge-small-en-v1___5"
    )

    volc_api_key = os.getenv("VOLCENGINE_API_KEY")
    volc_base_url = os.getenv("VOLCENGINE_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")

    lc_llm = ChatOpenAI(
        base_url=volc_base_url,
        api_key=volc_api_key,
        model="deepseek-v3-250324",
    )
    llm = LangChainLLM(lc_llm)

    Settings.llm = llm
    Settings.embed_model = embed_model

    return Settings


def build_ingestion_pipeline():
    """构建文档摄取流水线：切分、标题提取、关键词提取"""
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.extractors import (
        TitleExtractor,
        KeywordExtractor,
    )

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=512, chunk_overlap=20),
            TitleExtractor(),
            KeywordExtractor(keywords=3),
        ],
    )
    return pipeline


def run_ingestion(documents) -> List:
    """运行摄取流水线并返回节点列表"""
    pipeline = build_ingestion_pipeline()
    nodes = pipeline.run(documents=documents)
    return nodes


def build_index(nodes):
    """在 Milvus 中创建/覆盖集合并构建向量索引"""
    from llama_index.vector_stores.milvus import MilvusVectorStore
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.core import Settings

    uri = os.getenv("MILVUS_URI", "http://localhost:19530")
    vector_dim = len(Settings.embed_model.get_query_embedding("ping"))

    # 创建MilvusVectorStore实例
    vector_store = MilvusVectorStore(
        uri=uri,
        dim=vector_dim,
        enable_sparse=True,
        collection_name=os.getenv("MILVUS_COLLECTION", "kubernetes_rag"),
        overwrite=True,
        consistency_level="Strong",
    )

    # 创建索引
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True,
    )
    return index


def main() -> int:
    """主流程：解析 -> 摄取 -> 构建索引"""

    source_path = os.getenv("RAG_SOURCE","D:/gh-ai/AI53/项目/原始数据/docs")

    print(f"Using source: {source_path}")

    parser = create_parser()
    _settings = setup_models() 

    documents = load_documents(parser, source_path)
    print(f"Loaded documents: {len(documents)}")

    nodes = run_ingestion(documents)
    print(f"Generated nodes: {len(nodes)}")

    if nodes:
        first = nodes[0]
        doc_title = first.metadata.get("document_title")
        keywords = first.metadata.get("excerpt_keywords")
        if doc_title:
            print(f"Document title: {doc_title}")
        if keywords:
            print(f"Keywords: {keywords}")

    _index = build_index(nodes)
    print("Vector index created in Milvus.")

    return 0


if __name__ == "__main__":
    os.environ['LLAMA_CLOUD_API_KEY'] = 'llx-Cblzs1We600zfCqUh7ZRtFlKSyQopyOu6Jc76L4lwakUm8NC'
    os.environ['VOLCENGINE_API_KEY'] = 'c3fb7cdb-8e3c-422b-b55e-ba40ce5d0832'
    os.environ['VOLCENGINE_BASE_URL'] = 'https://ark.cn-beijing.volces.com/api/v3'
    os.environ['MILVUS_URI'] = 'http://localhost:19530'
    os.environ['MILVUS_COLLECTION'] = 'kubernetes_rag'
    # os.environ['RAG_SOURCE'] = 'D:/gh-ai/AI53/项目/原始数据/docs/tutorials/configuration'
    os.environ['RAG_SOURCE'] = 'D:/gh-ai/AI53/项目/原始数据/docs'
    
    raise SystemExit(main())


