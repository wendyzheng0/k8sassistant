"""
Reranking Steps
Various reranking strategies for improving retrieval quality
"""

import os
from typing import List, Dict, Any, Optional

from app.core.logging import get_logger
from .base import BaseStep
from ..pipeline import PipelineContext


logger = get_logger(__name__)


class RRFRerankStep(BaseStep):
    """
    Reciprocal Rank Fusion (RRF) reranking step
    
    Combines results from multiple retrieval sources using RRF algorithm.
    This is a lightweight, training-free approach that works well in practice.
    """
    
    def __init__(
        self, 
        name: str = "RRFRerank",
        enabled: bool = True,
        k: int = 60
    ):
        """
        Initialize RRF reranker
        
        Args:
            name: Step name
            enabled: Whether step is enabled
            k: RRF constant (typically 60)
        """
        super().__init__(name, enabled)
        self.k = k
    
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute RRF reranking
        
        基于统一的 doc_id (格式: relative_path#chunk_index) 进行融合，
        确保 Milvus 和 Elasticsearch 返回的相同分块能够正确匹配。
        
        Args:
            context: Pipeline context with vector_results and keyword_results
            
        Returns:
            Context with merged_results populated
        """
        vector_results = context.vector_results
        keyword_results = context.keyword_results
        
        # 使用 doc_id 作为统一标识符进行融合
        doc_scores: Dict[str, Dict[str, Any]] = {}
        
        # Process vector results
        for rank, doc in enumerate(vector_results):
            # 优先使用 doc_id 进行融合
            doc_id = doc.get("doc_id") or doc.get("id")
            # print(f"doc_id: {doc_id}, file_path: {doc.get('file_path', '')}")
            if doc_id:
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        "id": doc.get("id", doc_id),
                        "doc_id": doc_id,
                        "content": doc.get("content", ""),
                        "metadata": doc.get("metadata", {}),
                        "file_path": doc.get("file_path", ""),
                        "chunk_index": doc.get("chunk_index", 0),
                        "rrf_score": 0.0,
                        "vector_rank": rank + 1,
                        "vector_score": doc.get("score", 0.0),
                        "keyword_rank": None,
                        "keyword_score": None,
                        "sources": []
                    }
                
                # RRF score: 1 / (k + rank)
                rrf_score = 1.0 / (self.k + rank + 1)
                doc_scores[doc_id]["rrf_score"] += rrf_score
                doc_scores[doc_id]["sources"].append("vector")
        
        # Process keyword results
        for rank, doc in enumerate(keyword_results):
            # 使用 doc_id 进行融合
            doc_id = doc.get("doc_id") or doc.get("id")
            # print(f"doc_id: {doc_id}, file_path: {doc.get('file_path', '')}")
            if doc_id:
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        "id": doc.get("id", doc_id),
                        "doc_id": doc_id,
                        "content": doc.get("content", ""),
                        "metadata": {},
                        "file_path": doc.get("file_path", ""),
                        "chunk_index": doc.get("chunk_index", 0),
                        "rrf_score": 0.0,
                        "vector_rank": None,
                        "vector_score": None,
                        "keyword_rank": rank + 1,
                        "keyword_score": doc.get("score", 0.0),
                        "sources": []
                    }
                else:
                    doc_scores[doc_id]["keyword_rank"] = rank + 1
                    doc_scores[doc_id]["keyword_score"] = doc.get("score", 0.0)
                
                # RRF score
                rrf_score = 1.0 / (self.k + rank + 1)
                doc_scores[doc_id]["rrf_score"] += rrf_score
                doc_scores[doc_id]["sources"].append("keyword")
                
                # Add highlights if available
                if "highlights" in doc:
                    doc_scores[doc_id]["highlights"] = doc["highlights"]
        
        # Sort by RRF score
        merged_results = list(doc_scores.values())
        merged_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        
        # 统计融合效果
        both_sources = sum(1 for r in merged_results if len(r.get("sources", [])) == 2)
        
        context.merged_results = merged_results
        self.logger.info(
            f"RRF merged {len(merged_results)} unique documents "
            f"({both_sources} matched in both sources)"
        )
        
        return context


class CrossEncoderRerankStep(BaseStep):
    """
    CrossEncoder reranking step
    
    Uses a cross-encoder model to rerank documents based on query-document relevance.
    This provides higher quality ranking than embedding similarity alone.
    """
    
    def __init__(
        self, 
        name: str = "CrossEncoderRerank",
        enabled: bool = True,
        model_name: str = "BAAI/bge-reranker-base",
        top_k: Optional[int] = None
    ):
        super().__init__(name, enabled)
        self.model_name = model_name
        self.top_k = top_k
        self._model = None
    
    def _initialize_model(self):
        """Initialize the CrossEncoder model"""
        if self._model is not None:
            return
        
        import torch
        from sentence_transformers import CrossEncoder
        from huggingface_hub import snapshot_download
        from shared.config import get_settings
        
        settings = get_settings()
        
        self.logger.info(f"🔄 Loading CrossEncoder model: {self.model_name}")
        
        # Check device
        device = settings.EMBEDDING_DEVICE
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        
        # Set up HuggingFace mirror
        if settings.HF_MIRROR_BASE_URL:
            os.environ["HF_ENDPOINT"] = settings.HF_MIRROR_BASE_URL
            os.environ["HUGGINGFACE_HUB_BASE_URL"] = settings.HF_MIRROR_BASE_URL
        
        # Download model
        if not os.path.exists(settings.EMBEDDING_CACHE_DIR):
            os.makedirs(settings.EMBEDDING_CACHE_DIR)
        
        model_path = snapshot_download(
            self.model_name,
            endpoint=settings.HF_MIRROR_BASE_URL,
            cache_dir=settings.EMBEDDING_CACHE_DIR
        )
        
        self._model = CrossEncoder(model_path)
        self.logger.info(f"✅ CrossEncoder model loaded: {model_path}")
    
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute CrossEncoder reranking
        
        Args:
            context: Pipeline context with merged_results
            
        Returns:
            Context with reranked_results and final_documents populated
        """
        # Get documents to rerank
        documents = context.merged_results or context.vector_results
        
        if not documents:
            self.logger.warning("No documents to rerank")
            context.reranked_results = []
            context.final_documents = []
            return context
        
        # Initialize model if needed
        self._initialize_model()
        
        if not self._model:
            self.logger.warning("CrossEncoder model not available, using input order")
            context.reranked_results = documents
            context.final_documents = documents[:context.top_k]
            return context
        
        # Use refined semantic query if available, otherwise use original
        if context.refined_query:
            query = context.refined_query["semantic_search_query"]
        else:
            query = context.query
        
        # Prepare query-document pairs
        pairs = []
        valid_docs = []
        for doc in documents:
            content = doc.get("content", "")
            if content:
                pairs.append([query, content])
                valid_docs.append(doc)
        
        if not pairs:
            context.reranked_results = documents
            context.final_documents = documents[:context.top_k]
            return context
        
        # Compute relevance scores
        scores = self._model.predict(pairs)
        
        # Add scores to documents
        for i, doc in enumerate(valid_docs):
            doc["rerank_score"] = float(scores[i])
        
        # Sort by rerank score
        reranked = sorted(valid_docs, key=lambda x: x.get("rerank_score", 0), reverse=True)
        
        # Determine top_k
        top_k = self.top_k or context.top_k
        
        context.reranked_results = reranked
        context.final_documents = reranked[:top_k]
        
        self.logger.info(f"CrossEncoder reranked {len(reranked)} documents, returning top {top_k}")
        
        return context

