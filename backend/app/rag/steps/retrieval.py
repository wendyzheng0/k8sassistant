"""
Retrieval Steps
Vector and keyword-based document retrieval
"""

import asyncio
from typing import List, Dict, Any, Optional

from app.core.logging import get_logger
from .base import BaseStep
from ..pipeline import PipelineContext


logger = get_logger(__name__)


class VectorRetrievalStep(BaseStep):
    """
    Vector-based retrieval step
    
    Retrieves documents using vector similarity search from Milvus.
    """
    
    def __init__(
        self, 
        name: str = "VectorRetrieval",
        enabled: bool = True,
        top_k_multiplier: int = 5
    ):
        super().__init__(name, enabled)
        self.top_k_multiplier = top_k_multiplier
        self._milvus_service = None
        self._embedding_service = None
    
    async def _get_services(self):
        """Get or create services"""
        if self._embedding_service is None:
            from shared.embeddings import create_embedding_service
            self._embedding_service = create_embedding_service()
        
        if self._milvus_service is None:
            from app.services.milvus_service import MilvusService
            self._milvus_service = MilvusService()
            await self._milvus_service.initialize()
        
        return self._embedding_service, self._milvus_service
    
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute vector retrieval
        
        Args:
            context: Pipeline context
            
        Returns:
            Context with vector_results populated
        """
        embedding_service, milvus_service = await self._get_services()
        
        # Use refined semantic query if available, otherwise use original
        if context.refined_query:
            query = context.refined_query["semantic_search_query"]
        else:
            query = context.query
        
        # Generate query embedding
        query_embedding = embedding_service.encode(query)[0]
        
        # Search in Milvus
        results = await milvus_service.search_similar(
            query_embedding=query_embedding,
            top_k=context.top_k * self.top_k_multiplier
        )
        
        # Format results
        vector_results = []
        for doc in results:
            vector_results.append({
                "id": doc.get("id"),
                "content": doc.get("content"),
                "metadata": doc.get("metadata", {}),
                "score": doc.get("score", 0.0),
                "source": "milvus",
                "file_path": doc.get("file_path", ""),
                "distance": doc.get("distance", 0.0)
            })
        
        context.vector_results = vector_results
        self.logger.info(f"Retrieved {len(vector_results)} documents from vector search")
        
        return context


class KeywordRetrievalStep(BaseStep):
    """
    Keyword-based retrieval step
    
    Retrieves documents using keyword search from Elasticsearch.
    """
    
    def __init__(
        self, 
        name: str = "KeywordRetrieval",
        enabled: bool = True,
        top_k_multiplier: int = 5
    ):
        super().__init__(name, enabled)
        self.top_k_multiplier = top_k_multiplier
        self._es_service = None
    
    async def _get_es_service(self):
        """Get or create Elasticsearch service"""
        if self._es_service is None:
            from app.services.complex_retrieval_service import ElasticsearchService
            self._es_service = ElasticsearchService()
            await self._es_service.initialize()
        return self._es_service
    
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute keyword retrieval
        
        Args:
            context: Pipeline context
            
        Returns:
            Context with keyword_results populated
        """
        es_service = await self._get_es_service()
        
        # Use refined keyword list if available, otherwise use original query
        if context.refined_query:
            keywords = context.refined_query["keyword_list"]
            # Join keywords into a single query string for ES
            query = " ".join(keywords)
        else:
            query = context.query
        
        # Search in Elasticsearch
        results = await es_service.search(
            query=query,
            top_k=context.top_k * self.top_k_multiplier
        )
        
        context.keyword_results = results
        self.logger.info(f"Retrieved {len(results)} documents from keyword search")
        
        return context


class HybridRetrievalStep(BaseStep):
    """
    Hybrid retrieval step
    
    Combines vector and keyword retrieval in parallel for better coverage.
    """
    
    def __init__(
        self, 
        name: str = "HybridRetrieval",
        enabled: bool = True,
        top_k_multiplier: int = 5
    ):
        super().__init__(name, enabled)
        self.vector_step = VectorRetrievalStep(
            name="HybridRetrieval_Vector",
            top_k_multiplier=top_k_multiplier
        )
        self.keyword_step = KeywordRetrievalStep(
            name="HybridRetrieval_Keyword", 
            top_k_multiplier=top_k_multiplier
        )
    
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute hybrid retrieval (vector + keyword in parallel)
        
        Args:
            context: Pipeline context
            
        Returns:
            Context with both vector_results and keyword_results populated
        """
        # Run both retrievals in parallel
        vector_task = asyncio.create_task(self.vector_step.execute(context))
        keyword_task = asyncio.create_task(self.keyword_step.execute(context))
        
        # Wait for both to complete
        await asyncio.gather(vector_task, keyword_task, return_exceptions=True)
        
        # Log results
        self.logger.info(
            f"Hybrid retrieval: {len(context.vector_results)} vector, "
            f"{len(context.keyword_results)} keyword results"
        )
        
        return context

