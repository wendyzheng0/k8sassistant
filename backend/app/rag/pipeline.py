"""
RAG Pipeline Orchestrator
Manages the execution of RAG workflow steps
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import logging
import time


if TYPE_CHECKING:
    from .steps.base import BaseStep, StepResult


logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """
    Context object passed through the pipeline
    
    Contains all data needed by pipeline steps and accumulates results
    as it passes through each step.
    """
    # Input
    query: str
    conversation_id: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    
    # Query processing
    refined_query: Optional[str] = None
    
    # Retrieval results
    vector_results: List[Dict[str, Any]] = field(default_factory=list)
    keyword_results: List[Dict[str, Any]] = field(default_factory=list)
    merged_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Reranking results
    reranked_results: List[Dict[str, Any]] = field(default_factory=list)
    final_documents: List[Dict[str, Any]] = field(default_factory=list)
    
    # Generation results
    generated_response: Optional[str] = None
    
    # Pipeline metadata
    step_results: Dict[str, "StepResult"] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Configuration
    top_k: int = 10
    use_reranking: bool = True
    temperature: float = 0.7
    max_tokens: int = 4096


class RAGPipeline:
    """
    RAG Pipeline Orchestrator
    
    Manages the execution of a sequence of RAG steps.
    Steps can be added, removed, or reordered to customize the RAG workflow.
    
    Example:
        pipeline = (RAGPipeline()
            .add_step(QueryRewriteStep())
            .add_step(HybridRetrievalStep())
            .add_step(RRFRerankStep())
            .add_step(CrossEncoderRerankStep())
            .add_step(GenerationStep()))
        
        context = PipelineContext(query="How to create a Pod?")
        result = await pipeline.run(context)
    """
    
    def __init__(self, name: str = "RAGPipeline"):
        """
        Initialize the pipeline
        
        Args:
            name: Name of the pipeline
        """
        self.name = name
        self.steps: List["BaseStep"] = []
        self.logger = logging.getLogger(self.name)
        self._initialized = False
    
    def add_step(self, step: "BaseStep") -> "RAGPipeline":
        """
        Add a step to the pipeline
        
        Args:
            step: Step to add
            
        Returns:
            Self for method chaining
        """
        self.steps.append(step)
        self.logger.info(f"➕ Added step: {step.name}")
        return self
    
    def insert_step(self, index: int, step: "BaseStep") -> "RAGPipeline":
        """
        Insert a step at a specific position
        
        Args:
            index: Position to insert at
            step: Step to insert
            
        Returns:
            Self for method chaining
        """
        self.steps.insert(index, step)
        self.logger.info(f"➕ Inserted step at {index}: {step.name}")
        return self
    
    def remove_step(self, step_name: str) -> "RAGPipeline":
        """
        Remove a step by name
        
        Args:
            step_name: Name of the step to remove
            
        Returns:
            Self for method chaining
        """
        self.steps = [s for s in self.steps if s.name != step_name]
        self.logger.info(f"➖ Removed step: {step_name}")
        return self
    
    def get_step(self, step_name: str) -> Optional["BaseStep"]:
        """Get a step by name"""
        for step in self.steps:
            if step.name == step_name:
                return step
        return None
    
    def enable_step(self, step_name: str) -> "RAGPipeline":
        """Enable a step by name"""
        step = self.get_step(step_name)
        if step:
            step.enabled = True
        return self
    
    def disable_step(self, step_name: str) -> "RAGPipeline":
        """Disable a step by name"""
        step = self.get_step(step_name)
        if step:
            step.enabled = False
        return self
    
    async def run(
        self, 
        context: PipelineContext,
        stop_on_error: bool = True
    ) -> PipelineContext:
        """
        Execute all pipeline steps
        
        Args:
            context: Initial pipeline context
            stop_on_error: Whether to stop on first error
            
        Returns:
            Final pipeline context with all results
        """
        start_time = time.time()
        self.logger.info(f"🔄 Starting pipeline: {self.name}")
        self.logger.info(f"📝 Query: {context.query[:100]}...")
        
        context.metadata["pipeline_name"] = self.name
        context.metadata["start_time"] = start_time
        context.metadata["steps_count"] = len(self.steps)
        
        for i, step in enumerate(self.steps):
            try:
                self.logger.info(f"📍 Step {i+1}/{len(self.steps)}: {step.name}")
                context = await step.run(context)
                
            except Exception as e:
                self.logger.error(f"❌ Pipeline failed at step {step.name}: {e}")
                context.metadata["failed_step"] = step.name
                context.metadata["error"] = str(e)
                
                if stop_on_error:
                    break
        
        total_time = time.time() - start_time
        context.metadata["total_time"] = total_time
        
        self.logger.info(f"✅ Pipeline completed in {total_time:.2f}s")
        return context
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline configuration"""
        return {
            "name": self.name,
            "steps": [
                {
                    "name": step.name,
                    "class": step.__class__.__name__,
                    "enabled": step.enabled
                }
                for step in self.steps
            ],
            "steps_count": len(self.steps)
        }
    
    def __repr__(self) -> str:
        steps_str = " -> ".join(s.name for s in self.steps)
        return f"RAGPipeline({steps_str})"


def create_default_pipeline() -> RAGPipeline:
    """
    Create a default RAG pipeline with standard steps
    
    Returns:
        Configured RAGPipeline instance
    """
    from .steps import (
        QueryRewriteStep,
        HybridRetrievalStep,
        RRFRerankStep,
        CrossEncoderRerankStep,
        GenerationStep
    )
    
    pipeline = (RAGPipeline("DefaultRAGPipeline")
        .add_step(QueryRewriteStep())
        .add_step(HybridRetrievalStep())
        .add_step(RRFRerankStep())
        .add_step(CrossEncoderRerankStep())
        .add_step(GenerationStep()))
    
    return pipeline

