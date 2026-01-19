"""
Base Step for RAG Pipeline
Defines the interface that all pipeline steps must implement
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from enum import Enum
import time

from app.core.logging import get_logger

if TYPE_CHECKING:
    from ..pipeline import PipelineContext


logger = get_logger(__name__)


class StepStatus(Enum):
    """Step execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    """Result of a pipeline step execution"""
    status: StepStatus
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseStep(ABC):
    """
    Abstract base class for RAG pipeline steps
    
    Each step in the pipeline should:
    1. Receive a PipelineContext
    2. Perform its specific operation
    3. Update the context with results
    4. Return the updated context
    """
    
    def __init__(self, name: Optional[str] = None, enabled: bool = True):
        """
        Initialize the step
        
        Args:
            name: Optional custom name for the step
            enabled: Whether the step is enabled
        """
        self.name = name or self.__class__.__name__
        self.enabled = enabled
        self.logger = get_logger(self.name)
    
    @abstractmethod
    async def execute(self, context) -> Any:
        """
        Execute the step
        
        Args:
            context: Current pipeline context (PipelineContext)
            
        Returns:
            Updated pipeline context
        """
        pass
    
    async def run(self, context) -> Any:
        """
        Run the step with timing and error handling
        
        Args:
            context: Current pipeline context
            
        Returns:
            Updated pipeline context
        """
        if not self.enabled:
            self.logger.info(f"⏭️ Step {self.name} is disabled, skipping")
            context.step_results[self.name] = StepResult(
                status=StepStatus.SKIPPED,
                metadata={"reason": "disabled"}
            )
            return context
        
        start_time = time.time()
        self.logger.info(f"🚀 Starting step: {self.name}")
        
        try:
            context = await self.execute(context)
            execution_time = time.time() - start_time
            
            context.step_results[self.name] = StepResult(
                status=StepStatus.SUCCESS,
                execution_time=execution_time
            )
            
            self.logger.info(f"✅ Step {self.name} completed in {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Step {self.name} failed: {e}")
            
            context.step_results[self.name] = StepResult(
                status=StepStatus.FAILED,
                error=str(e),
                execution_time=execution_time
            )
            
            # Re-raise to let pipeline handle it
            raise
        
        return context
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, enabled={self.enabled})"

