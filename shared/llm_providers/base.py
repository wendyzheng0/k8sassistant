"""
Base LLM Provider abstraction
Defines the interface that all LLM providers must implement
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, AsyncIterator
import logging


logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM providers"""
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 60
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Standardized LLM response"""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    raw_response: Optional[Any] = None


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers
    
    All LLM providers (OpenAI, Anthropic, Ollama, etc.) should implement this interface.
    This allows for easy switching between providers without changing business logic.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize the LLM provider
        
        Args:
            config: LLM configuration. If None, will load from environment/settings.
        """
        self.config = config or self._load_default_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialized = False
    
    @abstractmethod
    def _load_default_config(self) -> LLMConfig:
        """Load default configuration from environment or settings"""
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider (e.g., create client)"""
        pass
    
    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from the LLM
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse with the generated content
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response from the LLM
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Additional provider-specific parameters
            
        Yields:
            String chunks of the generated response
        """
        pass
    
    async def test_connection(self) -> bool:
        """
        Test the connection to the LLM provider
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            test_messages = [{"role": "user", "content": "Hello"}]
            response = await self.generate(test_messages, max_tokens=10)
            self.logger.info(f"✅ LLM connection test successful: {self.config.model}")
            return True
        except Exception as e:
            self.logger.error(f"❌ LLM connection test failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration"""
        return {
            "provider": self.__class__.__name__,
            "model": self.config.model,
            "base_url": self.config.base_url,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
    
    async def close(self) -> None:
        """Clean up resources"""
        self._initialized = False
    
    async def __aenter__(self):
        """Async context manager entry"""
        if not self._initialized:
            await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

