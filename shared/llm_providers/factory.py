"""
LLM Provider Factory
Creates and manages LLM provider instances
"""

from typing import Dict, Type, Optional, List
import logging

from .base import BaseLLMProvider, LLMConfig
from .openai_provider import OpenAIProvider, OllamaProvider


logger = logging.getLogger(__name__)


# Registry of available providers
_PROVIDER_REGISTRY: Dict[str, Type[BaseLLMProvider]] = {
    "openai": OpenAIProvider,
    "deepseek": OpenAIProvider,  # DeepSeek uses OpenAI-compatible API
    "ollama": OllamaProvider,
}

# Singleton provider instances
_provider_instances: Dict[str, BaseLLMProvider] = {}


def register_provider(name: str, provider_class: Type[BaseLLMProvider]) -> None:
    """
    Register a new LLM provider
    
    Args:
        name: Provider identifier
        provider_class: Provider class that implements BaseLLMProvider
    """
    _PROVIDER_REGISTRY[name.lower()] = provider_class
    logger.info(f"✅ Registered LLM provider: {name}")


def get_available_providers() -> List[str]:
    """Get list of available provider names"""
    return list(_PROVIDER_REGISTRY.keys())


def create_llm_provider(
    provider_type: Optional[str] = None,
    config: Optional[LLMConfig] = None,
    use_singleton: bool = True
) -> BaseLLMProvider:
    """
    Create or get an LLM provider instance
    
    Args:
        provider_type: Provider type (e.g., "openai", "ollama"). 
                      If None, uses LLM_PROVIDER from settings.
        config: Optional custom configuration
        use_singleton: If True, returns cached instance for the provider type
        
    Returns:
        BaseLLMProvider instance
        
    Raises:
        ValueError: If provider type is not supported
    """
    # Get provider type from settings if not specified
    if provider_type is None:
        from shared.config import get_settings
        settings = get_settings()
        provider_type = settings.LLM_PROVIDER
    
    provider_type = provider_type.lower()
    
    # Check if provider is registered
    if provider_type not in _PROVIDER_REGISTRY:
        available = ", ".join(get_available_providers())
        raise ValueError(
            f"Unknown LLM provider: {provider_type}. "
            f"Available providers: {available}"
        )
    
    # Return singleton instance if available and requested
    if use_singleton and provider_type in _provider_instances and config is None:
        return _provider_instances[provider_type]
    
    # Create new provider instance
    provider_class = _PROVIDER_REGISTRY[provider_type]
    provider = provider_class(config)
    
    # Cache as singleton if requested
    if use_singleton and config is None:
        _provider_instances[provider_type] = provider
    
    logger.info(f"✅ Created LLM provider: {provider_type}")
    return provider


async def get_initialized_provider(
    provider_type: Optional[str] = None,
    config: Optional[LLMConfig] = None
) -> BaseLLMProvider:
    """
    Get an initialized LLM provider instance
    
    This is a convenience function that creates and initializes the provider.
    
    Args:
        provider_type: Provider type (e.g., "openai", "ollama")
        config: Optional custom configuration
        
    Returns:
        Initialized BaseLLMProvider instance
    """
    provider = create_llm_provider(provider_type, config)
    await provider.initialize()
    return provider


def clear_provider_cache() -> None:
    """Clear all cached provider instances"""
    global _provider_instances
    _provider_instances = {}
    logger.info("✅ Cleared LLM provider cache")

