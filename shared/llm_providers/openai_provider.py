"""
OpenAI-compatible LLM Provider
Supports OpenAI API and any compatible APIs (DeepSeek, Azure OpenAI, local LLMs, etc.)
"""

import asyncio
from typing import List, Dict, Any, Optional, AsyncIterator
import openai

from .base import BaseLLMProvider, LLMConfig, LLMResponse


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI-compatible LLM provider
    
    This provider works with:
    - OpenAI API
    - DeepSeek API
    - Azure OpenAI
    - Local LLMs with OpenAI-compatible APIs (Ollama, vLLM, etc.)
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        self.client: Optional[openai.AsyncOpenAI] = None
    
    def _load_default_config(self) -> LLMConfig:
        """Load configuration from shared settings"""
        from shared.config import get_settings
        settings = get_settings()
        
        return LLMConfig(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            model=settings.LLM_MODEL,
            max_tokens=settings.MAX_TOKENS,
            temperature=settings.TEMPERATURE,
            timeout=getattr(settings, "LLM_TIMEOUT", 180),
        )
    
    async def initialize(self) -> None:
        """Initialize the OpenAI client"""
        if self._initialized:
            return
        
        self.logger.info(f"🔄 Initializing OpenAI provider: {self.config.model}")
        
        self.client = openai.AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout
        )
        
        self._initialized = True
        self.logger.info(f"✅ OpenAI provider initialized: {self.config.base_url}")
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response using OpenAI API"""
        if not self._initialized:
            await self.initialize()
        
        temp = temperature if temperature is not None else self.config.temperature
        max_toks = max_tokens if max_tokens is not None else self.config.max_tokens
        
        try:
            self.logger.info(f"🤖 Generating response with model: {self.config.model}")
            
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temp,
                max_tokens=max_toks,
                **kwargs
            )
            
            content = response.choices[0].message.content
            
            return LLMResponse(
                content=content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                } if response.usage else None,
                finish_reason=response.choices[0].finish_reason,
                raw_response=response
            )
            
        except Exception as e:
            self.logger.error(f"❌ Generation failed: {e}")
            raise
    
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate a streaming response using OpenAI API"""
        if not self._initialized:
            await self.initialize()
        
        temp = temperature if temperature is not None else self.config.temperature
        max_toks = max_tokens if max_tokens is not None else self.config.max_tokens
        
        try:
            self.logger.info(f"🤖 Starting stream generation with model: {self.config.model}")
            
            stream = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temp,
                max_tokens=max_toks,
                stream=True,
                **kwargs
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            self.logger.error(f"❌ Stream generation failed: {e}")
            raise
    
    async def close(self) -> None:
        """Clean up resources"""
        if self.client:
            await self.client.close()
        await super().close()


class OllamaProvider(OpenAIProvider):
    """
    Ollama LLM provider
    
    Ollama provides OpenAI-compatible API, so this inherits from OpenAIProvider
    with Ollama-specific defaults.
    """
    
    def _load_default_config(self) -> LLMConfig:
        """Load Ollama-specific configuration"""
        import os
        
        return LLMConfig(
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),  # Ollama doesn't need API key
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            model=os.getenv("OLLAMA_MODEL", "qwen2:7b"),
            max_tokens=int(os.getenv("OLLAMA_MAX_TOKENS", "4096")),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
        )

