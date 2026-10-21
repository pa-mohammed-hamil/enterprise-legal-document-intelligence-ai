"""
Perplexity AI Client

Client for Perplexity AI models with real-time web search capabilities.
"""

import logging
import os
from typing import Any, AsyncIterator

import openai

from .base_llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class PerplexityClient(BaseLLMClient):
    """
    Client for Perplexity AI models.
    
    Supports:
    - pplx-7b-online (with web search)
    - pplx-70b-online (with web search)
    - llama-3.1-sonar-small-128k-online
    - llama-3.1-sonar-large-128k-online
    - llama-3.1-sonar-huge-128k-online
    """

    def __init__(
        self,
        model_name: str = "llama-3.1-sonar-large-128k-online",
        api_key: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize Perplexity client.

        Args:
            model_name: Model name (pplx-70b-online, etc.)
            api_key: Perplexity API key (defaults to PERPLEXITY_API_KEY env var)
            **kwargs: Additional configuration
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        
        if not self.api_key:
            raise ValueError("Perplexity API key is required")
        
        # Perplexity uses OpenAI-compatible API
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.perplexity.ai"
        )
        self.async_client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.perplexity.ai"
        )
        
        logger.info(f"Initialized Perplexity client with model: {model_name}")

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        **kwargs: Any,
    ) -> str:
        """Generate completion synchronously."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                **kwargs,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Perplexity generation error: {e}")
            raise RuntimeError(f"Perplexity generation failed: {e}")

    async def generate_async(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        **kwargs: Any,
    ) -> str:
        """Generate completion asynchronously."""
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                **kwargs,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Perplexity async generation error: {e}")
            raise RuntimeError(f"Perplexity async generation failed: {e}")

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream completion asynchronously."""
        try:
            stream = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=True,
                **kwargs,
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Perplexity streaming error: {e}")
            raise RuntimeError(f"Perplexity streaming failed: {e}")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        # Perplexity doesn't provide token counting, estimate
        return len(text) // 4

    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        return {
            "provider": "perplexity",
            "model": self.model_name,
            "supports_streaming": True,
            "supports_function_calling": False,
            "supports_web_search": "online" in self.model_name,
            "max_tokens": 127072 if "128k" in self.model_name else 4096,
        }
