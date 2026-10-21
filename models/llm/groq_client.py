"""
Groq Client

Client for Groq's ultra-fast LLM inference platform.
Supports Llama, Mixtral, and Gemma models.
"""

import logging
import os
from typing import Any, AsyncIterator

from groq import AsyncGroq, Groq

from .base_llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class GroqClient(BaseLLMClient):
    """
    Client for Groq inference platform.
    
    Supports:
    - Llama 3 (8B, 70B)
    - Llama 3.1 (8B, 70B, 405B)
    - Mixtral 8x7B
    - Gemma 7B
    """

    def __init__(
        self,
        model_name: str = "llama-3.1-70b-versatile",
        api_key: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize Groq client.

        Args:
            model_name: Model name (llama-3.1-70b, mixtral-8x7b, etc.)
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
            **kwargs: Additional configuration
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("Groq API key is required")
        
        # Initialize clients
        self.client = Groq(api_key=self.api_key)
        self.async_client = AsyncGroq(api_key=self.api_key)
        
        logger.info(f"Initialized Groq client with model: {model_name}")

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
            logger.error(f"Groq generation error: {e}")
            raise RuntimeError(f"Groq generation failed: {e}")

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
            logger.error(f"Groq async generation error: {e}")
            raise RuntimeError(f"Groq async generation failed: {e}")

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
            logger.error(f"Groq streaming error: {e}")
            raise RuntimeError(f"Groq streaming failed: {e}")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        # Groq doesn't provide token counting, estimate
        return len(text) // 4

    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        # Model context windows
        context_windows = {
            "llama-3.1-8b-instant": 131072,
            "llama-3.1-70b-versatile": 131072,
            "llama-3.1-405b-reasoning": 131072,
            "llama3-8b-8192": 8192,
            "llama3-70b-8192": 8192,
            "mixtral-8x7b-32768": 32768,
            "gemma-7b-it": 8192,
            "gemma2-9b-it": 8192,
        }
        
        return {
            "provider": "groq",
            "model": self.model_name,
            "supports_streaming": True,
            "supports_function_calling": True,
            "ultra_fast": True,  # Groq's key selling point
            "max_tokens": context_windows.get(self.model_name, 8192),
        }
