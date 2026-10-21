"""
Mistral AI Client

Client for Mistral AI models (Mistral, Mixtral, etc.)
"""

import logging
import os
from typing import Any, AsyncIterator

from mistralai.async_client import MistralAsyncClient
from mistralai.client import MistralClient as MistralSyncClient
from mistralai.models.chat_completion import ChatMessage

from .base_llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class MistralClient(BaseLLMClient):
    """
    Client for Mistral AI models.
    
    Supports:
    - Mistral 7B
    - Mistral Small
    - Mistral Medium
    - Mistral Large
    - Mixtral 8x7B
    - Mixtral 8x22B
    """

    def __init__(
        self,
        model_name: str = "mistral-large-latest",
        api_key: str | None = None,
        endpoint: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize Mistral AI client.

        Args:
            model_name: Model name (mistral-small, mistral-large, mixtral-8x7b, etc.)
            api_key: Mistral API key (defaults to MISTRAL_API_KEY env var)
            endpoint: Custom endpoint (for self-hosted)
            **kwargs: Additional configuration
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.endpoint = endpoint
        
        if not self.api_key:
            raise ValueError("Mistral API key is required")
        
        # Initialize clients
        client_kwargs = {"api_key": self.api_key}
        if endpoint:
            client_kwargs["endpoint"] = endpoint
            
        self.client = MistralSyncClient(**client_kwargs)
        self.async_client = MistralAsyncClient(**client_kwargs)
        
        logger.info(f"Initialized Mistral client with model: {model_name}")

    def _convert_messages(self, messages: list[dict[str, str]]) -> list[ChatMessage]:
        """
        Convert OpenAI-style messages to Mistral format.
        
        Args:
            messages: OpenAI-style message list
            
        Returns:
            Mistral ChatMessage list
        """
        mistral_messages = []
        
        for msg in messages:
            mistral_messages.append(
                ChatMessage(
                    role=msg["role"],
                    content=msg["content"]
                )
            )
        
        return mistral_messages

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
            mistral_messages = self._convert_messages(messages)
            
            response = self.client.chat(
                model=self.model_name,
                messages=mistral_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                **kwargs,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Mistral generation error: {e}")
            raise RuntimeError(f"Mistral generation failed: {e}")

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
            mistral_messages = self._convert_messages(messages)
            
            response = await self.async_client.chat(
                model=self.model_name,
                messages=mistral_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                **kwargs,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Mistral async generation error: {e}")
            raise RuntimeError(f"Mistral async generation failed: {e}")

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
            mistral_messages = self._convert_messages(messages)
            
            stream = self.async_client.chat_stream(
                model=self.model_name,
                messages=mistral_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                **kwargs,
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Mistral streaming error: {e}")
            raise RuntimeError(f"Mistral streaming failed: {e}")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        # Mistral doesn't provide a token counting API
        # Use rough estimate (4 chars per token)
        return len(text) // 4

    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        # Model context windows
        context_windows = {
            "mistral-tiny": 32000,
            "mistral-small": 32000,
            "mistral-small-latest": 32000,
            "mistral-medium": 32000,
            "mistral-medium-latest": 32000,
            "mistral-large": 32000,
            "mistral-large-latest": 32000,
            "open-mistral-7b": 32000,
            "open-mixtral-8x7b": 32000,
            "open-mixtral-8x22b": 64000,
            "mistral-embed": 8192,
        }
        
        return {
            "provider": "mistral",
            "model": self.model_name,
            "supports_streaming": True,
            "supports_function_calling": True,
            "max_tokens": context_windows.get(self.model_name, 32000),
        }
