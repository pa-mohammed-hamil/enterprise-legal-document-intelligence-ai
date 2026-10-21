"""
Anthropic Claude Client

Client for Anthropic's Claude models (Claude 3 family)
"""

import logging
import os
from typing import Any, AsyncIterator

import anthropic

from .base_llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class AnthropicClient(BaseLLMClient):
    """
    Client for Anthropic Claude models.
    
    Supports:
    - Claude 3 Opus
    - Claude 3 Sonnet
    - Claude 3 Haiku
    - Claude 3.5 Sonnet
    - Claude Instant
    """

    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20241022",
        api_key: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize Anthropic client.

        Args:
            model_name: Model name (claude-3-opus, claude-3-sonnet, etc.)
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            **kwargs: Additional configuration
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        
        # Initialize client
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.async_client = anthropic.AsyncAnthropic(api_key=self.api_key)
        
        logger.info(f"Initialized Anthropic client with model: {model_name}")

    def _convert_messages(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, list[dict[str, str]]]:
        """
        Convert OpenAI-style messages to Anthropic format.
        
        Args:
            messages: OpenAI-style message list
            
        Returns:
            Tuple of (system_prompt, messages)
        """
        system_prompt = ""
        anthropic_messages = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                system_prompt += content + "\n\n"
            else:
                anthropic_messages.append({
                    "role": role,
                    "content": content
                })
        
        return system_prompt.strip(), anthropic_messages

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
            system_prompt, anthropic_messages = self._convert_messages(messages)
            
            response = self.client.messages.create(
                model=self.model_name,
                system=system_prompt if system_prompt else None,
                messages=anthropic_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                **kwargs,
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            raise RuntimeError(f"Anthropic generation failed: {e}")

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
            system_prompt, anthropic_messages = self._convert_messages(messages)
            
            response = await self.async_client.messages.create(
                model=self.model_name,
                system=system_prompt if system_prompt else None,
                messages=anthropic_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                **kwargs,
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic async generation error: {e}")
            raise RuntimeError(f"Anthropic async generation failed: {e}")

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
            system_prompt, anthropic_messages = self._convert_messages(messages)
            
            async with self.async_client.messages.stream(
                model=self.model_name,
                system=system_prompt if system_prompt else None,
                messages=anthropic_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                **kwargs,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise RuntimeError(f"Anthropic streaming failed: {e}")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        try:
            # Use the count_tokens API
            response = self.client.count_tokens(text)
            return response
        except Exception as e:
            logger.warning(f"Token counting failed, estimating: {e}")
            # Fallback: rough estimate (4 chars per token)
            return len(text) // 4

    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        # Model context windows
        context_windows = {
            "claude-3-opus-20240229": 200000,
            "claude-3-sonnet-20240229": 200000,
            "claude-3-haiku-20240307": 200000,
            "claude-3-5-sonnet-20241022": 200000,
            "claude-2.1": 200000,
            "claude-2.0": 100000,
            "claude-instant-1.2": 100000,
        }
        
        return {
            "provider": "anthropic",
            "model": self.model_name,
            "supports_streaming": True,
            "supports_function_calling": True,
            "supports_vision": "3" in self.model_name,  # Claude 3 family supports vision
            "max_tokens": context_windows.get(self.model_name, 200000),
        }
