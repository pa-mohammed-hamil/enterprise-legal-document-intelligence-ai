"""
Together AI Client

Client for Together AI's inference platform with access to 50+ open-source models.
"""

import logging
import os
from typing import Any, AsyncIterator

import together

from .base_llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class TogetherClient(BaseLLMClient):
    """
    Client for Together AI platform.
    
    Supports 50+ models including:
    - Llama 2, Llama 3
    - Mixtral
    - Qwen
    - Yi
    - DeepSeek
    - And many more open-source models
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        api_key: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize Together AI client.

        Args:
            model_name: Model name (full path, e.g., "meta-llama/Llama-2-70b-chat-hf")
            api_key: Together API key (defaults to TOGETHER_API_KEY env var)
            **kwargs: Additional configuration
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        
        if not self.api_key:
            raise ValueError("Together API key is required")
        
        # Configure API key
        together.api_key = self.api_key
        
        logger.info(f"Initialized Together client with model: {model_name}")

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
            response = together.Complete.create(
                model=self.model_name,
                prompt=self._format_messages(messages),
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                **kwargs,
            )
            
            return response["output"]["choices"][0]["text"]
            
        except Exception as e:
            logger.error(f"Together generation error: {e}")
            raise RuntimeError(f"Together generation failed: {e}")

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
            # Together doesn't have native async, use chat endpoint
            response = together.Complete.create(
                model=self.model_name,
                prompt=self._format_messages(messages),
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                **kwargs,
            )
            
            return response["output"]["choices"][0]["text"]
            
        except Exception as e:
            logger.error(f"Together async generation error: {e}")
            raise RuntimeError(f"Together async generation failed: {e}")

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
            stream = together.Complete.create_streaming(
                model=self.model_name,
                prompt=self._format_messages(messages),
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                **kwargs,
            )
            
            for chunk in stream:
                if "choices" in chunk["output"] and len(chunk["output"]["choices"]) > 0:
                    text = chunk["output"]["choices"][0].get("text", "")
                    if text:
                        yield text
                    
        except Exception as e:
            logger.error(f"Together streaming error: {e}")
            raise RuntimeError(f"Together streaming failed: {e}")

    def _format_messages(self, messages: list[dict[str, str]]) -> str:
        """
        Format messages into a prompt string.
        
        Args:
            messages: OpenAI-style message list
            
        Returns:
            Formatted prompt string
        """
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
        
        prompt += "Assistant: "
        return prompt

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        # Together doesn't provide token counting, estimate
        return len(text) // 4

    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        return {
            "provider": "together",
            "model": self.model_name,
            "supports_streaming": True,
            "supports_function_calling": False,
            "open_source": True,
            "max_tokens": 8192,  # Varies by model
        }
