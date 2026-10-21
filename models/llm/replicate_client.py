"""
Replicate Client

Client for Replicate's model hosting platform.
Run thousands of open-source models via API.
"""

import logging
import os
from typing import Any, AsyncIterator

import replicate

from .base_llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class ReplicateClient(BaseLLMClient):
    """
    Client for Replicate platform.
    
    Supports thousands of models including:
    - meta/llama-2-70b-chat
    - meta/llama-3-70b-instruct
    - mistralai/mixtral-8x7b-instruct-v0.1
    - And many more
    """

    def __init__(
        self,
        model_name: str = "meta/llama-2-70b-chat",
        api_key: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize Replicate client.

        Args:
            model_name: Model name (owner/model-name format)
            api_key: Replicate API key (defaults to REPLICATE_API_TOKEN env var)
            **kwargs: Additional configuration
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("REPLICATE_API_TOKEN")
        
        if not self.api_key:
            raise ValueError("Replicate API token is required")
        
        # Set API token
        os.environ["REPLICATE_API_TOKEN"] = self.api_key
        
        logger.info(f"Initialized Replicate client with model: {model_name}")

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
            prompt = self._format_messages(messages)
            
            output = replicate.run(
                self.model_name,
                input={
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    "top_p": top_p,
                    **kwargs,
                }
            )
            
            # Join output chunks
            if isinstance(output, list):
                return "".join(str(chunk) for chunk in output)
            return str(output)
            
        except Exception as e:
            logger.error(f"Replicate generation error: {e}")
            raise RuntimeError(f"Replicate generation failed: {e}")

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
            prompt = self._format_messages(messages)
            
            output = await replicate.async_run(
                self.model_name,
                input={
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    "top_p": top_p,
                    **kwargs,
                }
            )
            
            # Join output chunks
            if isinstance(output, list):
                return "".join(str(chunk) for chunk in output)
            return str(output)
            
        except Exception as e:
            logger.error(f"Replicate async generation error: {e}")
            raise RuntimeError(f"Replicate async generation failed: {e}")

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
            prompt = self._format_messages(messages)
            
            async for output in replicate.stream(
                self.model_name,
                input={
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    "top_p": top_p,
                    **kwargs,
                }
            ):
                yield str(output)
                    
        except Exception as e:
            logger.error(f"Replicate streaming error: {e}")
            raise RuntimeError(f"Replicate streaming failed: {e}")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        # Replicate doesn't provide token counting, estimate
        return len(text) // 4

    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        return {
            "provider": "replicate",
            "model": self.model_name,
            "supports_streaming": True,
            "supports_function_calling": False,
            "open_source": True,
            "max_tokens": 4096,  # Varies by model
        }
