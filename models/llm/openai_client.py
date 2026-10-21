"""
OpenAI Client

Provides OpenAI integration for the Legal Document Intelligence AI system.
"""

import os
from typing import Any, AsyncIterator, Optional

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from backend.core.exceptions import ModelConfigurationError, ModelInferenceError
from backend.models.llm.base_llm_client import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    """
    OpenAI client for legal document intelligence.
    
    Supports:
    - GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, GPT-5.6
    - Streaming and non-streaming responses
    - Function calling
    - Token counting and usage tracking
    """

    SUPPORTED_MODELS = [
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4-turbo-preview",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "gpt-5.6",
    ]

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        timeout: int = 90,
        max_retries: int = 3,
    ):
        """
        Initialize OpenAI client.

        Args:
            model: Model name (e.g., 'gpt-4', 'gpt-3.5-turbo')
            api_key: OpenAI API key (or from env)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        super().__init__()

        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_key:
            raise ModelConfigurationError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
            )

        if model not in self.SUPPORTED_MODELS:
            raise ModelConfigurationError(
                f"Model '{model}' not supported. Supported models: {self.SUPPORTED_MODELS}"
            )

        # Initialize clients
        self.client = OpenAI(
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        **kwargs: Any,
    ) -> str:
        """
        Generate a completion using OpenAI.

        Args:
            messages: List of chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            **kwargs: Additional arguments (e.g., functions, function_call)

        Returns:
            Generated text response

        Raises:
            ModelInferenceError: If generation fails
        """
        try:
            response: ChatCompletion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                **kwargs,
            )

            if not response.choices:
                raise ModelInferenceError("No completion choices returned from OpenAI")

            content = response.choices[0].message.content
            if content is None:
                # Handle function calling case
                if response.choices[0].message.function_call:
                    return str(response.choices[0].message.function_call)
                raise ModelInferenceError("Completion content is None")

            return content

        except Exception as e:
            raise ModelInferenceError(f"OpenAI generation failed: {str(e)}") from e

    async def generate_async(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        **kwargs: Any,
    ) -> str:
        """
        Async generate a completion using OpenAI.

        Args:
            messages: List of chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            **kwargs: Additional arguments

        Returns:
            Generated text response

        Raises:
            ModelInferenceError: If generation fails
        """
        try:
            response: ChatCompletion = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                **kwargs,
            )

            if not response.choices:
                raise ModelInferenceError("No completion choices returned from OpenAI")

            content = response.choices[0].message.content
            if content is None:
                # Handle function calling case
                if response.choices[0].message.function_call:
                    return str(response.choices[0].message.function_call)
                raise ModelInferenceError("Completion content is None")

            return content

        except Exception as e:
            raise ModelInferenceError(f"OpenAI async generation failed: {str(e)}") from e

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream a completion using OpenAI.

        Args:
            messages: List of chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            **kwargs: Additional arguments

        Yields:
            Token chunks as they are generated

        Raises:
            ModelInferenceError: If streaming fails
        """
        try:
            stream = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=True,
                **kwargs,
            )

            async for chunk in stream:
                chunk: ChatCompletionChunk
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            raise ModelInferenceError(f"OpenAI streaming failed: {str(e)}") from e

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Input text

        Returns:
            Estimated token count
        
        Note:
            For production use, integrate tiktoken library for accurate counting
        """
        # Rough estimation: ~4 characters per token
        return len(text) // 4

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the OpenAI model.

        Returns:
            Dictionary with model information
        """
        # Model context windows
        context_windows = {
            "gpt-4": 8192,
            "gpt-4-turbo": 128000,
            "gpt-4-turbo-preview": 128000,
            "gpt-3.5-turbo": 16385,
            "gpt-3.5-turbo-16k": 16385,
            "gpt-5.6": 200000,
        }

        return {
            "provider": "openai",
            "model": self.model,
            "supports_streaming": True,
            "supports_function_calling": True,
            "max_tokens": context_windows.get(self.model, 8192),
            "supports_vision": "gpt-4" in self.model and "vision" in self.model.lower(),
        }

    def set_model(self, model: str) -> None:
        """
        Switch to a different OpenAI model.

        Args:
            model: New model name

        Raises:
            ModelConfigurationError: If model is not supported
        """
        if model not in self.SUPPORTED_MODELS:
            raise ModelConfigurationError(
                f"Model '{model}' not supported. Supported models: {self.SUPPORTED_MODELS}"
            )
        self.model = model
