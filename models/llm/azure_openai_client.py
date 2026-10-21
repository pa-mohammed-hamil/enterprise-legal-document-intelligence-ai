"""
Azure OpenAI Client

Provides Azure OpenAI integration for the Legal Document Intelligence AI system.
"""

import os
from typing import Any, AsyncIterator, Optional

from openai import AsyncAzureOpenAI, AzureOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from backend.core.exceptions import ModelConfigurationError, ModelInferenceError
from backend.models.llm.base_llm_client import BaseLLMClient


class AzureOpenAIClient(BaseLLMClient):
    """
    Azure OpenAI client for legal document intelligence.
    
    Supports:
    - GPT-4, GPT-4 Turbo, GPT-3.5 Turbo models
    - Streaming and non-streaming responses
    - Function calling
    - Azure-specific authentication and endpoints
    """

    def __init__(
        self,
        deployment_name: Optional[str] = None,
        api_version: str = "2024-02-15-preview",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: int = 90,
        max_retries: int = 3,
    ):
        """
        Initialize Azure OpenAI client.

        Args:
            deployment_name: Azure deployment name (model)
            api_version: Azure OpenAI API version
            api_key: Azure OpenAI API key (or from env)
            api_base: Azure OpenAI endpoint URL (or from env)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        super().__init__()

        # Get configuration from environment if not provided
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.api_base = api_base or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment_name = deployment_name or os.getenv(
            "AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"
        )
        self.api_version = api_version
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_key:
            raise ModelConfigurationError(
                "Azure OpenAI API key not found. Set AZURE_OPENAI_API_KEY environment variable."
            )

        if not self.api_base:
            raise ModelConfigurationError(
                "Azure OpenAI endpoint not found. Set AZURE_OPENAI_ENDPOINT environment variable."
            )

        # Initialize clients
        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.api_base,
            api_version=self.api_version,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

        self.async_client = AsyncAzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.api_base,
            api_version=self.api_version,
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
        Generate a completion using Azure OpenAI.

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
            response: ChatCompletion = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                **kwargs,
            )

            if not response.choices:
                raise ModelInferenceError("No completion choices returned from Azure OpenAI")

            content = response.choices[0].message.content
            if content is None:
                raise ModelInferenceError("Completion content is None")

            return content

        except Exception as e:
            raise ModelInferenceError(f"Azure OpenAI generation failed: {str(e)}") from e

    async def generate_async(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        **kwargs: Any,
    ) -> str:
        """
        Async generate a completion using Azure OpenAI.

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
                model=self.deployment_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                **kwargs,
            )

            if not response.choices:
                raise ModelInferenceError("No completion choices returned from Azure OpenAI")

            content = response.choices[0].message.content
            if content is None:
                raise ModelInferenceError("Completion content is None")

            return content

        except Exception as e:
            raise ModelInferenceError(f"Azure OpenAI async generation failed: {str(e)}") from e

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream a completion using Azure OpenAI.

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
                model=self.deployment_name,
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
            raise ModelInferenceError(f"Azure OpenAI streaming failed: {str(e)}") from e

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        # Rough estimation: ~4 characters per token
        # For production, use tiktoken library
        return len(text) // 4

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the Azure OpenAI model.

        Returns:
            Dictionary with model information
        """
        return {
            "provider": "azure_openai",
            "deployment_name": self.deployment_name,
            "api_version": self.api_version,
            "endpoint": self.api_base,
            "supports_streaming": True,
            "supports_function_calling": True,
            "max_tokens": 128000 if "gpt-4" in self.deployment_name else 16385,
        }
