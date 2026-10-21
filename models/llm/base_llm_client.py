"""
Base LLM Client

Abstract base class for all LLM provider clients.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.
    
    All LLM provider implementations (OpenAI, Azure, Anthropic, Local)
    should inherit from this class and implement its methods.
    """

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        **kwargs: Any,
    ) -> str:
        """
        Generate a completion synchronously.

        Args:
            messages: List of chat messages
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            **kwargs: Provider-specific arguments

        Returns:
            Generated text response

        Raises:
            ModelInferenceError: If generation fails
        """
        pass

    @abstractmethod
    async def generate_async(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        **kwargs: Any,
    ) -> str:
        """
        Generate a completion asynchronously.

        Args:
            messages: List of chat messages
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            **kwargs: Provider-specific arguments

        Returns:
            Generated text response

        Raises:
            ModelInferenceError: If generation fails
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream a completion asynchronously.

        Args:
            messages: List of chat messages
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            **kwargs: Provider-specific arguments

        Yields:
            Token chunks as they are generated

        Raises:
            ModelInferenceError: If streaming fails
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count or estimate tokens in text.

        Args:
            text: Input text

        Returns:
            Token count
        """
        pass

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dictionary with model metadata including:
            - provider: Provider name (openai, azure_openai, anthropic, local)
            - model: Model name/identifier
            - supports_streaming: Whether streaming is supported
            - supports_function_calling: Whether function calling is supported
            - max_tokens: Maximum context window size
        """
        pass
