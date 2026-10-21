"""
Cohere Client

Client for Cohere models (Command, Command R, Command R+)
"""

import logging
import os
from typing import Any, AsyncIterator

import cohere
from cohere.types import StreamedChatResponse

from .base_llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class CohereClient(BaseLLMClient):
    """
    Client for Cohere models.
    
    Supports:
    - Command
    - Command R
    - Command R+
    - Command Light
    """

    def __init__(
        self,
        model_name: str = "command-r-plus",
        api_key: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize Cohere client.

        Args:
            model_name: Model name (command, command-r, command-r-plus, etc.)
            api_key: Cohere API key (defaults to COHERE_API_KEY env var)
            **kwargs: Additional configuration
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Cohere API key is required")
        
        # Initialize client
        self.client = cohere.Client(api_key=self.api_key)
        self.async_client = cohere.AsyncClient(api_key=self.api_key)
        
        logger.info(f"Initialized Cohere client with model: {model_name}")

    def _convert_messages(self, messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
        """
        Convert OpenAI-style messages to Cohere format.
        
        Args:
            messages: OpenAI-style message list
            
        Returns:
            Tuple of (preamble, chat_history)
        """
        preamble = ""
        chat_history = []
        
        for msg in messages[:-1]:  # All but last message
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                preamble += content + "\n\n"
            elif role == "user":
                chat_history.append({
                    "role": "USER",
                    "message": content
                })
            elif role == "assistant":
                chat_history.append({
                    "role": "CHATBOT",
                    "message": content
                })
        
        # Last message should be user message
        last_message = messages[-1]["content"]
        
        return preamble.strip(), chat_history, last_message

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
            preamble, chat_history, message = self._convert_messages(messages)
            
            response = self.client.chat(
                model=self.model_name,
                message=message,
                chat_history=chat_history if chat_history else None,
                preamble=preamble if preamble else None,
                temperature=temperature,
                max_tokens=max_tokens,
                p=top_p,
                **kwargs,
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Cohere generation error: {e}")
            raise RuntimeError(f"Cohere generation failed: {e}")

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
            preamble, chat_history, message = self._convert_messages(messages)
            
            response = await self.async_client.chat(
                model=self.model_name,
                message=message,
                chat_history=chat_history if chat_history else None,
                preamble=preamble if preamble else None,
                temperature=temperature,
                max_tokens=max_tokens,
                p=top_p,
                **kwargs,
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Cohere async generation error: {e}")
            raise RuntimeError(f"Cohere async generation failed: {e}")

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
            preamble, chat_history, message = self._convert_messages(messages)
            
            stream = self.async_client.chat_stream(
                model=self.model_name,
                message=message,
                chat_history=chat_history if chat_history else None,
                preamble=preamble if preamble else None,
                temperature=temperature,
                max_tokens=max_tokens,
                p=top_p,
                **kwargs,
            )
            
            async for event in stream:
                if event.event_type == "text-generation":
                    yield event.text
                    
        except Exception as e:
            logger.error(f"Cohere streaming error: {e}")
            raise RuntimeError(f"Cohere streaming failed: {e}")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        try:
            response = self.client.tokenize(text=text, model=self.model_name)
            return len(response.tokens)
        except Exception as e:
            logger.warning(f"Token counting failed, estimating: {e}")
            # Fallback: rough estimate (4 chars per token)
            return len(text) // 4

    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        # Model context windows
        context_windows = {
            "command": 4096,
            "command-light": 4096,
            "command-r": 128000,
            "command-r-plus": 128000,
        }
        
        return {
            "provider": "cohere",
            "model": self.model_name,
            "supports_streaming": True,
            "supports_function_calling": True,
            "supports_rag": True,
            "max_tokens": context_windows.get(self.model_name, 4096),
        }
