"""
Google Gemini Client

Client for Google's Gemini models (Gemini Pro, Gemini Ultra, etc.)
"""

import logging
import os
from typing import Any, AsyncIterator

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from .base_llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class GeminiClient(BaseLLMClient):
    """
    Client for Google Gemini models.
    
    Supports:
    - Gemini Pro
    - Gemini Pro Vision
    - Gemini Ultra
    - Gemini 1.5 Pro
    - Gemini 1.5 Flash
    """

    def __init__(
        self,
        model_name: str = "gemini-pro",
        api_key: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize Gemini client.

        Args:
            model_name: Model name (gemini-pro, gemini-1.5-pro, etc.)
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            **kwargs: Additional configuration
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Google API key is required")
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel(model_name)
        
        logger.info(f"Initialized Gemini client with model: {model_name}")

    def _convert_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """
        Convert OpenAI-style messages to Gemini format.
        
        Args:
            messages: OpenAI-style message list
            
        Returns:
            Gemini-formatted messages
        """
        gemini_messages = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            # Map roles
            if role == "system":
                # Gemini doesn't have system role, prepend to first user message
                gemini_messages.append({
                    "role": "user",
                    "parts": [f"System: {content}"]
                })
            elif role == "assistant":
                gemini_messages.append({
                    "role": "model",
                    "parts": [content]
                })
            else:  # user
                gemini_messages.append({
                    "role": "user",
                    "parts": [content]
                })
        
        return gemini_messages

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
            generation_config = GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=top_p,
                **kwargs,
            )
            
            # Convert messages
            gemini_messages = self._convert_messages(messages)
            
            # For single message, use generate_content
            if len(gemini_messages) == 1:
                response = self.model.generate_content(
                    gemini_messages[0]["parts"][0],
                    generation_config=generation_config,
                )
            else:
                # For chat, use chat session
                chat = self.model.start_chat(history=gemini_messages[:-1])
                response = chat.send_message(
                    gemini_messages[-1]["parts"][0],
                    generation_config=generation_config,
                )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise RuntimeError(f"Gemini generation failed: {e}")

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
            generation_config = GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=top_p,
                **kwargs,
            )
            
            # Convert messages
            gemini_messages = self._convert_messages(messages)
            
            # For single message
            if len(gemini_messages) == 1:
                response = await self.model.generate_content_async(
                    gemini_messages[0]["parts"][0],
                    generation_config=generation_config,
                )
            else:
                # For chat
                chat = self.model.start_chat(history=gemini_messages[:-1])
                response = await chat.send_message_async(
                    gemini_messages[-1]["parts"][0],
                    generation_config=generation_config,
                )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini async generation error: {e}")
            raise RuntimeError(f"Gemini async generation failed: {e}")

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
            generation_config = GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=top_p,
                **kwargs,
            )
            
            # Convert messages
            gemini_messages = self._convert_messages(messages)
            
            # Stream response
            if len(gemini_messages) == 1:
                response = await self.model.generate_content_async(
                    gemini_messages[0]["parts"][0],
                    generation_config=generation_config,
                    stream=True,
                )
            else:
                chat = self.model.start_chat(history=gemini_messages[:-1])
                response = await chat.send_message_async(
                    gemini_messages[-1]["parts"][0],
                    generation_config=generation_config,
                    stream=True,
                )
            
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            raise RuntimeError(f"Gemini streaming failed: {e}")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        try:
            result = self.model.count_tokens(text)
            return result.total_tokens
        except Exception as e:
            logger.warning(f"Token counting failed, estimating: {e}")
            # Fallback: rough estimate (4 chars per token)
            return len(text) // 4

    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        # Model context windows
        context_windows = {
            "gemini-pro": 32768,
            "gemini-1.5-pro": 1048576,  # 1M tokens
            "gemini-1.5-flash": 1048576,
            "gemini-ultra": 32768,
            "gemini-pro-vision": 16384,
        }
        
        return {
            "provider": "google_gemini",
            "model": self.model_name,
            "supports_streaming": True,
            "supports_function_calling": True,
            "supports_vision": "vision" in self.model_name.lower(),
            "max_tokens": context_windows.get(self.model_name, 32768),
        }
