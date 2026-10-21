"""
LLM Factory

Factory for creating LLM clients based on provider configuration.
"""

import os
from typing import Optional

import yaml

from backend.core.exceptions import ModelConfigurationError
from backend.models.llm.anthropic_client import AnthropicClient
from backend.models.llm.azure_openai_client import AzureOpenAIClient
from backend.models.llm.base_llm_client import BaseLLMClient
from backend.models.llm.cohere_client import CohereClient
from backend.models.llm.gemini_client import GeminiClient
from backend.models.llm.groq_client import GroqClient
from backend.models.llm.mistral_client import MistralClient
from backend.models.llm.openai_client import OpenAIClient
from backend.models.llm.perplexity_client import PerplexityClient
from backend.models.llm.replicate_client import ReplicateClient
from backend.models.llm.together_client import TogetherClient


class LLMFactory:
    """
    Factory for creating LLM clients.
    
    Supports:
    - OpenAI (GPT-4, GPT-3.5, GPT-4o, o1, o3)
    - Azure OpenAI
    - Anthropic Claude (Claude 3 Opus, Sonnet, Haiku, 3.5 Sonnet)
    - Google Gemini (Gemini Pro, 1.5 Pro, 1.5 Flash)
    - Cohere (Command R, Command R+)
    - Mistral AI (Mistral Large, Mixtral)
    - Groq (Ultra-fast Llama, Mixtral inference)
    - Together AI (50+ open-source models)
    - Perplexity AI (with web search capabilities)
    - Replicate (1000+ open-source models)
    - Local models (via LM Studio, Ollama, vLLM)
    """

    @staticmethod
    def create_client(
        provider: Optional[str] = None,
        model: Optional[str] = None,
        config_path: str = "ml/configs/models.yaml",
        **kwargs,
    ) -> BaseLLMClient:
        """
        Create an LLM client based on provider.

        Args:
            provider: Provider name (openai, azure_openai, anthropic, local)
                     If None, uses default from config
            model: Model name. If None, uses default from config
            config_path: Path to models configuration file
            **kwargs: Additional provider-specific arguments

        Returns:
            Configured LLM client instance

        Raises:
            ModelConfigurationError: If provider is invalid or configuration fails
        """
        # Load configuration
        config = LLMFactory._load_config(config_path)

        # Determine provider
        if provider is None:
            provider = config.get("legal_llm", {}).get("default_provider", "openai")

        # Determine model
        if model is None:
            provider_config = config.get("llm_providers", {}).get(provider, {})
            model = provider_config.get("model")

        # Create client based on provider
        if provider == "openai":
            return LLMFactory._create_openai_client(config, model, **kwargs)
        elif provider == "azure_openai":
            return LLMFactory._create_azure_openai_client(config, model, **kwargs)
        elif provider == "anthropic":
            return LLMFactory._create_anthropic_client(config, model, **kwargs)
        elif provider == "gemini" or provider == "google_gemini":
            return LLMFactory._create_gemini_client(config, model, **kwargs)
        elif provider == "cohere":
            return LLMFactory._create_cohere_client(config, model, **kwargs)
        elif provider == "mistral":
            return LLMFactory._create_mistral_client(config, model, **kwargs)
        elif provider == "groq":
            return LLMFactory._create_groq_client(config, model, **kwargs)
        elif provider == "together":
            return LLMFactory._create_together_client(config, model, **kwargs)
        elif provider == "perplexity":
            return LLMFactory._create_perplexity_client(config, model, **kwargs)
        elif provider == "replicate":
            return LLMFactory._create_replicate_client(config, model, **kwargs)
        elif provider == "local":
            raise NotImplementedError("Local model support coming soon - use Ollama, LM Studio, or vLLM")
        else:
            raise ModelConfigurationError(f"Unknown provider: {provider}")

    @staticmethod
    def _load_config(config_path: str) -> dict:
        """Load models configuration from YAML file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Return default config if file not found
            return {
                "legal_llm": {
                    "default_provider": "openai",
                    "default_model": "gpt-4",
                },
                "llm_providers": {
                    "openai": {
                        "enabled": True,
                        "model": "gpt-4",
                        "timeout_seconds": 90,
                        "max_retries": 3,
                    }
                },
            }

    @staticmethod
    def _create_openai_client(
        config: dict, model: Optional[str] = None, **kwargs
    ) -> OpenAIClient:
        """Create OpenAI client from configuration."""
        provider_config = config.get("llm_providers", {}).get("openai", {})

        if not provider_config.get("enabled", True):
            raise ModelConfigurationError("OpenAI provider is disabled in configuration")

        # Get configuration with fallbacks
        model = model or provider_config.get("model", "gpt-4")
        timeout = kwargs.get("timeout", provider_config.get("timeout_seconds", 90))
        max_retries = kwargs.get("max_retries", provider_config.get("max_retries", 3))
        api_key = kwargs.get("api_key", os.getenv("OPENAI_API_KEY"))

        return OpenAIClient(
            model=model,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    @staticmethod
    def _create_azure_openai_client(
        config: dict, model: Optional[str] = None, **kwargs
    ) -> AzureOpenAIClient:
        """Create Azure OpenAI client from configuration."""
        provider_config = config.get("llm_providers", {}).get("azure_openai", {})

        if not provider_config.get("enabled", False):
            raise ModelConfigurationError(
                "Azure OpenAI provider is disabled in configuration"
            )

        # Get configuration with fallbacks
        deployment_name = model or provider_config.get("model", "gpt-4")
        api_version = kwargs.get("api_version", provider_config.get("api_version", "2024-02-15-preview"))
        timeout = kwargs.get("timeout", provider_config.get("timeout_seconds", 90))
        max_retries = kwargs.get("max_retries", provider_config.get("max_retries", 3))
        api_key = kwargs.get("api_key", os.getenv("AZURE_OPENAI_API_KEY"))
        api_base = kwargs.get("api_base", os.getenv("AZURE_OPENAI_ENDPOINT"))

        return AzureOpenAIClient(
            deployment_name=deployment_name,
            api_version=api_version,
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries,
        )

    @staticmethod
    def get_available_providers(config_path: str = "ml/configs/models.yaml") -> list[str]:
        """
        Get list of available (enabled) providers.

        Args:
            config_path: Path to models configuration file

        Returns:
            List of enabled provider names
        """
        config = LLMFactory._load_config(config_path)
        providers = config.get("llm_providers", {})

        return [
            name
            for name, provider_config in providers.items()
            if provider_config.get("enabled", False)
        ]

    @staticmethod
    def get_available_models(
        provider: str, config_path: str = "ml/configs/models.yaml"
    ) -> list[str]:
        """
        Get list of available models for a provider.

        Args:
            provider: Provider name
            config_path: Path to models configuration file

        Returns:
            List of model names
        """
        config = LLMFactory._load_config(config_path)
        provider_config = config.get("llm_providers", {}).get(provider, {})

        return provider_config.get("available_models", [provider_config.get("model", "")])

    @staticmethod
    def _create_anthropic_client(
        config: dict, model: Optional[str] = None, **kwargs
    ) -> AnthropicClient:
        """Create Anthropic Claude client from configuration."""
        provider_config = config.get("llm_providers", {}).get("anthropic", {})

        if not provider_config.get("enabled", False):
            raise ModelConfigurationError("Anthropic provider is disabled in configuration")

        model = model or provider_config.get("model", "claude-3-5-sonnet-20241022")
        api_key = kwargs.get("api_key", os.getenv("ANTHROPIC_API_KEY"))

        return AnthropicClient(model_name=model, api_key=api_key, **kwargs)

    @staticmethod
    def _create_gemini_client(
        config: dict, model: Optional[str] = None, **kwargs
    ) -> GeminiClient:
        """Create Google Gemini client from configuration."""
        provider_config = config.get("llm_providers", {}).get("gemini", {})
        if not provider_config:
            provider_config = config.get("llm_providers", {}).get("google_gemini", {})

        if not provider_config.get("enabled", False):
            raise ModelConfigurationError("Gemini provider is disabled in configuration")

        model = model or provider_config.get("model", "gemini-1.5-pro")
        api_key = kwargs.get("api_key", os.getenv("GOOGLE_API_KEY"))

        return GeminiClient(model_name=model, api_key=api_key, **kwargs)

    @staticmethod
    def _create_cohere_client(
        config: dict, model: Optional[str] = None, **kwargs
    ) -> CohereClient:
        """Create Cohere client from configuration."""
        provider_config = config.get("llm_providers", {}).get("cohere", {})

        if not provider_config.get("enabled", False):
            raise ModelConfigurationError("Cohere provider is disabled in configuration")

        model = model or provider_config.get("model", "command-r-plus")
        api_key = kwargs.get("api_key", os.getenv("COHERE_API_KEY"))

        return CohereClient(model_name=model, api_key=api_key, **kwargs)

    @staticmethod
    def _create_mistral_client(
        config: dict, model: Optional[str] = None, **kwargs
    ) -> MistralClient:
        """Create Mistral AI client from configuration."""
        provider_config = config.get("llm_providers", {}).get("mistral", {})

        if not provider_config.get("enabled", False):
            raise ModelConfigurationError("Mistral provider is disabled in configuration")

        model = model or provider_config.get("model", "mistral-large-latest")
        api_key = kwargs.get("api_key", os.getenv("MISTRAL_API_KEY"))
        endpoint = kwargs.get("endpoint", provider_config.get("endpoint"))

        return MistralClient(model_name=model, api_key=api_key, endpoint=endpoint, **kwargs)

    @staticmethod
    def _create_groq_client(
        config: dict, model: Optional[str] = None, **kwargs
    ) -> GroqClient:
        """Create Groq client from configuration."""
        provider_config = config.get("llm_providers", {}).get("groq", {})

        if not provider_config.get("enabled", False):
            raise ModelConfigurationError("Groq provider is disabled in configuration")

        model = model or provider_config.get("model", "llama-3.1-70b-versatile")
        api_key = kwargs.get("api_key", os.getenv("GROQ_API_KEY"))

        return GroqClient(model_name=model, api_key=api_key, **kwargs)

    @staticmethod
    def _create_together_client(
        config: dict, model: Optional[str] = None, **kwargs
    ) -> TogetherClient:
        """Create Together AI client from configuration."""
        provider_config = config.get("llm_providers", {}).get("together", {})

        if not provider_config.get("enabled", False):
            raise ModelConfigurationError("Together provider is disabled in configuration")

        model = model or provider_config.get("model", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")
        api_key = kwargs.get("api_key", os.getenv("TOGETHER_API_KEY"))

        return TogetherClient(model_name=model, api_key=api_key, **kwargs)

    @staticmethod
    def _create_perplexity_client(
        config: dict, model: Optional[str] = None, **kwargs
    ) -> PerplexityClient:
        """Create Perplexity AI client from configuration."""
        provider_config = config.get("llm_providers", {}).get("perplexity", {})

        if not provider_config.get("enabled", False):
            raise ModelConfigurationError("Perplexity provider is disabled in configuration")

        model = model or provider_config.get("model", "llama-3.1-sonar-large-128k-online")
        api_key = kwargs.get("api_key", os.getenv("PERPLEXITY_API_KEY"))

        return PerplexityClient(model_name=model, api_key=api_key, **kwargs)

    @staticmethod
    def _create_replicate_client(
        config: dict, model: Optional[str] = None, **kwargs
    ) -> ReplicateClient:
        """Create Replicate client from configuration."""
        provider_config = config.get("llm_providers", {}).get("replicate", {})

        if not provider_config.get("enabled", False):
            raise ModelConfigurationError("Replicate provider is disabled in configuration")

        model = model or provider_config.get("model", "meta/llama-2-70b-chat")
        api_key = kwargs.get("api_key", os.getenv("REPLICATE_API_TOKEN"))

        return ReplicateClient(model_name=model, api_key=api_key, **kwargs)
