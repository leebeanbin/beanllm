"""
API Key Validation Service

Validates API keys by using beanllm's existing provider infrastructure.
Leverages the health_check() method from each provider.
"""

import logging
import os
from typing import List, Optional

from schemas.database import ApiKeyValidationResult

logger = logging.getLogger(__name__)


class KeyValidator:
    """
    Service for validating API keys against their respective providers.

    Uses beanllm's existing provider infrastructure where possible.

    Usage:
        validator = KeyValidator()
        result = await validator.validate("openai", "sk-1234...")
    """

    def __init__(self):
        self._timeout = 30.0  # seconds

    async def validate(self, provider: str, api_key: str) -> ApiKeyValidationResult:
        """
        Validate an API key by using beanllm's provider infrastructure.

        Args:
            provider: Provider name (openai, anthropic, etc.)
            api_key: The API key to validate

        Returns:
            Validation result with available models if successful.
        """
        try:
            # Use beanllm's existing providers when possible
            if provider in [
                "openai",
                "anthropic",
                "google",
                "gemini",
                "deepseek",
                "perplexity",
                "ollama",
            ]:
                return await self._validate_with_beanllm_provider(provider, api_key)
            else:
                # For other providers, use custom validation
                validator_method = getattr(self, f"_validate_{provider}", None)
                if validator_method:
                    return await validator_method(api_key)
                else:
                    # Basic validation - just check if key is non-empty
                    return ApiKeyValidationResult(
                        provider=provider,
                        is_valid=bool(api_key and len(api_key) > 0),
                        error=None if api_key else "API key is empty",
                    )
        except Exception as e:
            logger.error(f"Validation error for {provider}: {e}")
            return ApiKeyValidationResult(
                provider=provider,
                is_valid=False,
                error=str(e),
            )

    async def _validate_with_beanllm_provider(
        self, provider: str, api_key: str
    ) -> ApiKeyValidationResult:
        """
        Validate using beanllm's existing provider infrastructure.

        Temporarily sets the environment variable, creates a provider,
        and calls health_check().
        """
        from beanllm.utils.config import EnvConfig

        # Map provider names to their environment variables and classes
        provider_map = {
            "openai": ("OPENAI_API_KEY", "beanllm.providers.openai_provider", "OpenAIProvider"),
            "anthropic": (
                "ANTHROPIC_API_KEY",
                "beanllm.providers.claude_provider",
                "ClaudeProvider",
            ),
            "google": ("GEMINI_API_KEY", "beanllm.providers.gemini_provider", "GeminiProvider"),
            "gemini": ("GEMINI_API_KEY", "beanllm.providers.gemini_provider", "GeminiProvider"),
            "deepseek": (
                "DEEPSEEK_API_KEY",
                "beanllm.providers.deepseek_provider",
                "DeepSeekProvider",
            ),
            "perplexity": (
                "PERPLEXITY_API_KEY",
                "beanllm.providers.perplexity_provider",
                "PerplexityProvider",
            ),
            "ollama": ("OLLAMA_HOST", "beanllm.providers.ollama_provider", "OllamaProvider"),
        }

        if provider not in provider_map:
            return ApiKeyValidationResult(
                provider=provider,
                is_valid=False,
                error=f"Unknown provider: {provider}",
            )

        env_var, module_path, class_name = provider_map[provider]

        # Save current value
        old_value = os.environ.get(env_var)

        try:
            # Temporarily set the environment variable
            if provider == "ollama":
                # Ollama uses host URL, not API key
                if api_key:
                    os.environ[env_var] = api_key
            else:
                os.environ[env_var] = api_key

            # Reload EnvConfig to pick up new value
            setattr(
                EnvConfig,
                env_var.replace("_API_KEY", "_API_KEY").replace("ANTHROPIC", "ANTHROPIC"),
                api_key,
            )

            # Dynamically import the provider class
            import importlib

            module = importlib.import_module(module_path)
            provider_class = getattr(module, class_name)

            # Create provider instance
            if provider == "ollama":
                provider_instance = provider_class({"host": api_key or EnvConfig.OLLAMA_HOST})
            else:
                provider_instance = provider_class()

            # Check if available
            if not provider_instance.is_available():
                return ApiKeyValidationResult(
                    provider=provider,
                    is_valid=False,
                    error="Provider not available (missing API key or configuration)",
                )

            # Run health check
            is_healthy = await provider_instance.health_check()

            if is_healthy:
                # Get available models if possible
                models = await self._get_available_models(provider, provider_instance)
                return ApiKeyValidationResult(
                    provider=provider,
                    is_valid=True,
                    models_available=models,
                )
            else:
                return ApiKeyValidationResult(
                    provider=provider,
                    is_valid=False,
                    error="Health check failed",
                )

        except Exception as e:
            logger.error(f"Provider validation failed for {provider}: {e}")
            return ApiKeyValidationResult(
                provider=provider,
                is_valid=False,
                error=str(e),
            )

        finally:
            # Restore original value
            if old_value is not None:
                os.environ[env_var] = old_value
            elif env_var in os.environ:
                del os.environ[env_var]

    async def _get_available_models(self, provider: str, provider_instance) -> Optional[List[str]]:
        """Get available models for a provider."""
        try:
            if provider == "ollama":
                # Ollama has list_models method
                if hasattr(provider_instance, "list_models"):
                    models = provider_instance.list_models()
                    return [m.get("name", "") for m in models[:10]]
            elif provider == "openai":
                return ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo", "o1", "o1-mini"]
            elif provider == "anthropic":
                return [
                    "claude-sonnet-4-20250514",
                    "claude-opus-4-20250514",
                    "claude-haiku-4-20250514",
                ]
            elif provider in ["google", "gemini"]:
                return ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.5-pro"]
            elif provider == "deepseek":
                return ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"]
            elif provider == "perplexity":
                return ["llama-3.1-sonar-large-128k-online", "llama-3.1-sonar-small-128k-online"]
        except Exception as e:
            logger.debug(f"Failed to get models for {provider}: {e}")

        return None

    # =========================================
    # Custom validators for non-LLM providers
    # =========================================

    async def _validate_tavily(self, api_key: str) -> ApiKeyValidationResult:
        """Validate Tavily API key."""
        import httpx

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": api_key,
                        "query": "test",
                        "max_results": 1,
                    },
                )

                if response.status_code == 200:
                    return ApiKeyValidationResult(
                        provider="tavily",
                        is_valid=True,
                    )
                elif response.status_code in [401, 403]:
                    return ApiKeyValidationResult(
                        provider="tavily",
                        is_valid=False,
                        error="Invalid API key",
                    )
                else:
                    return ApiKeyValidationResult(
                        provider="tavily",
                        is_valid=False,
                        error=f"API error: {response.status_code}",
                    )
            except Exception as e:
                return ApiKeyValidationResult(
                    provider="tavily",
                    is_valid=False,
                    error=str(e),
                )

    async def _validate_serpapi(self, api_key: str) -> ApiKeyValidationResult:
        """Validate SerpAPI key."""
        import httpx

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.get(
                    "https://serpapi.com/account",
                    params={"api_key": api_key},
                )

                if response.status_code == 200:
                    return ApiKeyValidationResult(
                        provider="serpapi",
                        is_valid=True,
                    )
                else:
                    return ApiKeyValidationResult(
                        provider="serpapi",
                        is_valid=False,
                        error="Invalid API key",
                    )
            except Exception as e:
                return ApiKeyValidationResult(
                    provider="serpapi",
                    is_valid=False,
                    error=str(e),
                )

    async def _validate_pinecone(self, api_key: str) -> ApiKeyValidationResult:
        """Validate Pinecone API key."""
        import httpx

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.get(
                    "https://api.pinecone.io/indexes",
                    headers={"Api-Key": api_key},
                )

                if response.status_code == 200:
                    return ApiKeyValidationResult(
                        provider="pinecone",
                        is_valid=True,
                    )
                elif response.status_code == 401:
                    return ApiKeyValidationResult(
                        provider="pinecone",
                        is_valid=False,
                        error="Invalid API key",
                    )
                else:
                    return ApiKeyValidationResult(
                        provider="pinecone",
                        is_valid=False,
                        error=f"API error: {response.status_code}",
                    )
            except Exception as e:
                return ApiKeyValidationResult(
                    provider="pinecone",
                    is_valid=False,
                    error=str(e),
                )

    async def _validate_qdrant(self, api_key: str) -> ApiKeyValidationResult:
        """Validate Qdrant API key."""
        import httpx

        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                headers = {}
                if api_key:
                    headers["api-key"] = api_key

                response = await client.get(
                    f"{qdrant_url}/collections",
                    headers=headers,
                )

                if response.status_code == 200:
                    return ApiKeyValidationResult(
                        provider="qdrant",
                        is_valid=True,
                    )
                elif response.status_code == 401:
                    return ApiKeyValidationResult(
                        provider="qdrant",
                        is_valid=False,
                        error="Invalid API key",
                    )
                else:
                    return ApiKeyValidationResult(
                        provider="qdrant",
                        is_valid=False,
                        error=f"API error: {response.status_code}",
                    )
            except Exception as e:
                return ApiKeyValidationResult(
                    provider="qdrant",
                    is_valid=False,
                    error=str(e),
                )

    async def _validate_weaviate(self, api_key: str) -> ApiKeyValidationResult:
        """Validate Weaviate API key."""
        import httpx

        weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                headers = {}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"

                response = await client.get(
                    f"{weaviate_url}/v1/schema",
                    headers=headers,
                )

                if response.status_code == 200:
                    return ApiKeyValidationResult(
                        provider="weaviate",
                        is_valid=True,
                    )
                elif response.status_code == 401:
                    return ApiKeyValidationResult(
                        provider="weaviate",
                        is_valid=False,
                        error="Invalid API key",
                    )
                else:
                    return ApiKeyValidationResult(
                        provider="weaviate",
                        is_valid=False,
                        error=f"API error: {response.status_code}",
                    )
            except Exception as e:
                return ApiKeyValidationResult(
                    provider="weaviate",
                    is_valid=False,
                    error=str(e),
                )


# Singleton instance
key_validator = KeyValidator()
