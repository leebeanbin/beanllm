"""
Runtime Configuration Service

Manages loading API keys from MongoDB and injecting them into environment variables.
Also handles reloading beanllm's EnvConfig when keys change.
"""

import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ConfigService:
    """
    Service for managing runtime configuration.

    Handles:
    - Loading API keys from MongoDB on startup
    - Injecting keys into environment variables
    - Reloading beanllm's EnvConfig
    - Managing configuration state

    Usage:
        config_service = ConfigService()
        await config_service.load_keys_from_db(db)
    """

    _instance: Optional["ConfigService"] = None
    _loaded_providers: Dict[str, bool] = {}

    def __new__(cls) -> "ConfigService":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def loaded_providers(self) -> Dict[str, bool]:
        """Get the list of loaded providers and their status."""
        return self._loaded_providers.copy()

    async def load_keys_from_db(self, db) -> int:
        """
        Load all API keys from MongoDB and inject into environment variables.

        Args:
            db: MongoDB database instance

        Returns:
            Number of keys loaded successfully
        """
        from services.encryption_service import encryption_service
        from schemas.database import PROVIDER_CONFIG

        keys_collection = db["api_keys"]
        count = 0
        self._loaded_providers = {}

        try:
            async for doc in keys_collection.find({}):
                provider = doc.get("provider")
                if not provider or provider not in PROVIDER_CONFIG:
                    continue

                env_var = PROVIDER_CONFIG[provider]["env_var"]

                try:
                    # Decrypt the key
                    api_key = encryption_service.decrypt(doc["key_encrypted"])

                    # Inject into environment
                    os.environ[env_var] = api_key

                    # Track loaded provider
                    self._loaded_providers[provider] = doc.get("is_valid", False)
                    count += 1

                    logger.info(f"Loaded API key for {provider} (valid: {doc.get('is_valid', False)})")

                except Exception as e:
                    logger.warning(f"Failed to load API key for {provider}: {e}")
                    self._loaded_providers[provider] = False

            # Reload beanllm's EnvConfig
            self._reload_env_config()

            logger.info(f"Loaded {count} API keys from MongoDB")
            return count

        except Exception as e:
            logger.error(f"Failed to load API keys from MongoDB: {e}")
            return 0

    def _reload_env_config(self) -> None:
        """
        Reload beanllm's EnvConfig to pick up new environment variables.

        This ensures that beanllm's providers see the updated keys.
        """
        try:
            from beanllm.utils.config import EnvConfig

            # Update EnvConfig class attributes from environment
            EnvConfig.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            EnvConfig.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
            EnvConfig.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
            EnvConfig.DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
            EnvConfig.PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
            EnvConfig.OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

            # Clear the secure config cache to force reload
            EnvConfig._secure_config = None

            logger.debug("Reloaded beanllm EnvConfig")

        except ImportError:
            logger.warning("beanllm not installed, skipping EnvConfig reload")
        except Exception as e:
            logger.warning(f"Failed to reload EnvConfig: {e}")

    def set_key(self, provider: str, api_key: str) -> None:
        """
        Set an API key in the environment and reload config.

        Args:
            provider: Provider name
            api_key: The API key to set
        """
        from schemas.database import PROVIDER_CONFIG

        if provider not in PROVIDER_CONFIG:
            raise ValueError(f"Unknown provider: {provider}")

        env_var = PROVIDER_CONFIG[provider]["env_var"]
        os.environ[env_var] = api_key

        # Update loaded providers
        self._loaded_providers[provider] = True  # Assume valid until validated

        # Reload config
        self._reload_env_config()

        logger.info(f"Set API key for {provider}")

    def remove_key(self, provider: str) -> None:
        """
        Remove an API key from the environment.

        Args:
            provider: Provider name
        """
        from schemas.database import PROVIDER_CONFIG

        if provider not in PROVIDER_CONFIG:
            return

        env_var = PROVIDER_CONFIG[provider]["env_var"]

        if env_var in os.environ:
            del os.environ[env_var]

        # Update loaded providers
        if provider in self._loaded_providers:
            del self._loaded_providers[provider]

        # Reload config
        self._reload_env_config()

        logger.info(f"Removed API key for {provider}")

    def get_config_status(self) -> Dict[str, any]:
        """
        Get current configuration status.

        Returns:
            Dictionary with provider status and active providers list.
        """
        from schemas.database import PROVIDER_CONFIG

        status = {
            "providers": {},
            "active_providers": [],
        }

        for provider_id, config in PROVIDER_CONFIG.items():
            env_var = config["env_var"]
            has_key = bool(os.getenv(env_var))
            is_valid = self._loaded_providers.get(provider_id)

            status["providers"][provider_id] = {
                "name": config["name"],
                "has_key": has_key,
                "is_valid": is_valid,
            }

            if has_key:
                status["active_providers"].append(provider_id)

        return status

    def clear_provider_cache(self) -> None:
        """
        Clear beanllm's ProviderFactory cache.

        Call this after adding/removing keys to ensure fresh provider instances.
        """
        try:
            from beanllm.providers.provider_factory import ProviderFactory
            ProviderFactory.clear_cache()
            logger.debug("Cleared ProviderFactory cache")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to clear ProviderFactory cache: {e}")


# Singleton instance
config_service = ConfigService()


async def init_config_on_startup(db) -> None:
    """
    Initialize configuration on application startup.

    This should be called during FastAPI lifespan startup.

    Args:
        db: MongoDB database instance
    """
    logger.info("Initializing configuration from MongoDB...")
    count = await config_service.load_keys_from_db(db)
    logger.info(f"Configuration initialized: {count} API keys loaded")


def get_config_service() -> ConfigService:
    """Get the config service instance."""
    return config_service
