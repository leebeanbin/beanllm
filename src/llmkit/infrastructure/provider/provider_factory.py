"""
Provider Factory
제공자 팩토리
"""

from typing import List, Optional

from ...utils.config import Config


class ProviderFactory:
    PROVIDER_PRIORITY = [
        ("openai", "OPENAI_API_KEY"),
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("google", "GEMINI_API_KEY"),
        ("ollama", "OLLAMA_HOST"),
    ]

    @classmethod
    def get_available_providers(cls) -> List[str]:
        available = []
        for name, env_key in cls.PROVIDER_PRIORITY:
            try:
                if name == "ollama":
                    available.append(name)
                elif env_key == "OPENAI_API_KEY" and Config.OPENAI_API_KEY:
                    available.append(name)
                elif env_key == "ANTHROPIC_API_KEY" and Config.ANTHROPIC_API_KEY:
                    available.append(name)
                elif env_key == "GEMINI_API_KEY" and Config.GEMINI_API_KEY:
                    available.append(name)
            except Exception:
                pass
        return available

    @classmethod
    def is_provider_available(cls, provider_name: str) -> bool:
        return provider_name in cls.get_available_providers()

    @classmethod
    def get_default_provider(cls) -> Optional[str]:
        available = cls.get_available_providers()
        return available[0] if available else None
