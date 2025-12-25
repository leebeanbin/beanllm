"""
Environment Configuration
환경변수 관리 (통합)
"""

import os
from pathlib import Path
from typing import Optional

# dotenv 선택적 로드
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
except ImportError:
    # dotenv가 없어도 작동하도록
    pass


class EnvConfig:
    """환경변수 설정 (외부 의존성 없음)"""

    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")

    # Hosts
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    @classmethod
    def get_active_providers(cls) -> list[str]:
        """활성화된 제공자 목록"""
        providers = []
        if cls.OPENAI_API_KEY:
            providers.append("openai")
        if cls.ANTHROPIC_API_KEY:
            providers.append("anthropic")
        if cls.GEMINI_API_KEY:
            providers.append("google")
        providers.append("ollama")  # 항상 가능
        return providers

    @classmethod
    def is_provider_available(cls, provider: str) -> bool:
        """특정 Provider 사용 가능 여부"""
        provider_map = {
            "openai": cls.OPENAI_API_KEY,
            "anthropic": cls.ANTHROPIC_API_KEY,
            "google": cls.GEMINI_API_KEY,
            "gemini": cls.GEMINI_API_KEY,
            "ollama": True,  # 항상 가능
        }
        return bool(provider_map.get(provider.lower()))


# 하위 호환성을 위한 별칭
Config = EnvConfig
