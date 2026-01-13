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

# SecureConfig 선택적 import (보안 강화)
try:
    from beanllm.infrastructure.security import SecureConfig
    SECURE_CONFIG_AVAILABLE = True
except ImportError:
    SecureConfig = None  # type: ignore
    SECURE_CONFIG_AVAILABLE = False


class EnvConfig:
    """환경변수 설정 (외부 의존성 없음)"""

    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    DEEPSEEK_API_KEY: Optional[str] = os.getenv("DEEPSEEK_API_KEY")
    PERPLEXITY_API_KEY: Optional[str] = os.getenv("PERPLEXITY_API_KEY")

    # Hosts
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # SecureConfig 인스턴스 (마스킹 지원, lazy initialization)
    _secure_config: Optional["SecureConfig"] = None

    @classmethod
    def _get_secure_config(cls) -> Optional["SecureConfig"]:
        """SecureConfig 인스턴스 가져오기 (lazy initialization)"""
        if not SECURE_CONFIG_AVAILABLE:
            return None

        if cls._secure_config is None:
            cls._secure_config = SecureConfig(
                openai_api_key=cls.OPENAI_API_KEY,
                anthropic_api_key=cls.ANTHROPIC_API_KEY,
                gemini_api_key=cls.GEMINI_API_KEY,
                deepseek_api_key=cls.DEEPSEEK_API_KEY,
                perplexity_api_key=cls.PERPLEXITY_API_KEY,
            )
        return cls._secure_config

    @classmethod
    def get_safe_config_dict(cls) -> dict:
        """
        안전한 설정 딕셔너리 반환 (API 키 마스킹)

        로그나 출력에 사용할 수 있는 안전한 설정 딕셔너리입니다.

        Returns:
            마스킹된 설정 딕셔너리

        Example:
            >>> config_dict = EnvConfig.get_safe_config_dict()
            >>> print(config_dict)
            {'openai_api_key': '***MASKED***', 'anthropic_api_key': None, ...}
        """
        secure_config = cls._get_secure_config()
        if secure_config:
            return secure_config.to_dict(mask_secrets=True)
        else:
            # SecureConfig가 없으면 수동 마스킹
            return {
                "openai_api_key": "***MASKED***" if cls.OPENAI_API_KEY else None,
                "anthropic_api_key": "***MASKED***" if cls.ANTHROPIC_API_KEY else None,
                "gemini_api_key": "***MASKED***" if cls.GEMINI_API_KEY else None,
                "deepseek_api_key": "***MASKED***" if cls.DEEPSEEK_API_KEY else None,
                "perplexity_api_key": "***MASKED***" if cls.PERPLEXITY_API_KEY else None,
            }

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
        if cls.DEEPSEEK_API_KEY:
            providers.append("deepseek")
        if cls.PERPLEXITY_API_KEY:
            providers.append("perplexity")
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
            "deepseek": cls.DEEPSEEK_API_KEY,
            "perplexity": cls.PERPLEXITY_API_KEY,
            "ollama": True,  # 항상 가능
        }
        return bool(provider_map.get(provider.lower()))


# 하위 호환성을 위한 별칭
Config = EnvConfig
