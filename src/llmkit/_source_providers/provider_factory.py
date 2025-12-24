"""
Provider Factory
환경 변수 기반 LLM 제공자 자동 선택 및 생성 (dotenv 중앙 관리)
"""

from typing import List, Optional

from ..utils.config import EnvConfig
from ..utils.logger import get_logger
from .base_provider import BaseLLMProvider

# 선택적 의존성
try:
    from .claude_provider import ClaudeProvider
except ImportError:
    ClaudeProvider = None  # type: ignore

try:
    from .ollama_provider import OllamaProvider
except ImportError:
    OllamaProvider = None  # type: ignore

try:
    from .gemini_provider import GeminiProvider
except ImportError:
    GeminiProvider = None  # type: ignore

try:
    from .openai_provider import OpenAIProvider
except ImportError:
    OpenAIProvider = None  # type: ignore

logger = get_logger(__name__)


class ProviderFactory:
    """LLM 제공자 팩토리"""

    _instances: dict[str, BaseLLMProvider] = {}

    @classmethod
    def _get_provider_priority(cls):
        """동적으로 제공자 우선순위 리스트 생성 (선택적 의존성 처리)"""
        priority = []

        if OpenAIProvider is not None:
            priority.append(("openai", OpenAIProvider, "OPENAI_API_KEY"))

        if ClaudeProvider is not None:
            priority.append(("claude", ClaudeProvider, "ANTHROPIC_API_KEY"))

        if GeminiProvider is not None:
            priority.append(("gemini", GeminiProvider, "GEMINI_API_KEY"))

        if OllamaProvider is not None:
            priority.append(("ollama", OllamaProvider, "OLLAMA_HOST"))  # API 키 없음

        return priority

    @classmethod
    def get_available_providers(cls) -> List[str]:
        """
        사용 가능한 제공자 목록 조회

        Returns:
            제공자 이름 리스트
        """
        available = []

        for name, provider_class, env_key in cls._get_provider_priority():
            try:
                # 환경 변수 확인 (EnvConfig 사용)
                if name == "ollama":
                    # Ollama는 API 키가 없어도 사용 가능
                    available.append(name)
                elif env_key == "OPENAI_API_KEY" and EnvConfig.OPENAI_API_KEY:
                    available.append(name)
                elif env_key == "ANTHROPIC_API_KEY" and EnvConfig.ANTHROPIC_API_KEY:
                    available.append(name)
                elif env_key == "GEMINI_API_KEY" and EnvConfig.GEMINI_API_KEY:
                    available.append(name)
            except Exception as e:
                logger.debug(f"Provider {name} not available: {e}")

        return available

    @classmethod
    def get_provider(
        cls,
        provider_name: Optional[str] = None,
        fallback: bool = True,
    ) -> BaseLLMProvider:
        """
        LLM 제공자 인스턴스 생성 또는 반환

        Args:
            provider_name: 제공자 이름 (None이면 자동 선택)
            fallback: 사용 불가 시 다음 제공자로 폴백

        Returns:
            BaseLLMProvider 인스턴스

        Raises:
            ValueError: 사용 가능한 제공자가 없을 때
        """
        # 캐시된 인스턴스 반환
        if provider_name and provider_name in cls._instances:
            return cls._instances[provider_name]

        # 제공자 선택
        if provider_name:
            # 지정된 제공자 사용
            providers_to_try = [(provider_name, None, None)]
        else:
            # 자동 선택 (환경 변수 기반)
            providers_to_try = cls._get_provider_priority()

        # 제공자 생성 시도
        last_error = None
        for name, provider_class, env_key in providers_to_try:
            try:
                # 환경 변수 확인 (EnvConfig 사용)
                if name == "ollama":
                    # Ollama는 항상 시도 (로컬 서버)
                    pass
                elif env_key == "OPENAI_API_KEY" and not EnvConfig.OPENAI_API_KEY:
                    if not fallback:
                        continue
                    logger.debug(f"Provider {name} not available (missing {env_key})")
                    continue
                elif env_key == "ANTHROPIC_API_KEY" and not EnvConfig.ANTHROPIC_API_KEY:
                    if not fallback:
                        continue
                    logger.debug(f"Provider {name} not available (missing {env_key})")
                    continue
                elif env_key == "GEMINI_API_KEY" and not EnvConfig.GEMINI_API_KEY:
                    if not fallback:
                        continue
                    logger.debug(f"Provider {name} not available (missing {env_key})")
                    continue

                # 제공자 인스턴스 생성
                if name == "ollama":
                    config = {"host": EnvConfig.OLLAMA_HOST}
                    provider = provider_class(config)
                else:
                    provider = provider_class()

                # 사용 가능 여부 확인
                if provider.is_available():
                    logger.info(f"Using LLM provider: {name}")
                    cls._instances[name] = provider
                    return provider
                else:
                    logger.debug(f"Provider {name} is not available")
                    continue

            except Exception as e:
                # Ollama는 선택적이므로 실패해도 조용히 처리 (DEBUG 레벨)
                if name == "ollama":
                    logger.debug(f"Ollama provider not available: {e}")
                else:
                    logger.debug(f"Failed to initialize provider {name}: {e}")
                last_error = e
                if not fallback:
                    break
                continue

        # 사용 가능한 제공자가 없음
        error_msg = f"No available LLM provider found. Last error: {last_error}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    @classmethod
    def get_default_provider(cls) -> BaseLLMProvider:
        """기본 제공자 반환 (자동 선택)"""
        return cls.get_provider()

    @classmethod
    def clear_cache(cls):
        """인스턴스 캐시 초기화"""
        # 리소스 정리
        for provider in cls._instances.values():
            if hasattr(provider, "close"):
                import asyncio

                try:
                    asyncio.run(provider.close())
                except Exception:
                    pass

        cls._instances.clear()
