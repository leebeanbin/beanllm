"""
Provider Factory
환경 변수 기반 LLM 제공자 자동 선택 및 생성 (dotenv 중앙 관리)
"""

from typing import List, Optional

from beanllm.utils.config import EnvConfig
from beanllm.utils.logging import get_logger
from .base_provider import BaseLLMProvider

# Get logger first for error logging
logger = get_logger(__name__)

# 선택적 의존성
try:
    from .claude_provider import ClaudeProvider
except Exception as e:
    logger.warning(f"Failed to import ClaudeProvider: {e}")
    ClaudeProvider = None  # type: ignore

try:
    from .ollama_provider import OllamaProvider
except Exception as e:
    logger.warning(f"Failed to import OllamaProvider: {e}")
    OllamaProvider = None  # type: ignore

try:
    from .gemini_provider import GeminiProvider
except Exception as e:
    logger.warning(f"Failed to import GeminiProvider: {e}")
    GeminiProvider = None  # type: ignore

try:
    from .openai_provider import OpenAIProvider
except Exception as e:
    logger.warning(f"Failed to import OpenAIProvider: {e}")
    OpenAIProvider = None  # type: ignore

try:
    from .deepseek_provider import DeepSeekProvider
except Exception as e:
    logger.warning(f"Failed to import DeepSeekProvider: {e}")
    DeepSeekProvider = None  # type: ignore

try:
    from .perplexity_provider import PerplexityProvider
except Exception as e:
    logger.warning(f"Failed to import PerplexityProvider: {e}")
    PerplexityProvider = None  # type: ignore

# Debug: Log which providers are available
logger.info(f"Provider import status: OpenAI={OpenAIProvider is not None}, Claude={ClaudeProvider is not None}, "
            f"Gemini={GeminiProvider is not None}, DeepSeek={DeepSeekProvider is not None}, "
            f"Perplexity={PerplexityProvider is not None}, Ollama={OllamaProvider is not None}")


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

        if DeepSeekProvider is not None:
            priority.append(("deepseek", DeepSeekProvider, "DEEPSEEK_API_KEY"))

        if PerplexityProvider is not None:
            priority.append(("perplexity", PerplexityProvider, "PERPLEXITY_API_KEY"))

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
                elif env_key == "DEEPSEEK_API_KEY" and EnvConfig.DEEPSEEK_API_KEY:
                    available.append(name)
                elif env_key == "PERPLEXITY_API_KEY" and EnvConfig.PERPLEXITY_API_KEY:
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
            # 지정된 제공자 사용 - priority 리스트에서 찾기
            priority_list = cls._get_provider_priority()
            matching_providers = [p for p in priority_list if p[0] == provider_name]
            if matching_providers:
                providers_to_try = matching_providers
            else:
                # Provider가 priority 리스트에 없어도 생성 시도 (API 키 없어도 가능)
                # Provider 클래스 매핑
                provider_map = {
                    "openai": (OpenAIProvider, "OPENAI_API_KEY"),
                    "claude": (ClaudeProvider, "ANTHROPIC_API_KEY"),
                    "gemini": (GeminiProvider, "GEMINI_API_KEY"),
                    "deepseek": (DeepSeekProvider, "DEEPSEEK_API_KEY"),
                    "perplexity": (PerplexityProvider, "PERPLEXITY_API_KEY"),
                    "ollama": (OllamaProvider, "OLLAMA_HOST"),
                }
                if provider_name in provider_map:
                    prov_class, env_key = provider_map[provider_name]
                    if prov_class is not None:
                        providers_to_try = [(provider_name, prov_class, env_key)]
                    else:
                        # Provider class is None (failed to import) - try fallback
                        if fallback:
                            providers_to_try = cls._get_provider_priority()
                        else:
                            raise ValueError(f"Provider {provider_name} is not installed")
                else:
                    raise ValueError(f"Unknown provider: {provider_name}")
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
                elif env_key == "DEEPSEEK_API_KEY" and not EnvConfig.DEEPSEEK_API_KEY:
                    if not fallback:
                        continue
                    logger.debug(f"Provider {name} not available (missing {env_key})")
                    continue
                elif env_key == "PERPLEXITY_API_KEY" and not EnvConfig.PERPLEXITY_API_KEY:
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
                # 에러 메시지에서 API 키 마스킹 (Helper 함수 사용)
                from beanllm.utils.integration.security import sanitize_error_message
                error_str = sanitize_error_message(e)
                
                if name == "ollama":
                    logger.debug(f"Ollama provider not available: {error_str}")
                else:
                    logger.debug(f"Failed to initialize provider {name}: {error_str}")
                last_error = e
                if not fallback:
                    break
                continue

        # 사용 가능한 제공자가 없음
        error_msg = f"No available LLM provider found"
        if last_error:
            from beanllm.utils.integration.security import sanitize_error_message
            safe_error = sanitize_error_message(last_error)
            error_msg = f"{error_msg}. Last error: {safe_error}"
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
                except Exception as e:
                    logger.debug(f"Provider close failed in cleanup (safe to ignore): {e}")

        cls._instances.clear()
