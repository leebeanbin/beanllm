"""
Client Facade - 기존 Client API를 위한 Facade
책임: 하위 호환성 유지, 내부적으로는 Handler/Service 사용
SOLID 원칙:
- Facade 패턴: 복잡한 내부 구조를 단순한 인터페이스로
- DIP: 인터페이스에 의존
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, cast

from beanllm.dto.response.core.chat_response import ChatResponse
from beanllm.facade.base import FacadeBase
from beanllm.infrastructure.registry import get_model_registry
from beanllm.utils.constants import DEFAULT_TEMPERATURE
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from beanllm.providers.base_provider import BaseLLMProvider
    from beanllm.providers.provider_factory import ProviderFactory as SourceProviderFactory
else:
    from beanllm.providers.provider_factory import ProviderFactory as SourceProviderFactory


class Client(FacadeBase):
    """
    통일된 LLM 클라이언트 (Facade 패턴)

    기존 API를 유지하면서 내부적으로는 Handler/Service 사용

    Example:
        ```python
        from beanllm import Client

        # 명시적 provider
        client = Client(provider="openai", model="gpt-4o-mini")
        response = await client.chat(messages, temperature=DEFAULT_TEMPERATURE)

        # provider 자동 감지
        client = Client(model="gpt-4o-mini")
        response = await client.chat(messages, temperature=DEFAULT_TEMPERATURE)
        ```
    """

    def __init__(
        self,
        model: str,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model: 모델 ID (예: "gpt-4o-mini", "claude-3-5-sonnet-20241022")
            provider: Provider 이름 (생략 시 자동 감지)
            api_key: API 키 (생략 시 환경변수에서 로드)
            **kwargs: Provider별 추가 설정
        """
        self.model = model
        self.api_key = api_key
        self.extra_kwargs = kwargs

        # Provider 결정 (기존 로직 유지)
        if provider:
            self.provider = provider
        else:
            self.provider = self._detect_provider(model)

        # Handler/Service 초기화 (의존성 주입)
        super().__init__()

    def _init_handlers(self) -> None:
        """Create ChatHandler via handler factory."""
        self._chat_handler = self._handler_factory.create_chat_handler()

    async def chat(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """
        채팅 완료 (비스트리밍)

        내부적으로 Handler를 사용하여 처리

        Args:
            messages: 메시지 목록 [{"role": "user", "content": "..."}]
            system: 시스템 프롬프트
            temperature: 온도 (0.0-1.0)
            max_tokens: 최대 토큰 수
            top_p: Top-p 샘플링
            **kwargs: 추가 파라미터

        Returns:
            ChatResponse: 응답
        """
        # Handler를 통한 처리 (기존 API 유지)
        result = await self._chat_handler.handle_chat(
            messages=messages,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            system=system,
            stream=False,
            provider=self.provider,
            **{**self.extra_kwargs, **kwargs},
        )
        return cast(ChatResponse, result)

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        채팅 스트리밍

        내부적으로 Handler를 사용하여 처리

        Args:
            messages: 메시지 목록
            system: 시스템 프롬프트
            temperature: 온도
            max_tokens: 최대 토큰 수
            top_p: Top-p
            **kwargs: 추가 파라미터

        Yields:
            str: 스트리밍 청크
        """
        # Handler를 통한 처리
        async for chunk in self._chat_handler.handle_stream_chat(
            messages=messages,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            system=system,
            provider=self.provider,
            **{**self.extra_kwargs, **kwargs},
        ):
            yield chunk

    def _detect_provider(self, model: str) -> str:
        """모델 ID로 Provider 자동 감지

        우선순위:
        1. Registry에서 모델 찾기 (가장 정확함 - 등록된 provider 사용)
        2. 패턴 기반 감지 (provider_registry 활용)

        참고: Registry에 등록된 모델은 그 provider를 우선 사용합니다.
        예: deepseek-chat은 Registry에 DEEPSEEK provider로 등록되어 있으므로
        처음부터 올바른 provider를 사용합니다.
        """
        from beanllm.providers.provider_registry import detect_provider_from_model

        registry = get_model_registry()

        # 1. Registry에서 모델 찾기 (가장 우선)
        try:
            model_info = registry.get_model_info(model)
            if model_info:
                return model_info.provider
        except Exception as e:
            logger.debug("Failed to get provider from registry (using fallback): %s", e)

        # 2. 중앙 Registry 기반 패턴 감지
        return detect_provider_from_model(model)

    def __repr__(self) -> str:
        return f"Client(provider={self.provider!r}, model={self.model!r})"


class SourceProviderFactoryAdapter:
    """
    SourceProviderFactory를 ServiceFactory가 사용할 수 있도록 어댑터

    책임:
    - 기존 ProviderFactory를 새로운 인터페이스에 맞게 변환
    - Adapter 패턴 적용
    """

    def __init__(self, source_factory: SourceProviderFactory) -> None:
        """
        Args:
            source_factory: providers의 ProviderFactory
        """
        self._source_factory = source_factory
        self._provider_name_map = {
            "openai": "openai",
            "claude": "claude",  # ProviderFactory는 "claude" 사용
            "anthropic": "claude",
            "gemini": "gemini",  # ProviderFactory는 "gemini" 사용
            "google": "gemini",
            "deepseek": "deepseek",
            "perplexity": "perplexity",
            "ollama": "ollama",
        }

    def create(self, model: str, provider_name: Optional[str] = None) -> "BaseLLMProvider":
        """
        Provider 생성 (어댑터 메서드)

        Args:
            model: 모델 이름
            provider_name: Provider 이름 (선택적)

        Returns:
            Provider 인스턴스 (name 속성 포함, dict 반환)
        """
        # Provider 이름 정규화
        if provider_name:
            normalized_name = self._provider_name_map.get(provider_name, provider_name)
        else:
            # 모델로부터 provider 감지
            normalized_name = self._detect_provider_from_model(model)

        # 기존 ProviderFactory 사용
        provider = self._source_factory.get_provider(provider_name=normalized_name)

        # name 속성 설정 (Service에서 필요 - 소문자 이름)
        provider.name = normalized_name

        # chat 메서드가 dict를 반환하도록 래핑 (LLMResponse -> dict)
        if not getattr(provider, "_wrapped", None):
            from beanllm.providers.base_provider import LLMResponse

            original_chat = provider.chat
            original_stream_chat = provider.stream_chat

            async def wrapped_chat(
                messages: Any, model: str, system: Any = None, **kwargs: Any
            ) -> Any:
                """LLMResponse를 dict로 변환"""
                response = await original_chat(messages, model, system, **kwargs)
                # LLMResponse를 dict로 변환
                if isinstance(response, LLMResponse):
                    return {
                        "content": response.content,
                        "usage": response.usage,
                        "finish_reason": None,  # LLMResponse에는 없음
                    }
                # 이미 dict인 경우
                return response

            setattr(provider, "chat", wrapped_chat)
            setattr(provider, "stream_chat", original_stream_chat)
            setattr(provider, "_wrapped", True)

        return provider

    def _detect_provider_from_model(self, model: str) -> str:
        """모델 이름으로부터 Provider 감지 (ProviderFactory용 이름 반환)"""
        from beanllm.providers.provider_registry import (
            PROVIDER_FACTORY_NAME_MAP,
            detect_provider_from_model,
        )

        detected = detect_provider_from_model(model)
        return PROVIDER_FACTORY_NAME_MAP.get(detected, detected)


# 편의 함수 (기존 API 유지)
def create_client(
    model: str, provider: Optional[str] = None, api_key: Optional[str] = None, **kwargs
) -> Client:
    """
    Client 생성 (편의 함수)

    Example:
        ```python
        client = create_client("gpt-4o-mini", temperature=DEFAULT_TEMPERATURE)
        ```
    """
    return Client(model=model, provider=provider, api_key=api_key, **kwargs)
