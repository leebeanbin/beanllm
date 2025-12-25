"""
Client Facade - 기존 Client API를 위한 Facade
책임: 하위 호환성 유지, 내부적으로는 Handler/Service 사용
SOLID 원칙:
- Facade 패턴: 복잡한 내부 구조를 단순한 인터페이스로
- DIP: 인터페이스에 의존
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional

from ..dto.response.chat_response import ChatResponse
from ..infrastructure.registry import get_model_registry

if TYPE_CHECKING:
    from .._source_providers.base_provider import BaseLLMProvider
    from .._source_providers.provider_factory import ProviderFactory as SourceProviderFactory
else:
    from .._source_providers.provider_factory import ProviderFactory as SourceProviderFactory


class Client:
    """
    통일된 LLM 클라이언트 (Facade 패턴)

    기존 API를 유지하면서 내부적으로는 Handler/Service 사용

    Example:
        ```python
        from beanllm import Client

        # 명시적 provider
        client = Client(provider="openai", model="gpt-4o-mini")
        response = await client.chat(messages, temperature=0.7)

        # provider 자동 감지
        client = Client(model="gpt-4o-mini")
        response = await client.chat(messages, temperature=0.7)
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
        self._init_services()

    def _init_services(self) -> None:
        """Service 및 Handler 초기화 (의존성 주입) - DI Container 사용"""
        from ..utils.di_container import get_container

        container = get_container()
        handler_factory = container.handler_factory

        # ChatHandler 생성
        self._chat_handler = handler_factory.create_chat_handler()

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
        return await self._chat_handler.handle_chat(
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
        """모델 ID로 Provider 자동 감지 (기존 로직 유지)"""
        registry = get_model_registry()

        # Registry에서 모델 찾기
        try:
            model_info = registry.get_model_info(model)
            if model_info:
                return model_info.provider
        except Exception:
            pass

        # 패턴 기반 감지
        model_lower = model.lower()

        if any(x in model_lower for x in ["gpt", "o1", "o3", "o4"]):
            return "openai"
        elif "claude" in model_lower:
            return "anthropic"
        elif "gemini" in model_lower:
            return "google"
        else:
            return "ollama"  # 기본값

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
            source_factory: _source_providers의 ProviderFactory
        """
        self._source_factory = source_factory
        self._provider_name_map = {
            "openai": "openai",
            "claude": "claude",  # ProviderFactory는 "claude" 사용
            "anthropic": "claude",
            "gemini": "gemini",  # ProviderFactory는 "gemini" 사용
            "google": "gemini",
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
        if not hasattr(provider, "_wrapped"):
            from .._source_providers.base_provider import LLMResponse

            original_chat = provider.chat
            original_stream_chat = provider.stream_chat

            async def wrapped_chat(messages, model, system=None, **kwargs):
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

            provider.chat = wrapped_chat
            provider.stream_chat = original_stream_chat  # 이미 str을 yield
            provider._wrapped = True

        return provider

    def _detect_provider_from_model(self, model: str) -> str:
        """모델 이름으로부터 Provider 감지"""
        model_lower = model.lower()
        if any(x in model_lower for x in ["gpt", "o1", "o3", "o4"]):
            return "openai"
        elif "claude" in model_lower:
            return "claude"
        elif "gemini" in model_lower:
            return "gemini"
        else:
            return "ollama"


# 편의 함수 (기존 API 유지)
def create_client(
    model: str, provider: Optional[str] = None, api_key: Optional[str] = None, **kwargs
) -> Client:
    """
    Client 생성 (편의 함수)

    Example:
        ```python
        client = create_client("gpt-4o-mini", temperature=0.7)
        ```
    """
    return Client(model=model, provider=provider, api_key=api_key, **kwargs)
