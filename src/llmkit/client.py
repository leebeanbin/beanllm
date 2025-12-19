"""
Client - Unified LLM Interface
모든 Provider를 통일된 방식으로 사용
"""

from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

from .adapter import adapt_parameters
from .registry import get_model_registry
from .utils.exceptions import ProviderError
from .utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ChatResponse:
    """채팅 응답"""

    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    raw_response: Optional[Any] = None


class Client:
    """
    통일된 LLM 클라이언트

    모든 Provider를 동일한 인터페이스로 사용:
    - 파라미터 자동 변환
    - 에러 처리 통일
    - Provider 자동 감지 (선택)

    Example:
        ```python
        from llmkit import Client

        # 명시적 provider
        client = Client(provider="openai", model="gpt-4o-mini")
        response = await client.chat(messages, temperature=0.7)

        # provider 자동 감지
        client = Client(model="gpt-4o-mini")
        response = await client.chat(messages, temperature=0.7)
        ```
    """

    def __init__(
        self, model: str, provider: Optional[str] = None, api_key: Optional[str] = None, **kwargs
    ):
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

        # Provider 결정
        if provider:
            self.provider = provider
        else:
            self.provider = self._detect_provider(model)
            logger.info(f"Auto-detected provider: {self.provider} for model: {model}")

        # Provider 인스턴스 생성
        self._provider_instance = self._create_provider(self.provider)

        logger.info(f"Client initialized: {self.provider}/{self.model}")

    async def chat(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> ChatResponse:
        """
        채팅 완료 (비스트리밍)

        Args:
            messages: 메시지 목록 [{"role": "user", "content": "..."}]
            system: 시스템 프롬프트
            temperature: 온도 (0.0-1.0)
            max_tokens: 최대 토큰 수
            top_p: Top-p 샘플링
            **kwargs: 추가 파라미터

        Returns:
            ChatResponse: 응답

        Example:
            ```python
            messages = [{"role": "user", "content": "Hello!"}]
            response = await client.chat(messages, temperature=0.7)
            print(response.content)
            ```
        """
        # 파라미터 준비
        params = self._prepare_parameters(
            temperature=temperature, max_tokens=max_tokens, top_p=top_p, stream=False, **kwargs
        )

        logger.debug(f"Calling {self.provider}/{self.model} with params: {params}")

        try:
            # Provider 호출
            response = await self._provider_instance.chat(
                messages=messages, model=self.model, system=system, **params
            )

            return ChatResponse(
                content=response.get("content", ""),
                model=self.model,
                provider=self.provider,
                usage=response.get("usage"),
                finish_reason=response.get("finish_reason"),
                raw_response=response,
            )

        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise ProviderError(f"Chat failed for {self.provider}/{self.model}: {e}")

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        채팅 스트리밍

        Args:
            messages: 메시지 목록
            system: 시스템 프롬프트
            temperature: 온도
            max_tokens: 최대 토큰 수
            top_p: Top-p
            **kwargs: 추가 파라미터

        Yields:
            str: 스트리밍 청크

        Example:
            ```python
            async for chunk in client.stream_chat(messages, temperature=0.7):
                print(chunk, end="", flush=True)
            ```
        """
        # 파라미터 준비
        params = self._prepare_parameters(
            temperature=temperature, max_tokens=max_tokens, top_p=top_p, stream=True, **kwargs
        )

        logger.debug(f"Streaming {self.provider}/{self.model} with params: {params}")

        try:
            # Provider 호출 (스트리밍)
            async for chunk in self._provider_instance.stream_chat(
                messages=messages, model=self.model, system=system, **params
            ):
                yield chunk

        except Exception as e:
            logger.error(f"Stream chat error: {e}")
            raise ProviderError(f"Stream chat failed for {self.provider}/{self.model}: {e}")

    def _prepare_parameters(self, **kwargs) -> Dict[str, Any]:
        """파라미터 준비 및 변환"""
        # None 값 제거
        params = {k: v for k, v in kwargs.items() if v is not None}

        # ParameterAdapter로 변환
        adapted = adapt_parameters(self.provider, self.model, params)

        if adapted.removed:
            logger.warning(f"Removed parameters: {adapted.removed}")

        if adapted.warnings:
            for warning in adapted.warnings:
                logger.warning(warning)

        return adapted.params

    def _create_provider(self, provider: str):
        """Provider 인스턴스 생성"""
        try:
            if provider == "openai":
                from ._source_providers.openai_provider import OpenAIProvider

                return OpenAIProvider()
            elif provider == "anthropic":
                from ._source_providers.claude_provider import ClaudeProvider

                return ClaudeProvider()
            elif provider == "google":
                from ._source_providers.gemini_provider import GeminiProvider

                return GeminiProvider()
            elif provider == "ollama":
                from ._source_providers.ollama_provider import OllamaProvider

                return OllamaProvider()
            else:
                raise ProviderError(f"Unknown provider: {provider}")
        except ImportError as e:
            raise ProviderError(
                f"Failed to import provider '{provider}': {e}. "
                f"Make sure the required SDK is installed."
            )
        except Exception as e:
            raise ProviderError(f"Failed to initialize provider '{provider}': {e}")

    def _detect_provider(self, model: str) -> str:
        """모델 ID로 Provider 자동 감지"""
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


# 편의 함수
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
