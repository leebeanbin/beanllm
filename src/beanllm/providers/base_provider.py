"""
Base LLM Provider
LLM 제공자 추상화 인터페이스
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator, Callable, Dict, List, Optional, TypeVar

# 선택적 의존성 - ProviderError 임포트 시도
try:
    from beanllm.utils.exceptions import ProviderError
except ImportError:
    # Fallback: 기본 Exception 사용
    class ProviderError(Exception):  # type: ignore
        """Provider 에러"""
        pass

# logger 임포트 시도
try:
    from beanllm.utils.logger import get_logger
except ImportError:
    def get_logger(name: str):
        return logging.getLogger(name)


@dataclass
class LLMResponse:
    """LLM 응답 모델"""

    content: str
    model: str
    usage: Optional[Dict] = None


T = TypeVar('T')


class BaseLLMProvider(ABC):
    """
    LLM 제공자 기본 인터페이스

    Updated with consolidated error handling utilities to reduce duplication across providers.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.name = self.__class__.__name__
        self._logger = get_logger(self.__class__.__name__)

    @abstractmethod
    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """
        스트리밍 채팅

        Args:
            messages: 대화 메시지 리스트
            model: 사용할 모델
            system: 시스템 메시지
            temperature: 온도
            max_tokens: 최대 토큰 수

        Yields:
            응답 청크 (str)
        """
        pass

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        일반 채팅 (비스트리밍)

        Args:
            messages: 대화 메시지 리스트
            model: 사용할 모델
            system: 시스템 메시지
            temperature: 온도
            max_tokens: 최대 토큰 수

        Returns:
            LLMResponse
        """
        pass

    @abstractmethod
    async def list_models(self) -> List[str]:
        """사용 가능한 모델 목록 조회"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """제공자 사용 가능 여부"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """건강 상태 확인"""
        pass

    # ============================================================================
    # Error Handling Utilities (공통 에러 핸들링 - 중복 제거)
    # ============================================================================

    def _handle_provider_error(
        self,
        error: Exception,
        operation: str,
        fallback_message: Optional[str] = None,
    ) -> ProviderError:
        """
        Provider 에러를 일관되게 처리하고 ProviderError로 변환

        모든 provider에서 반복되는 error logging + raise ProviderError 패턴을 통합

        Args:
            error: 원본 예외
            operation: 작업 이름 (예: "stream_chat", "chat", "list_models")
            fallback_message: 커스텀 에러 메시지 (None이면 자동 생성)

        Returns:
            ProviderError 인스턴스 (raise 용)

        Example:
            ```python
            try:
                # API 호출
                response = await self.client.chat(...)
            except APIError as e:
                raise self._handle_provider_error(
                    e, "chat", "OpenAI API error"
                ) from e
            except Exception as e:
                raise self._handle_provider_error(e, "chat") from e
            ```
        """
        error_message = fallback_message or f"{self.name} {operation} failed"
        full_message = f"{error_message}: {str(error)}"

        # 로깅
        self._logger.error(f"{self.name} {operation} error: {error}")

        # ProviderError로 래핑
        return ProviderError(full_message)

    async def _safe_health_check(
        self, health_check_fn: Callable[[], bool]
    ) -> bool:
        """
        Health check를 안전하게 실행 (모든 provider에서 동일한 패턴)

        모든 예외를 잡아서 False를 반환하고 로깅합니다.

        Args:
            health_check_fn: Health check 로직 함수

        Returns:
            Health check 성공 여부 (예외 발생 시 False)

        Example:
            ```python
            async def health_check(self) -> bool:
                async def check():
                    response = await self.client.chat(...)
                    return bool(response.content)

                return await self._safe_health_check(check)
            ```
        """
        try:
            return await health_check_fn()
        except Exception as e:
            self._logger.error(f"{self.name} health check failed: {e}")
            return False

    def _safe_is_available(self, check_fn: Callable[[], bool]) -> bool:
        """
        is_available을 안전하게 실행 (모든 provider에서 동일한 패턴)

        모든 예외를 잡아서 False를 반환합니다.

        Args:
            check_fn: 가용성 체크 로직 함수

        Returns:
            가용성 여부 (예외 발생 시 False)

        Example:
            ```python
            def is_available(self) -> bool:
                return self._safe_is_available(
                    lambda: bool(EnvConfig.OPENAI_API_KEY)
                )
            ```
        """
        try:
            return check_fn()
        except Exception:
            return False
