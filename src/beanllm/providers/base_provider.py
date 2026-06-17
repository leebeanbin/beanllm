"""
Base LLM Provider
LLM 제공자 추상화 인터페이스
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, TypeVar

from beanllm.utils.constants import DEFAULT_TEMPERATURE
from beanllm.utils.exceptions import CircuitBreakerError, ProviderError
from beanllm.utils.logging import get_logger
from beanllm.utils.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerConfig


@dataclass
class LLMResponse:
    """LLM 응답 모델"""

    content: str
    model: str
    usage: Optional[Dict] = None


T = TypeVar("T")


class BaseLLMProvider(ABC):
    """
    LLM 제공자 기본 인터페이스

    Updated with consolidated error handling utilities to reduce duplication across providers.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.name = self.__class__.__name__
        self._logger = get_logger(self.__class__.__name__)

        # Circuit Breaker: provider별 독립 인스턴스.
        # 설정은 config["circuit_breaker"] 딕셔너리로 주입 가능.
        cb_cfg = config.get("circuit_breaker", {})
        self._circuit_breaker = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=cb_cfg.get("failure_threshold", 5),
                success_threshold=cb_cfg.get("success_threshold", 2),
                timeout=cb_cfg.get("timeout", 60.0),
            )
        )

    @abstractmethod
    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        system: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
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
        ...

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        system: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
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

        # 에러 메시지에서 API 키 마스킹 (Helper 함수 사용)
        from beanllm.utils.integration.security import sanitize_error_message

        error_str = sanitize_error_message(error)

        full_message = f"{error_message}: {error_str}"

        # 로깅 (마스킹된 메시지)
        self._logger.error(f"{self.name} {operation} error: {error_str}")

        # ProviderError로 래핑
        return ProviderError(full_message)

    def _extract_params(
        self,
        kwargs: Dict[str, Any],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        파라미터 추출 헬퍼 (kwargs에서 우선순위로 추출)

        모든 Provider에서 반복되는 kwargs.get() 패턴을 통합

        Args:
            kwargs: 추가 파라미터 딕셔너리
            temperature: 기본 temperature 값
            max_tokens: 기본 max_tokens 값

        Returns:
            추출된 파라미터 딕셔너리

        Example:
            ```python
            params = self._extract_params(kwargs, temperature, max_tokens)
            temperature_param = params["temperature"]
            max_tokens_param = params["max_tokens"]
            ```
        """
        return {
            "temperature": kwargs.get("temperature", temperature),
            "max_tokens": kwargs.get("max_tokens", max_tokens),
        }

    def _prepare_openai_messages(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        OpenAI 호환 메시지 형식으로 변환

        모든 OpenAI 호환 Provider에서 반복되는 메시지 변환 패턴을 통합

        Args:
            messages: 원본 메시지 리스트
            system: 시스템 프롬프트 (선택적)

        Returns:
            변환된 메시지 리스트

        Example:
            ```python
            openai_messages = self._prepare_openai_messages(messages, system)
            ```
        """
        openai_messages = messages.copy()
        if system:
            openai_messages.insert(0, {"role": "system", "content": system})
        return openai_messages

    def _extract_openai_usage(self, response: Any) -> Optional[Dict[str, int]]:
        """
        OpenAI 호환 API 응답에서 Usage 정보 추출

        모든 OpenAI 호환 Provider에서 반복되는 usage 추출 패턴을 통합

        Args:
            response: API 응답 객체

        Returns:
            Usage 정보 딕셔너리 또는 None

        Example:
            ```python
            usage_info = self._extract_openai_usage(response)
            ```
        """
        if hasattr(response, "usage") and response.usage:
            return {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return None

    async def _safe_health_check(self, health_check_fn: Callable[[], Awaitable[bool]]) -> bool:
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

    # ============================================================================
    # Rate Limiting Utilities (분산 Rate Limiting 지원)
    # ============================================================================

    async def _acquire_rate_limit(
        self,
        key: Optional[str] = None,
        cost: float = 1.0,
    ) -> None:
        """
        Rate Limit 획득 (분산 또는 인메모리)

        모든 Provider에서 API 호출 전 Rate Limiting 적용

        Args:
            key: Rate Limit 키 (None이면 provider 이름 사용)
            cost: 요청 비용 (기본값: 1.0)

        Example:
            ```python
            async def chat(self, ...):
                await self._acquire_rate_limit(f"{self.name}:{model}", cost=1.0)
                # API 호출
                response = await self.client.chat(...)
            ```
        """
        try:
            from beanllm.infrastructure.distributed.factory import get_rate_limiter

            rate_limiter = get_rate_limiter()
            rate_limit_key = key or f"llm:{self.name.lower()}"
            await rate_limiter.wait(rate_limit_key, cost=cost)
        except Exception as e:
            # Rate Limiting 실패 시 로깅만 하고 계속 진행 (Fallback)
            self._logger.warning(
                f"Rate limiting failed for {self.name}: {e}, continuing without rate limit"
            )

    async def _call_with_circuit_breaker(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Circuit breaker를 통해 async API 함수를 보호 실행.

        연속 실패가 failure_threshold(기본 5)를 넘으면 OPEN 상태로 전환되어
        timeout(기본 60초) 동안 모든 호출을 즉시 CircuitBreakerError로 차단합니다.
        이후 HALF_OPEN 상태에서 success_threshold(기본 2)번 성공하면 복구됩니다.

        Example:
            response = await self._call_with_circuit_breaker(
                self.client.messages.create, **api_params
            )

        Raises:
            CircuitBreakerError: circuit이 OPEN 상태일 때
            ProviderError: API 호출 자체가 실패했을 때
        """
        try:
            return await self._circuit_breaker.async_call(func, *args, **kwargs)
        except CircuitBreakerError:
            self._logger.warning(
                f"{self.name} circuit breaker OPEN — " f"state: {self._circuit_breaker.get_state()}"
            )
            raise

    def get_circuit_breaker_state(self) -> Dict[str, Any]:
        """현재 circuit breaker 상태 조회. 모니터링·디버깅용."""
        return {
            "provider": self.name,
            **self._circuit_breaker.get_state(),
        }
