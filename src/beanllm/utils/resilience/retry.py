"""
beanllm.utils.resilience.retry - Retry Logic
재시도 로직

이 모듈은 자동 재시도 메커니즘을 제공합니다:
- 다양한 재시도 전략 (고정, 선형, 지수, 지터)
- 커스터마이징 가능한 재시도 조건
- 데코레이터 지원
- Async/Sync 함수 모두 지원
"""

import asyncio
import inspect
import time
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, AsyncGenerator, Callable, Optional

from beanllm.utils.constants import DEFAULT_MAX_RETRIES
from beanllm.utils.exceptions import MaxRetriesExceededError
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class RetryStrategy(Enum):
    """재시도 전략"""

    FIXED = "fixed"  # 고정 간격
    EXPONENTIAL = "exponential"  # 지수 백오프
    LINEAR = "linear"  # 선형 증가
    JITTER = "jitter"  # 지수 백오프 + 지터


@dataclass
class RetryConfig:
    """재시도 설정"""

    max_retries: int = DEFAULT_MAX_RETRIES
    initial_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    retry_on_exceptions: tuple = (Exception,)
    retry_condition: Optional[Callable[[Exception], bool]] = None


class RetryHandler:
    """
    재시도 핸들러

    자동 재시도 로직 구현
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()

    def _calculate_delay(self, attempt: int) -> float:
        """재시도 지연 시간 계산"""
        import random

        if self.config.strategy == RetryStrategy.FIXED:
            delay = self.config.initial_delay

        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.initial_delay * attempt

        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.initial_delay * (self.config.multiplier ** (attempt - 1))

        elif self.config.strategy == RetryStrategy.JITTER:
            # Exponential backoff with jitter
            base_delay = self.config.initial_delay * (self.config.multiplier ** (attempt - 1))
            jitter = random.uniform(0, base_delay * 0.1)  # 10% jitter
            delay = base_delay + jitter

        else:
            delay = self.config.initial_delay

        # Max delay 제한
        return min(delay, self.config.max_delay)

    def _should_retry(self, exception: Exception) -> bool:
        """재시도 여부 판단"""
        # 예외 타입 확인
        if not isinstance(exception, self.config.retry_on_exceptions):
            return False

        # 커스텀 조건 확인
        if self.config.retry_condition:
            return self.config.retry_condition(exception)

        return True

    async def execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        재시도 로직으로 async 함수 실행

        Args:
            func: 실행할 async 함수
            *args, **kwargs: 함수 인자

        Returns:
            함수 실행 결과

        Raises:
            MaxRetriesExceededError: 최대 재시도 횟수 초과
        """
        last_exception = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                return await func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                if not self._should_retry(e):
                    raise

                if attempt >= self.config.max_retries:
                    logger.error(
                        f"Failed after {self.config.max_retries} attempts: {func.__name__}"
                    )
                    raise MaxRetriesExceededError(
                        f"Max retries ({self.config.max_retries}) exceeded. Last error: {str(e)}"
                    ) from e

                # 재시도 전 대기
                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"Attempt {attempt}/{self.config.max_retries} failed for {func.__name__}: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                await asyncio.sleep(delay)

        # Should not reach here
        if last_exception is not None:
            raise last_exception
        raise RuntimeError("Retry loop exited without raising")

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        재시도 로직으로 함수 실행 (sync)

        Args:
            func: 실행할 함수
            *args, **kwargs: 함수 인자

        Returns:
            함수 실행 결과

        Raises:
            MaxRetriesExceededError: 최대 재시도 횟수 초과
        """
        last_exception = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                if not self._should_retry(e):
                    raise

                if attempt >= self.config.max_retries:
                    logger.error(
                        f"Failed after {self.config.max_retries} attempts: {func.__name__}"
                    )
                    raise MaxRetriesExceededError(
                        f"Max retries ({self.config.max_retries}) exceeded. Last error: {str(e)}"
                    ) from e

                # 재시도 전 대기
                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"Attempt {attempt}/{self.config.max_retries} failed for {func.__name__}: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)

        # Should not reach here
        if last_exception is not None:
            raise last_exception
        raise RuntimeError("Retry loop exited without raising")


def retry(
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = 1.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    retry_on: tuple = (Exception,),
):
    """재시도 데코레이터 (Async / Sync / AsyncGenerator 모두 지원).

    주의: retry_on에 asyncio.CancelledError를 포함하지 말 것.
    CancelledError는 BaseException 파생이므로 Exception 튜플에도 잡히지 않지만,
    명시적으로 제외 의도를 코드로 표현하는 것이 좋음.

    Example:
        @retry(max_retries=3, retry_on=(APITimeoutError, RateLimitError))
        async def api_call(): ...

        @retry(max_retries=3, retry_on=(APITimeoutError,))
        async def stream_chat(...) -> AsyncGenerator[str, None]:
            async for chunk in client.stream(...):
                yield chunk
    """

    def decorator(func: Callable) -> Callable:
        config = RetryConfig(
            max_retries=max_retries,
            initial_delay=initial_delay,
            strategy=strategy,
            retry_on_exceptions=retry_on,
        )
        handler = RetryHandler(config)

        # 1) Async generator 함수: asyncio.iscoroutinefunction()은 False를 반환하므로
        #    반드시 먼저 확인해야 함. 재시도 시 제너레이터 전체를 재시작.
        if inspect.isasyncgenfunction(func):

            @wraps(func)
            async def asyncgen_wrapper(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
                last_exc: Optional[BaseException] = None
                for attempt in range(1, max_retries + 1):
                    try:
                        async for item in func(*args, **kwargs):
                            yield item
                        return  # 성공
                    except asyncio.CancelledError:
                        raise  # 취소는 재시도 불가 — 즉시 전파
                    except Exception as e:
                        last_exc = e
                        if not isinstance(e, retry_on):
                            raise
                        if attempt >= max_retries:
                            logger.error(
                                f"AsyncGenerator {func.__name__} failed after "
                                f"{max_retries} retries: {e!r}"
                            )
                            raise MaxRetriesExceededError(
                                f"Max retries ({max_retries}) exceeded for "
                                f"{func.__name__}. Last error: {e}"
                            ) from e
                        delay = handler._calculate_delay(attempt)
                        logger.warning(
                            f"AsyncGenerator {func.__name__} attempt {attempt}/{max_retries} "
                            f"failed: {e!r}. Retrying in {delay:.2f}s..."
                        )
                        await asyncio.sleep(delay)

            return asyncgen_wrapper  # type: ignore[return-value]

        # 2) 일반 async 함수
        elif asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await handler.execute_async(func, *args, **kwargs)

            return async_wrapper

        # 3) 동기 함수
        else:

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return handler.execute(func, *args, **kwargs)

            return sync_wrapper

    return decorator
