"""
beanllm.utils.resilience.retry - Retry Logic
재시도 로직

이 모듈은 자동 재시도 메커니즘을 제공합니다:
- 다양한 재시도 전략 (고정, 선형, 지수, 지터)
- 커스터마이징 가능한 재시도 조건
- 데코레이터 지원
"""

import time
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional

from ..exceptions import MaxRetriesExceededError


class RetryStrategy(Enum):
    """재시도 전략"""

    FIXED = "fixed"  # 고정 간격
    EXPONENTIAL = "exponential"  # 지수 백오프
    LINEAR = "linear"  # 선형 증가
    JITTER = "jitter"  # 지수 백오프 + 지터


@dataclass
class RetryConfig:
    """재시도 설정"""

    max_retries: int = 3
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

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        재시도 로직으로 함수 실행

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
                    raise MaxRetriesExceededError(
                        f"Max retries ({self.config.max_retries}) exceeded. Last error: {str(e)}"
                    ) from e

                # 재시도 전 대기
                delay = self._calculate_delay(attempt)
                time.sleep(delay)

        # Should not reach here
        raise last_exception


def retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    retry_on: tuple = (Exception,),
):
    """
    재시도 데코레이터

    Example:
        @retry(max_retries=5, strategy=RetryStrategy.EXPONENTIAL)
        def api_call():
            ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            config = RetryConfig(
                max_retries=max_retries,
                initial_delay=initial_delay,
                strategy=strategy,
                retry_on_exceptions=retry_on,
            )
            handler = RetryHandler(config)
            return handler.execute(func, *args, **kwargs)

        return wrapper

    return decorator
