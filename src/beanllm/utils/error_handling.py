"""
beanllm.error_handling - Advanced Error Handling
고급 에러 처리 시스템

이 모듈은 프로덕션급 에러 처리를 제공합니다.

Note: 이 모듈은 backward compatibility를 위해 모든 클래스를 re-export합니다.
새로운 코드에서는 다음 모듈에서 직접 import하는 것을 권장합니다:
- beanllm.utils.exceptions - 예외 클래스
- beanllm.utils.resilience.retry - 재시도 로직
- beanllm.utils.resilience.circuit_breaker - Circuit Breaker
- beanllm.utils.resilience.rate_limiter - Rate Limiting
- beanllm.utils.resilience.error_tracker - Error Tracking
"""

import signal
import threading
from functools import wraps
from typing import Any, Callable, Dict, Optional

# ===== Re-export Exceptions =====
from .exceptions import (
    CircuitBreakerError,
    LLMKitError,
    MaxRetriesExceededError,
    ProviderError,
    RateLimitError,
    TimeoutError,
    ValidationError,
)
from .resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    circuit_breaker,
)
from .resilience.error_tracker import (
    ErrorRecord,
    ErrorTracker,
    FallbackHandler,
    ProductionErrorSanitizer,
    create_safe_error_response,
    get_error_tracker,
    sanitize_error_message,
)
from .resilience.rate_limiter import (
    AsyncTokenBucket,
    RateLimitConfig,
    RateLimiter,
    rate_limit,
)

# ===== Re-export Resilience Components =====
from .resilience.retry import (
    RetryConfig,
    RetryHandler,
    RetryStrategy,
    retry,
)

# ===== Combined Error Handler =====


class ErrorHandlerConfig:
    """통합 에러 핸들러 설정"""

    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        rate_limit_config: Optional[RateLimitConfig] = None,
        enable_tracking: bool = True,
    ):
        self.retry_config = retry_config
        self.circuit_breaker_config = circuit_breaker_config
        self.rate_limit_config = rate_limit_config
        self.enable_tracking = enable_tracking


class ErrorHandler:
    """
    통합 에러 핸들러

    Retry, Circuit Breaker, Rate Limit를 통합 적용
    """

    def __init__(self, config: Optional[ErrorHandlerConfig] = None):
        self.config = config or ErrorHandlerConfig()

        # 핸들러 초기화
        self.retry_handler = None
        if self.config.retry_config:
            self.retry_handler = RetryHandler(self.config.retry_config)

        self.circuit_breaker = None
        if self.config.circuit_breaker_config:
            self.circuit_breaker = CircuitBreaker(self.config.circuit_breaker_config)

        self.rate_limiter = None
        if self.config.rate_limit_config:
            self.rate_limiter = RateLimiter(self.config.rate_limit_config)

        self.error_tracker = get_error_tracker() if self.config.enable_tracking else None

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        에러 핸들링이 적용된 함수 호출

        적용 순서: Rate Limit -> Circuit Breaker -> Retry

        Args:
            func: 실행할 함수
            *args, **kwargs: 함수 인자

        Returns:
            함수 실행 결과
        """

        def wrapped_func():
            result = func(*args, **kwargs)
            return result

        try:
            # Rate Limit 적용
            if self.rate_limiter:

                def wrapped_func_rl():
                    return self.rate_limiter.call(wrapped_func)
            else:
                wrapped_func_rl = wrapped_func

            # Circuit Breaker 적용
            if self.circuit_breaker:

                def wrapped_func_cb():
                    return self.circuit_breaker.call(wrapped_func_rl)
            else:
                wrapped_func_cb = wrapped_func_rl

            # Retry 적용
            if self.retry_handler:
                result = self.retry_handler.execute(wrapped_func_cb)
            else:
                result = wrapped_func_cb()

            return result

        except Exception as e:
            # 에러 추적
            if self.error_tracker:
                self.error_tracker.record(e)
            raise

    def get_status(self) -> Dict[str, Any]:
        """현재 상태 조회"""
        status = {}

        if self.circuit_breaker:
            status["circuit_breaker"] = self.circuit_breaker.get_state()

        if self.rate_limiter:
            status["rate_limiter"] = self.rate_limiter.get_status()

        if self.error_tracker:
            status["errors"] = self.error_tracker.get_error_summary()

        return status


def with_error_handling(
    max_retries: int = 3, failure_threshold: int = 5, max_calls: int = 10, time_window: float = 60.0
):
    """
    통합 에러 핸들링 데코레이터

    Example:
        @with_error_handling(max_retries=5, failure_threshold=10)
        def api_call():
            ...
    """
    config = ErrorHandlerConfig(
        retry_config=RetryConfig(max_retries=max_retries),
        circuit_breaker_config=CircuitBreakerConfig(failure_threshold=failure_threshold),
        rate_limit_config=RateLimitConfig(max_calls=max_calls, time_window=time_window),
    )
    handler = ErrorHandler(config)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return handler.call(func, *args, **kwargs)

        return wrapper

    return decorator


# ===== Timeout Handler =====


def timeout(seconds: float):
    """
    타임아웃 데코레이터

    Example:
        @timeout(30.0)
        def slow_function():
            ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function timed out after {seconds}s")

            # Set alarm
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))

            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            return result

        return wrapper

    return decorator


# ===== Fallback Decorator =====


def fallback(fallback_func: Optional[Callable] = None, fallback_value: Optional[Any] = None):
    """
    Fallback 데코레이터

    Example:
        @fallback(fallback_value="Default response")
        def api_call():
            ...
    """
    handler = FallbackHandler(fallback_func=fallback_func, fallback_value=fallback_value)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return handler.call(func, *args, **kwargs)

        return wrapper

    return decorator


# ===== Exports for Backward Compatibility =====

__all__ = [
    # Exceptions
    "LLMKitError",
    "ProviderError",
    "RateLimitError",
    "TimeoutError",
    "ValidationError",
    "CircuitBreakerError",
    "MaxRetriesExceededError",
    # Retry
    "RetryStrategy",
    "RetryConfig",
    "RetryHandler",
    "retry",
    # Circuit Breaker
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "circuit_breaker",
    # Rate Limiter
    "RateLimitConfig",
    "RateLimiter",
    "AsyncTokenBucket",
    "rate_limit",
    # Error Tracker
    "ErrorRecord",
    "ErrorTracker",
    "get_error_tracker",
    "FallbackHandler",
    "ProductionErrorSanitizer",
    "sanitize_error_message",
    "create_safe_error_response",
    # Combined Handler
    "ErrorHandlerConfig",
    "ErrorHandler",
    "with_error_handling",
    # Utilities
    "timeout",
    "fallback",
]
