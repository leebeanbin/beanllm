"""
beanllm.utils.resilience - Resilience Patterns
복원력 패턴

이 모듈은 프로덕션급 복원력 패턴을 제공합니다:
- Retry: 자동 재시도
- Circuit Breaker: 장애 차단
- Rate Limiter: 속도 제한
- Error Tracker: 에러 추적 및 보안 정제
"""

# Retry
# Circuit Breaker
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    circuit_breaker,
)

# Error Tracker
from .error_tracker import (
    ErrorRecord,
    ErrorTracker,
    FallbackHandler,
    ProductionErrorSanitizer,
    create_safe_error_response,
    get_error_tracker,
    sanitize_error_message,
)

# Rate Limiter
from .rate_limiter import (
    AsyncTokenBucket,
    RateLimitConfig,
    RateLimiter,
    rate_limit,
)
from .retry import (
    RetryConfig,
    RetryHandler,
    RetryStrategy,
    retry,
)

__all__ = [
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
]
