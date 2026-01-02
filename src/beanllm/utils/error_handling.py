"""
beanllm.error_handling - Advanced Error Handling
고급 에러 처리 시스템

이 모듈은 프로덕션급 에러 처리를 제공합니다.
"""

import asyncio
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

# ===== Exceptions =====


class LLMKitError(Exception):
    """beanllm 베이스 예외"""

    pass


class ProviderError(LLMKitError):
    """프로바이더 에러"""

    pass


class RateLimitError(ProviderError):
    """Rate limit 에러"""

    pass


class TimeoutError(LLMKitError):
    """Timeout 에러"""

    pass


class ValidationError(LLMKitError):
    """검증 에러"""

    pass


class CircuitBreakerError(LLMKitError):
    """Circuit breaker open 에러"""

    pass


class MaxRetriesExceededError(LLMKitError):
    """최대 재시도 횟수 초과"""

    pass


# ===== Retry Logic =====


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


# ===== Circuit Breaker =====


class CircuitState(Enum):
    """Circuit breaker 상태"""

    CLOSED = "closed"  # 정상 동작
    OPEN = "open"  # 차단됨
    HALF_OPEN = "half_open"  # 복구 테스트 중


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker 설정"""

    failure_threshold: int = 5  # 실패 임계값
    success_threshold: int = 2  # 성공 임계값 (HALF_OPEN)
    timeout: float = 60.0  # OPEN 상태 유지 시간
    window_size: int = 10  # 슬라이딩 윈도우 크기


class CircuitBreaker:
    """
    Circuit Breaker 패턴 구현

    연속된 실패 발생 시 요청을 자동으로 차단하여
    cascading failure 방지
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.recent_calls = deque(maxlen=self.config.window_size)
        self._lock = threading.Lock()

    def _should_attempt_reset(self) -> bool:
        """OPEN -> HALF_OPEN 전환 여부"""
        if self.state != CircuitState.OPEN:
            return False

        if self.last_failure_time is None:
            return False

        elapsed = time.time() - self.last_failure_time
        return elapsed >= self.config.timeout

    def _record_success(self):
        """성공 기록"""
        with self._lock:
            self.recent_calls.append(True)

            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1

                if self.success_count >= self.config.success_threshold:
                    # 복구 성공 -> CLOSED
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0

            elif self.state == CircuitState.CLOSED:
                # 실패 카운트 감소
                self.failure_count = max(0, self.failure_count - 1)

    def _record_failure(self):
        """실패 기록"""
        with self._lock:
            self.recent_calls.append(False)
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                # HALF_OPEN 중 실패 -> 다시 OPEN
                self.state = CircuitState.OPEN
                self.success_count = 0

            elif self.state == CircuitState.CLOSED:
                # 임계값 초과 -> OPEN
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Circuit breaker를 통한 함수 호출

        Args:
            func: 실행할 함수
            *args, **kwargs: 함수 인자

        Returns:
            함수 실행 결과

        Raises:
            CircuitBreakerError: Circuit이 OPEN 상태일 때
        """
        with self._lock:
            # OPEN -> HALF_OPEN 전환 시도
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0

            # OPEN 상태면 차단
            if self.state == CircuitState.OPEN:
                raise CircuitBreakerError(
                    f"Circuit breaker is OPEN. Wait {self.config.timeout}s before retry."
                )

        # 함수 실행
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result

        except Exception:
            self._record_failure()
            raise

    def get_state(self) -> Dict[str, Any]:
        """현재 상태 조회"""
        with self._lock:
            success_rate = 0.0
            if self.recent_calls:
                success_rate = sum(self.recent_calls) / len(self.recent_calls)

            return {
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "success_rate": success_rate,
                "recent_calls": len(self.recent_calls),
            }

    def reset(self):
        """상태 초기화"""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.recent_calls.clear()


def circuit_breaker(failure_threshold: int = 5, timeout: float = 60.0):
    """
    Circuit breaker 데코레이터

    Example:
        @circuit_breaker(failure_threshold=5, timeout=60)
        def api_call():
            ...
    """
    config = CircuitBreakerConfig(failure_threshold=failure_threshold, timeout=timeout)
    breaker = CircuitBreaker(config)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)

        return wrapper

    return decorator


# ===== Rate Limiter =====


@dataclass
class RateLimitConfig:
    """Rate limit 설정"""

    max_calls: int = 10  # 최대 호출 횟수
    time_window: float = 60.0  # 시간 윈도우 (초)


class RateLimiter:
    """
    Rate Limiter

    일정 시간 내 최대 호출 횟수 제한
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.calls = deque()
        self._lock = threading.Lock()

    def _clean_old_calls(self):
        """오래된 호출 기록 제거"""
        now = time.time()
        cutoff = now - self.config.time_window

        while self.calls and self.calls[0] < cutoff:
            self.calls.popleft()

    def _is_allowed(self) -> bool:
        """호출 허용 여부"""
        self._clean_old_calls()
        return len(self.calls) < self.config.max_calls

    def _wait_time(self) -> float:
        """대기 시간 계산"""
        if not self.calls:
            return 0.0

        oldest_call = self.calls[0]
        elapsed = time.time() - oldest_call
        remaining = self.config.time_window - elapsed

        return max(0.0, remaining)

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Rate limit이 적용된 함수 호출

        Args:
            func: 실행할 함수
            *args, **kwargs: 함수 인자

        Returns:
            함수 실행 결과

        Raises:
            RateLimitError: Rate limit 초과
        """
        with self._lock:
            if not self._is_allowed():
                wait_time = self._wait_time()
                raise RateLimitError(f"Rate limit exceeded. Wait {wait_time:.2f}s before retry.")

            # 호출 기록
            self.calls.append(time.time())

        # 함수 실행
        return func(*args, **kwargs)

    def wait_and_call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Rate limit 대기 후 함수 호출

        Args:
            func: 실행할 함수
            *args, **kwargs: 함수 인자

        Returns:
            함수 실행 결과
        """
        while True:
            with self._lock:
                if self._is_allowed():
                    self.calls.append(time.time())
                    break

                wait_time = self._wait_time()

            # 대기
            time.sleep(wait_time)

        # 함수 실행
        return func(*args, **kwargs)

    def get_status(self) -> Dict[str, Any]:
        """현재 상태 조회"""
        with self._lock:
            self._clean_old_calls()
            return {
                "current_calls": len(self.calls),
                "max_calls": self.config.max_calls,
                "time_window": self.config.time_window,
                "calls_remaining": self.config.max_calls - len(self.calls),
            }


def rate_limit(max_calls: int = 10, time_window: float = 60.0, wait: bool = False):
    """
    Rate limiter 데코레이터

    Example:
        @rate_limit(max_calls=10, time_window=60, wait=True)
        def api_call():
            ...
    """
    config = RateLimitConfig(max_calls=max_calls, time_window=time_window)
    limiter = RateLimiter(config)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if wait:
                return limiter.wait_and_call(func, *args, **kwargs)
            else:
                return limiter.call(func, *args, **kwargs)

        return wrapper

    return decorator


# ===== Async Token Bucket Rate Limiter =====


class AsyncTokenBucket:
    """
    비동기 Token Bucket Rate Limiter

    Token Bucket 알고리즘을 사용한 비동기 Rate Limiter
    - 버스트 허용: 토큰이 축적되면 짧은 시간에 많은 요청 처리 가능
    - 평균 속도 제어: 장기적으로는 평균 속도 유지
    - Semaphore보다 더 유연한 제어
    """

    def __init__(self, rate: float = 1.0, capacity: float = 20.0):
        """
        Args:
            rate: 평균 속도 (토큰/초)
            capacity: 버스트 용량 (최대 토큰 수)
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, cost: float = 1.0) -> bool:
        """
        토큰 획득 시도 (대기하지 않음)

        Args:
            cost: 필요한 토큰 수

        Returns:
            True: 토큰 획득 성공, False: 토큰 부족
        """
        async with self._lock:
            self._refill_tokens()
            if self.tokens >= cost:
                self.tokens -= cost
                return True
            return False

    async def wait(self, cost: float = 1.0):
        """
        토큰이 충분할 때까지 대기

        Args:
            cost: 필요한 토큰 수
        """
        while True:
            async with self._lock:
                self._refill_tokens()
                if self.tokens >= cost:
                    self.tokens -= cost
                    return

                # 필요한 토큰 계산
                needed = cost - self.tokens
                wait_time = needed / self.rate
                if wait_time > 0:
                    await asyncio.sleep(min(wait_time, 1.0))  # 최대 1초씩 대기
                else:
                    await asyncio.sleep(0.01)  # 짧은 대기

    def _refill_tokens(self):
        """토큰 충전"""
        now = time.time()
        delta_t = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + self.rate * delta_t)
        self.last_update = now

    def get_status(self) -> Dict[str, Any]:
        """현재 상태 조회"""
        return {
            "tokens": self.tokens,
            "capacity": self.capacity,
            "rate": self.rate,
            "available": self.tokens,
        }


# ===== Fallback Handler =====


class FallbackHandler:
    """
    Fallback 핸들러

    에러 발생 시 대체 전략 실행
    """

    def __init__(
        self,
        fallback_func: Optional[Callable] = None,
        fallback_value: Optional[Any] = None,
        raise_on_fallback: bool = False,
    ):
        self.fallback_func = fallback_func
        self.fallback_value = fallback_value
        self.raise_on_fallback = raise_on_fallback

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Fallback이 적용된 함수 호출

        Args:
            func: 실행할 함수
            *args, **kwargs: 함수 인자

        Returns:
            함수 실행 결과 또는 fallback 값
        """
        try:
            return func(*args, **kwargs)

        except Exception as e:
            if self.raise_on_fallback:
                raise

            # Fallback 전략 실행
            if self.fallback_func:
                return self.fallback_func(e, *args, **kwargs)
            else:
                return self.fallback_value


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


# ===== Error Tracker =====


@dataclass
class ErrorRecord:
    """에러 기록"""

    timestamp: float
    error_type: str
    error_message: str
    traceback: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ErrorTracker:
    """
    에러 추적기

    에러 발생을 기록하고 분석
    """

    def __init__(self, max_records: int = 1000):
        self.max_records = max_records
        self.errors = deque(maxlen=max_records)
        self._lock = threading.Lock()

    def record(self, exception: Exception, metadata: Optional[Dict[str, Any]] = None):
        """에러 기록"""
        import traceback as tb

        with self._lock:
            record = ErrorRecord(
                timestamp=time.time(),
                error_type=type(exception).__name__,
                error_message=str(exception),
                traceback=tb.format_exc(),
                metadata=metadata or {},
            )
            self.errors.append(record)

    def get_recent_errors(self, n: int = 10) -> List[ErrorRecord]:
        """최근 에러 조회"""
        with self._lock:
            return list(self.errors)[-n:]

    def get_error_summary(self) -> Dict[str, Any]:
        """에러 요약 통계"""
        with self._lock:
            if not self.errors:
                return {"total_errors": 0, "error_types": {}, "error_rate": 0.0}

            # 에러 타입별 카운트
            type_counts = {}
            for error in self.errors:
                error_type = error.error_type
                type_counts[error_type] = type_counts.get(error_type, 0) + 1

            # 에러율 계산 (최근 1시간)
            now = time.time()
            recent_errors = sum(1 for e in self.errors if now - e.timestamp <= 3600)

            return {
                "total_errors": len(self.errors),
                "error_types": type_counts,
                "recent_errors_1h": recent_errors,
                "most_common_error": (
                    max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None
                ),
            }

    def clear(self):
        """에러 기록 초기화"""
        with self._lock:
            self.errors.clear()


# 전역 에러 트래커
_global_error_tracker = ErrorTracker()


def get_error_tracker() -> ErrorTracker:
    """전역 에러 트래커 가져오기"""
    return _global_error_tracker


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
            import signal

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


# ===== Production Error Sanitization =====


import re
import traceback as tb_module
from typing import Pattern


class ProductionErrorSanitizer:
    """
    프로덕션 환경용 에러 메시지 정제기

    민감한 정보를 제거하여 안전한 에러 메시지를 생성합니다:
    - API 키, 비밀번호 패턴 마스킹
    - 파일 경로 제거/축약
    - 스택 트레이스 간소화
    - 데이터베이스 스키마 정보 제거
    - IP 주소, 포트 번호 마스킹

    Security Benefits:
        - API 키 노출 방지
        - 내부 파일 구조 숨김
        - 데이터베이스 스키마 보호
        - 네트워크 토폴로지 보호
    """

    # 민감 정보 패턴
    PATTERNS: Dict[str, Pattern] = {
        # API 키 패턴 (예: sk-..., api_key_..., token_...)
        "api_key": re.compile(
            r"(api[_-]?key|token|secret|password|passwd|pwd)['\"\s:=]+([a-zA-Z0-9_\-./]{10,})",
            re.IGNORECASE,
        ),
        # 환경변수 패턴 (예: OPENAI_API_KEY=sk-...)
        "env_var": re.compile(
            r"([A-Z_]+_(?:API_KEY|TOKEN|SECRET|PASSWORD))['\"\s:=]+([a-zA-Z0-9_\-./]{10,})"
        ),
        # Bearer 토큰
        "bearer": re.compile(r"Bearer\s+([a-zA-Z0-9_\-./]{10,})", re.IGNORECASE),
        # 절대 파일 경로 (Unix/Windows)
        "abs_path": re.compile(r"(/[a-zA-Z0-9_./\-]+/[a-zA-Z0-9_./\-]+|[C-Z]:\\[^\s]+)"),
        # IP 주소
        "ipv4": re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
        # 포트 번호 포함 주소
        "host_port": re.compile(r"(localhost|127\.0\.0\.1|0\.0\.0\.0):(\d{2,5})"),
        # 데이터베이스 연결 문자열
        "db_conn": re.compile(
            r"(postgresql|mysql|mongodb)://([^:]+):([^@]+)@([^:/]+)(:\d+)?(/[^\s]+)?",
            re.IGNORECASE,
        ),
        # SQL 테이블/컬럼명
        "sql_schema": re.compile(r"\b(table|column|schema)\s+['\"]?([a-zA-Z0-9_]+)['\"]?", re.IGNORECASE),
    }

    # 마스킹 문자열
    MASK_STR = "***"
    MASK_PATH = "[PATH]"
    MASK_IP = "[IP]"
    MASK_PORT = "[PORT]"
    MASK_DB = "[DB_CONN]"

    @classmethod
    def sanitize_message(cls, message: str, production: bool = True) -> str:
        """
        에러 메시지 정제

        Args:
            message: 원본 에러 메시지
            production: 프로덕션 모드 (기본: True)

        Returns:
            정제된 에러 메시지

        Example:
            >>> ProductionErrorSanitizer.sanitize_message(
            ...     "API key sk-1234567890 failed at /home/user/app/config.py:42"
            ... )
            'API key *** failed at [PATH]'
        """
        if not production:
            return message

        sanitized = message

        # API 키/토큰 마스킹
        sanitized = cls.PATTERNS["api_key"].sub(rf"\1={cls.MASK_STR}", sanitized)
        sanitized = cls.PATTERNS["env_var"].sub(rf"\1={cls.MASK_STR}", sanitized)
        sanitized = cls.PATTERNS["bearer"].sub(f"Bearer {cls.MASK_STR}", sanitized)

        # 데이터베이스 연결 문자열 마스킹
        sanitized = cls.PATTERNS["db_conn"].sub(cls.MASK_DB, sanitized)

        # 파일 경로 마스킹
        sanitized = cls.PATTERNS["abs_path"].sub(cls.MASK_PATH, sanitized)

        # IP 주소 마스킹 (localhost 제외)
        sanitized = cls.PATTERNS["ipv4"].sub(
            lambda m: m.group(0) if m.group(0).startswith("127.") else cls.MASK_IP, sanitized
        )

        # 포트 번호 마스킹
        sanitized = cls.PATTERNS["host_port"].sub(rf"\1:{cls.MASK_PORT}", sanitized)

        # SQL 스키마 정보 마스킹
        sanitized = cls.PATTERNS["sql_schema"].sub(rf"\1 {cls.MASK_STR}", sanitized)

        return sanitized

    @classmethod
    def sanitize_traceback(cls, traceback_str: str, production: bool = True, max_frames: int = 3) -> str:
        """
        스택 트레이스 정제

        Args:
            traceback_str: 원본 트레이스백 문자열
            production: 프로덕션 모드 (기본: True)
            max_frames: 표시할 최대 프레임 수 (프로덕션 모드)

        Returns:
            정제된 트레이스백

        Example:
            >>> ProductionErrorSanitizer.sanitize_traceback(
            ...     "File '/home/user/app.py', line 42..."
            ... )
            'File [PATH], line 42...'
        """
        if not production:
            return traceback_str

        # 파일 경로 마스킹
        sanitized = cls.PATTERNS["abs_path"].sub(cls.MASK_PATH, traceback_str)

        # 프로덕션 모드: 스택 프레임 수 제한
        lines = sanitized.split("\n")
        if len(lines) > max_frames * 2:  # 각 프레임은 보통 2줄
            # 처음 몇 프레임만 유지
            sanitized = "\n".join(lines[: max_frames * 2] + ["  ... (truncated for security)"])

        return sanitized

    @classmethod
    def create_safe_error(cls, exception: Exception, production: bool = True) -> Dict[str, Any]:
        """
        안전한 에러 응답 생성

        Args:
            exception: 원본 예외
            production: 프로덕션 모드 (기본: True)

        Returns:
            안전한 에러 정보 딕셔너리

        Example:
            >>> try:
            ...     raise ValueError("API key sk-123 is invalid")
            ... except Exception as e:
            ...     safe_error = ProductionErrorSanitizer.create_safe_error(e)
            ...     print(safe_error["message"])
            'API key *** is invalid'
        """
        error_type = type(exception).__name__
        error_message = str(exception)

        # 메시지 정제
        safe_message = cls.sanitize_message(error_message, production)

        result = {
            "error_type": error_type,
            "message": safe_message,
            "production": production,
        }

        # 스택 트레이스 (프로덕션에서는 제한적)
        if production:
            # 프로덕션: 간소화된 트레이스
            traceback_str = tb_module.format_exc()
            result["traceback"] = cls.sanitize_traceback(traceback_str, production, max_frames=2)
        else:
            # 개발: 전체 트레이스
            result["traceback"] = tb_module.format_exc()

        return result


def sanitize_error_message(message: str, production: bool = True) -> str:
    """
    에러 메시지 정제 (헬퍼 함수)

    Args:
        message: 원본 에러 메시지
        production: 프로덕션 모드

    Returns:
        정제된 에러 메시지
    """
    return ProductionErrorSanitizer.sanitize_message(message, production)


def create_safe_error_response(exception: Exception, production: bool = True) -> Dict[str, Any]:
    """
    안전한 에러 응답 생성 (헬퍼 함수)

    Args:
        exception: 원본 예외
        production: 프로덕션 모드

    Returns:
        안전한 에러 정보
    """
    return ProductionErrorSanitizer.create_safe_error(exception, production)
