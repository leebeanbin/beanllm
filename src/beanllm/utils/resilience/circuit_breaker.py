"""
beanllm.utils.resilience.circuit_breaker - Circuit Breaker Pattern
서킷 브레이커 패턴

이 모듈은 Circuit Breaker 패턴을 구현하여 cascading failure를 방지합니다:
- CLOSED: 정상 동작
- OPEN: 차단됨 (실패 임계값 초과)
- HALF_OPEN: 복구 테스트 중
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Optional

from ..exceptions import CircuitBreakerError


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
