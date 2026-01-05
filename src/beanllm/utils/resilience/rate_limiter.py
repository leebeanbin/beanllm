"""
beanllm.utils.resilience.rate_limiter - Rate Limiting
속도 제한

이 모듈은 API 호출 속도 제한을 제공합니다:
- 슬라이딩 윈도우 기반 Rate Limiter
- 비동기 Token Bucket Rate Limiter
- 대기 옵션 지원
"""

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, Optional

from ..exceptions import RateLimitError


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
