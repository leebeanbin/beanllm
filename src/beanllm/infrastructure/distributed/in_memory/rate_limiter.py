"""
인메모리 Rate Limiter (기존 AsyncTokenBucket 래핑)
"""

import asyncio
from typing import Dict, Any

from beanllm.infrastructure.distributed.interfaces import RateLimiterInterface

try:
    from beanllm.utils.resilience.rate_limiter import AsyncTokenBucket
except ImportError:
    # Fallback if not available
    class AsyncTokenBucket:
        def __init__(self, rate: float = 1.0, capacity: float = 20.0):
            self.rate = rate
            self.capacity = capacity
            self.tokens = capacity
            self.last_update = asyncio.get_event_loop().time()
            self._lock = asyncio.Lock()

        async def acquire(self, cost: float = 1.0) -> bool:
            async with self._lock:
                self._refill_tokens()
                if self.tokens >= cost:
                    self.tokens -= cost
                    return True
                return False

        async def wait(self, cost: float = 1.0):
            while True:
                async with self._lock:
                    self._refill_tokens()
                    if self.tokens >= cost:
                        self.tokens -= cost
                        return
                    needed = cost - self.tokens
                    wait_time = needed / self.rate
                    await asyncio.sleep(min(wait_time, 1.0))

        def _refill_tokens(self):
            now = asyncio.get_event_loop().time()
            delta_t = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + self.rate * delta_t)
            self.last_update = now

        def get_status(self) -> Dict[str, Any]:
            return {
                "tokens": self.tokens,
                "rate": self.rate,
                "capacity": self.capacity,
            }


class InMemoryRateLimiter(RateLimiterInterface):
    """
    인메모리 Rate Limiter

    기존 AsyncTokenBucket을 래핑하여 인터페이스 구현
    키별로 별도의 TokenBucket 인스턴스 관리
    """

    def __init__(self, default_rate: float = 1.0, default_capacity: float = 20.0):
        """
        Args:
            default_rate: 기본 속도 (토큰/초)
            default_capacity: 기본 용량 (최대 토큰 수)
        """
        self.default_rate = default_rate
        self.default_capacity = default_capacity
        self._buckets: Dict[str, AsyncTokenBucket] = {}
        self._lock = asyncio.Lock()

    def _get_bucket(self, key: str) -> AsyncTokenBucket:
        """키에 해당하는 TokenBucket 반환 (없으면 생성)"""
        if key not in self._buckets:
            self._buckets[key] = AsyncTokenBucket(
                rate=self.default_rate, capacity=self.default_capacity
            )
        return self._buckets[key]

    async def acquire(self, key: str, cost: float = 1.0) -> bool:
        """토큰 획득 시도"""
        bucket = self._get_bucket(key)
        return await bucket.acquire(cost)

    async def wait(self, key: str, cost: float = 1.0):
        """토큰이 충분할 때까지 대기"""
        bucket = self._get_bucket(key)
        await bucket.wait(cost)

    def get_status(self, key: str) -> Dict[str, Any]:
        """현재 상태 조회"""
        bucket = self._get_bucket(key)
        status = bucket.get_status()
        return {
            "key": key,
            **status,
        }

