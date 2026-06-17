"""
인메모리 Rate Limiter (기존 AsyncTokenBucket 래핑)
"""

import time
from collections import OrderedDict
from typing import Any, Dict

from beanllm.infrastructure.distributed.interfaces import RateLimiterInterface

try:
    from beanllm.utils.resilience.rate_limiter import AsyncTokenBucket
except ImportError:
    # Fallback if not available
    import asyncio

    class AsyncTokenBucket:  # type: ignore[no-redef]
        def __init__(self, rate: float = 1.0, capacity: float = 20.0):
            self.rate = rate
            self.capacity = capacity
            self.tokens = capacity
            # time.monotonic() 사용 — asyncio 이벤트 루프 불필요
            self.last_update = time.monotonic()
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
                import asyncio as _asyncio

                await _asyncio.sleep(min(wait_time, 1.0))

        def _refill_tokens(self):
            now = time.monotonic()
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

    기존 AsyncTokenBucket을 래핑하여 인터페이스 구현.
    키별로 별도의 TokenBucket 인스턴스 관리.

    max_buckets 제한으로 OOM 방지: 초과 시 가장 오래 사용하지 않은 키의 bucket 제거.
    """

    def __init__(
        self,
        default_rate: float = 1.0,
        default_capacity: float = 20.0,
        max_buckets: int = 1_000,
    ):
        """
        Args:
            default_rate: 기본 속도 (토큰/초)
            default_capacity: 기본 용량 (최대 토큰 수)
            max_buckets: 동시에 보관할 최대 키 수. 초과 시 LRU 방식으로 제거.
        """
        self.default_rate = default_rate
        self.default_capacity = default_capacity
        self._max_buckets = max_buckets
        # OrderedDict: 삽입/접근 순서 추적으로 LRU eviction 구현
        self._buckets: OrderedDict[str, AsyncTokenBucket] = OrderedDict()

    def _get_bucket(self, key: str) -> AsyncTokenBucket:
        """키에 해당하는 TokenBucket 반환. 없으면 생성하며 LRU 순서 갱신."""
        if key in self._buckets:
            self._buckets.move_to_end(key)
            return self._buckets[key]

        bucket = AsyncTokenBucket(rate=self.default_rate, capacity=self.default_capacity)
        self._buckets[key] = bucket

        if len(self._buckets) > self._max_buckets:
            # 가장 오래 사용하지 않은 bucket 제거
            self._buckets.popitem(last=False)

        return bucket

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
        return {"key": key, **bucket.get_status()}

    @property
    def bucket_count(self) -> int:
        """현재 관리 중인 bucket 수 (모니터링용)"""
        return len(self._buckets)
