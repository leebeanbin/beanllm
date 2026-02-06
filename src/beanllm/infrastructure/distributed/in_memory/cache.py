"""
인메모리 Cache (기존 LRUCache 래핑)
"""

from typing import Generic, Optional, TypeVar

from beanllm.infrastructure.distributed.interfaces import CacheInterface

try:
    from beanllm.utils.core.cache import LRUCache
except ImportError:
    # Fallback if not available
    import threading
    import time
    from collections import OrderedDict

    class LRUCache:
        def __init__(self, max_size: int = 1000, ttl: Optional[int] = None):
            self.max_size = max_size
            self.ttl = ttl
            self._cache: OrderedDict = OrderedDict()
            self._lock = threading.RLock()

        def get(self, key):
            with self._lock:
                if key not in self._cache:
                    return None
                value, timestamp = self._cache[key]
                if self.ttl and time.time() - timestamp > self.ttl:
                    del self._cache[key]
                    return None
                # Move to end (LRU)
                self._cache.move_to_end(key)
                return value

        def set(self, key, value, ttl: Optional[int] = None):
            with self._lock:
                if len(self._cache) >= self.max_size:
                    self._cache.popitem(last=False)  # Remove oldest
                self._cache[key] = (value, time.time())

        def delete(self, key):
            with self._lock:
                if key in self._cache:
                    del self._cache[key]

        def clear(self):
            with self._lock:
                self._cache.clear()


K = TypeVar("K")
V = TypeVar("V")


class InMemoryCache(CacheInterface[K, V], Generic[K, V]):
    """
    인메모리 Cache

    기존 LRUCache를 래핑하여 인터페이스 구현
    """

    def __init__(self, max_size: int = 1000, ttl: Optional[int] = None):
        """
        Args:
            max_size: 최대 캐시 크기
            ttl: Time-to-Live (초), None이면 만료 없음
        """
        self._cache = LRUCache(max_size=max_size, ttl=ttl)

    async def get(self, key: K) -> Optional[V]:
        """값 조회"""
        return self._cache.get(key)

    async def set(self, key: K, value: V, ttl: Optional[int] = None):
        """값 저장"""
        # LRUCache는 생성자에서 ttl을 받지만, 개별 항목별 ttl은 지원하지 않음
        # 여기서는 기본 ttl 사용
        self._cache.set(key, value)

    async def delete(self, key: K):
        """값 삭제"""
        self._cache.delete(key)

    async def clear(self):
        """모든 캐시 삭제"""
        self._cache.clear()
