"""
인메모리 Cache (기존 LRUCache 래핑)
"""

import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Generic, Optional, TypeVar, cast

from beanllm.infrastructure.distributed.interfaces import CacheInterface

try:
    from beanllm.utils.core.cache import LRUCache

    _REAL_LRUCACHE = True
except ImportError:
    _REAL_LRUCACHE = False

    class LRUCache:  # type: ignore[no-redef]
        """Fallback LRUCache with per-item TTL support."""

        def __init__(self, max_size: int = 1000, ttl: Optional[int] = None):
            self.max_size = max_size
            self.ttl = ttl
            self._cache: OrderedDict = OrderedDict()
            self._lock = threading.RLock()

        def get(self, key):
            with self._lock:
                if key not in self._cache:
                    return None
                value, timestamp, item_ttl = self._cache[key]
                effective_ttl = item_ttl if item_ttl is not None else self.ttl
                if effective_ttl and time.time() - timestamp > effective_ttl:
                    del self._cache[key]
                    return None
                self._cache.move_to_end(key)
                return value

        def set(self, key, value, ttl: Optional[int] = None):
            with self._lock:
                if len(self._cache) >= self.max_size:
                    self._cache.popitem(last=False)
                self._cache[key] = (value, time.time(), ttl)

        def delete(self, key):
            with self._lock:
                self._cache.pop(key, None)

        def clear(self):
            with self._lock:
                self._cache.clear()


K = TypeVar("K")
V = TypeVar("V")


class InMemoryCache(CacheInterface[K, V], Generic[K, V]):
    """
    인메모리 Cache — per-item TTL 완전 지원.

    기존 LRUCache를 래핑하되, underlying 구현이 per-item TTL을 지원하지 않는 경우
    별도의 TTL 오버레이(dict + lock)로 보완한다.

    Example::

        cache = InMemoryCache(max_size=1000, ttl=3600)

        # 기본 TTL(3600s) 적용
        await cache.set("session:abc", user_data)

        # 이 항목만 30초 TTL
        await cache.set("token:xyz", token, ttl=30)
    """

    def __init__(self, max_size: int = 1000, ttl: Optional[int] = None):
        """
        Args:
            max_size: 최대 캐시 크기
            ttl: 기본 Time-to-Live (초). None이면 만료 없음.
                 set(ttl=…) 호출 시 해당 항목에만 적용되는 값이 우선.
        """
        self._cache: Any = LRUCache(max_size=max_size, ttl=ttl)
        self._default_ttl = ttl

        # 실제 LRUCache가 per-item TTL을 지원하지 않으면 오버레이 사용
        if _REAL_LRUCACHE:
            self._ttl_overlay: Optional[Dict[Any, tuple]] = {}
            self._overlay_lock = threading.Lock()
        else:
            self._ttl_overlay = None

    def _check_overlay_expiry(self, key: K) -> bool:
        """TTL 오버레이에서 키가 만료됐는지 확인. 만료 시 True 반환 후 제거."""
        if self._ttl_overlay is None:
            return False
        with self._overlay_lock:
            if key not in self._ttl_overlay:
                return False
            expiry_ts = self._ttl_overlay[key]
            if time.time() > expiry_ts:
                self._ttl_overlay.pop(key, None)
                return True
        return False

    async def get(self, key: K) -> Optional[V]:
        """값 조회. TTL 만료된 항목은 None 반환."""
        if self._check_overlay_expiry(key):
            self._cache.delete(key)
            return None
        return cast(Optional[V], self._cache.get(key))

    async def set(self, key: K, value: V, ttl: Optional[int] = None) -> None:
        """
        값 저장.

        Args:
            key: 캐시 키
            value: 저장할 값
            ttl: 이 항목의 만료 시간(초). None이면 생성자 기본값 사용.
        """
        if _REAL_LRUCACHE:
            # 실제 LRUCache는 per-item TTL 미지원 → 오버레이로 관리
            self._cache.set(key, value)
            effective_ttl = ttl if ttl is not None else self._default_ttl
            if effective_ttl is not None and self._ttl_overlay is not None:
                with self._overlay_lock:
                    self._ttl_overlay[key] = time.time() + effective_ttl
        else:
            # fallback LRUCache는 per-item TTL 지원
            self._cache.set(key, value, ttl=ttl)

    async def delete(self, key: K) -> None:
        """값 삭제"""
        self._cache.delete(key)
        if self._ttl_overlay is not None:
            with self._overlay_lock:
                self._ttl_overlay.pop(key, None)

    async def clear(self) -> None:
        """모든 캐시 삭제"""
        self._cache.clear()
        if self._ttl_overlay is not None:
            with self._overlay_lock:
                self._ttl_overlay.clear()
