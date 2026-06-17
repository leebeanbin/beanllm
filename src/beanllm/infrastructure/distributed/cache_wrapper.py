"""
캐시 래퍼 (동기/비동기 호환)

기존 동기 코드와 분산 캐시(비동기)를 연결하는 래퍼
"""

import asyncio
from typing import Generic, Optional, TypeVar

from beanllm.infrastructure.distributed.factory import get_cache
from beanllm.infrastructure.distributed.interfaces import CacheInterface

K = TypeVar("K")
V = TypeVar("V")


def _run_coro(coro):
    """
    동기 컨텍스트에서 코루틴 실행.

    - 이미 실행 중인 루프가 있으면 None을 반환 (blocking 불가)
    - 없으면 asyncio.run()으로 실행 (deprecated get_event_loop() 사용 안 함)
    """
    try:
        asyncio.get_running_loop()
        # 실행 중인 루프 안에서 호출됨 — 동기 blocking 불가
        coro.close()
        return None
    except RuntimeError:
        pass
    return asyncio.run(coro)


class SyncCacheWrapper(Generic[K, V]):
    """
    동기 캐시 래퍼

    분산 캐시(비동기)를 동기 인터페이스로 래핑.
    기존 코드와의 호환성 유지.
    """

    def __init__(self, max_size: int = 1000, ttl: Optional[int] = None):
        """
        Args:
            max_size: 최대 캐시 크기 (인메모리 모드에서만 사용)
            ttl: Time-to-Live (초)
        """
        self._async_cache: CacheInterface[K, V] = get_cache(max_size=max_size, ttl=ttl)

    def get(self, key: K) -> Optional[V]:
        """값 조회 (동기). 실행 중인 루프 안에서 호출 시 None 반환."""
        return _run_coro(self._async_cache.get(key))

    def set(self, key: K, value: V, ttl: Optional[int] = None) -> None:
        """값 저장 (동기). 실행 중인 루프 안에서 호출 시 no-op."""
        _run_coro(self._async_cache.set(key, value, ttl=ttl))

    def delete(self, key: K) -> None:
        """값 삭제 (동기). 실행 중인 루프 안에서 호출 시 no-op."""
        _run_coro(self._async_cache.delete(key))

    def clear(self) -> None:
        """모든 캐시 삭제 (동기). 실행 중인 루프 안에서 호출 시 no-op."""
        _run_coro(self._async_cache.clear())

    def stats(self) -> dict:
        """캐시 통계 (기존 인터페이스 호환)"""
        return {
            "size": 0,
            "max_size": 0,
            "ttl": None,
            "hits": 0,
            "misses": 0,
            "hit_rate": 0.0,
            "evictions": 0,
            "expirations": 0,
        }

    def shutdown(self) -> None:
        """캐시 정리 (기존 인터페이스 호환)"""
        pass


def get_distributed_cache(max_size: int = 1000, ttl: Optional[int] = None) -> SyncCacheWrapper:
    """
    분산 캐시 래퍼 반환 (동기 인터페이스)

    기존 코드와의 호환성을 위해 동기 인터페이스 제공.
    내부적으로는 분산 캐시(비동기) 사용.

    Args:
        max_size: 최대 캐시 크기
        ttl: Time-to-Live (초)

    Returns:
        SyncCacheWrapper 인스턴스
    """
    return SyncCacheWrapper(max_size=max_size, ttl=ttl)
