"""
캐시 래퍼 (동기/비동기 호환)

기존 동기 코드와 분산 캐시(비동기)를 연결하는 래퍼
"""

import asyncio
from typing import Generic, Optional, TypeVar

from .factory import get_cache
from .interfaces import CacheInterface

K = TypeVar("K")
V = TypeVar("V")


class SyncCacheWrapper(Generic[K, V]):
    """
    동기 캐시 래퍼

    분산 캐시(비동기)를 동기 인터페이스로 래핑
    기존 코드와의 호환성 유지
    """

    def __init__(self, max_size: int = 1000, ttl: Optional[int] = None):
        """
        Args:
            max_size: 최대 캐시 크기 (인메모리 모드에서만 사용)
            ttl: Time-to-Live (초)
        """
        self._async_cache: CacheInterface[K, V] = get_cache(max_size=max_size, ttl=ttl)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """이벤트 루프 가져오기 (없으면 생성)"""
        if self._loop is None:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

    def get(self, key: K) -> Optional[V]:
        """값 조회 (동기)"""
        loop = self._get_loop()
        if loop.is_running():
            # 이미 실행 중인 루프가 있으면 새 태스크로 실행
            # 하지만 동기 컨텍스트에서는 await 불가능하므로 None 반환
            # 실제로는 비동기 코드에서 사용하는 것이 권장됨
            return None
        else:
            return loop.run_until_complete(self._async_cache.get(key))

    def set(self, key: K, value: V, ttl: Optional[int] = None):
        """값 저장 (동기)"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 이미 실행 중인 루프가 있으면 건너뛰기
                return
            else:
                loop.run_until_complete(self._async_cache.set(key, value, ttl=ttl))
        except RuntimeError:
            # 루프가 없으면 새로 생성
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._async_cache.set(key, value, ttl=ttl))
            finally:
                loop.close()

    def delete(self, key: K):
        """값 삭제 (동기)"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return
            else:
                loop.run_until_complete(self._async_cache.delete(key))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._async_cache.delete(key))
            finally:
                loop.close()

    def clear(self):
        """모든 캐시 삭제 (동기)"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return
            else:
                loop.run_until_complete(self._async_cache.clear())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._async_cache.clear())
            finally:
                loop.close()

    def stats(self) -> dict:
        """캐시 통계 (기존 인터페이스 호환)"""
        # 분산 캐시는 통계를 제공하지 않으므로 기본값 반환
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

    def shutdown(self):
        """캐시 정리 (기존 인터페이스 호환)"""
        # 분산 캐시는 별도 정리 불필요
        pass


def get_distributed_cache(max_size: int = 1000, ttl: Optional[int] = None) -> SyncCacheWrapper:
    """
    분산 캐시 래퍼 반환 (동기 인터페이스)

    기존 코드와의 호환성을 위해 동기 인터페이스 제공
    내부적으로는 분산 캐시(비동기) 사용

    Args:
        max_size: 최대 캐시 크기
        ttl: Time-to-Live (초)

    Returns:
        SyncCacheWrapper 인스턴스
    """
    return SyncCacheWrapper(max_size=max_size, ttl=ttl)

