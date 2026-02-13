"""
Prompts Cache - 프롬프트 캐시

Updated to use generic LRUCache with TTL and automatic cleanup
"""

import json
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

if TYPE_CHECKING:
    from beanllm.domain.protocols import CacheProtocol

try:
    from beanllm.utils.constants import DEFAULT_CACHE_MAX_SIZE
except ImportError:
    DEFAULT_CACHE_MAX_SIZE = 1000

try:
    from beanllm.utils.core.cache import LRUCache
except ImportError:
    # Fallback: simple dict-based cache without TTL
    class LRUCache:  # type: ignore[no-redef]
        def __init__(
            self, max_size: int = DEFAULT_CACHE_MAX_SIZE, ttl: Optional[int] = None, **kwargs
        ):
            self.cache: Dict = {}
            self.max_size = max_size

        def get(self, key, default=None):
            return self.cache.get(key, default)

        def set(self, key, value):
            if len(self.cache) >= self.max_size:
                first_key = next(iter(self.cache))
                del self.cache[first_key]
            self.cache[key] = value

        def clear(self):
            self.cache.clear()

        def stats(self):
            return {"size": len(self.cache), "max_size": self.max_size}

        def shutdown(self):
            pass


from beanllm.domain.prompts.base import BasePromptTemplate
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class PromptCache:
    """
    프롬프트 캐시 (성능 최적화)

    Features (Updated):
        - ✅ Proper LRU eviction (least recently used)
        - ✅ TTL (Time-to-Live) expiration
        - ✅ Automatic background cleanup of expired entries
        - ✅ Thread-safe operations
        - ✅ Cache statistics (hits, misses, evictions)

    Example:
        ```python
        from beanllm.domain.prompts import PromptCache

        # 캐시 생성 (1시간 TTL)
        cache = PromptCache(max_size=DEFAULT_CACHE_MAX_SIZE, ttl=3600)

        # 캐시에 저장
        cache.set("prompt_key", "formatted prompt text")

        # 캐시에서 가져오기
        cached_value = cache.get("prompt_key")

        # 통계 확인
        stats = cache.get_stats()
        print(f"Hit rate: {stats['hit_rate']:.2%}")

        # 종료 시 cleanup 스레드 정리
        cache.shutdown()
        ```
    """

    def __init__(
        self,
        max_size: int = DEFAULT_CACHE_MAX_SIZE,
        ttl: Optional[int] = None,
        cleanup_interval: int = 60,
        cache: Optional["CacheProtocol"] = None,
    ):
        """
        Args:
            max_size: 최대 캐시 항목 수 (default: 1000)
            ttl: 캐시 유지 시간 초 (default: None = 무제한)
            cleanup_interval: 자동 정리 주기 초 (default: 60초)
            cache: 캐시 구현체 (옵션). None이면 로컬 LRUCache 사용.

        Example:
            # 로컬 캐시 (기본)
            cache = PromptCache()

            # 분산 캐시 (Service layer에서 주입)
            # distributed_cache는 Service layer에서 생성하여 주입
            cache = PromptCache(cache=injected_distributed_cache)
        """
        if cache is not None:
            # 외부에서 주입된 캐시 사용 (분산 캐시 등)
            self._cache: Any = cache
        else:
            # 기본 LRUCache 사용 (인메모리)
            self._cache = LRUCache(
                max_size=max_size,
                ttl=ttl,
                cleanup_interval=cleanup_interval,
            )
        self.max_size = max_size
        self.ttl = ttl

    def get(self, key: str) -> Optional[str]:
        """
        캐시에서 가져오기

        Args:
            key: 캐시 키

        Returns:
            캐시된 값 또는 None (캐시 미스 또는 만료)
        """
        return cast(Optional[str], self._cache.get(key))

    def set(self, key: str, value: str) -> None:
        """
        캐시에 저장

        Args:
            key: 캐시 키
            value: 저장할 값
        """
        self._cache.set(key, value)

    def get_stats(self) -> Dict[str, Any]:
        """
        캐시 통계

        Returns:
            Dictionary with:
                - size: 현재 캐시 항목 수
                - max_size: 최대 캐시 항목 수
                - ttl: TTL (초, None이면 무제한)
                - hits: 캐시 히트 수
                - misses: 캐시 미스 수
                - hit_rate: 히트율 (0.0 ~ 1.0)
                - evictions: LRU 제거 수
                - expirations: TTL 만료 수
        """
        return cast(Dict[str, Any], self._cache.stats())

    def clear(self) -> None:
        """캐시 초기화 (모든 항목 삭제)"""
        self._cache.clear()

    def shutdown(self):
        """
        캐시 정리 및 cleanup 스레드 종료

        Important: 애플리케이션 종료 시 반드시 호출하여 리소스 정리
        """
        self._cache.shutdown()

    def __del__(self):
        """소멸자 - 자동 리소스 정리"""
        try:
            self.shutdown()
        except Exception as e:
            logger.debug(f"Shutdown in destructor failed (safe to ignore): {e}")


# 전역 캐시 인스턴스
_global_cache = PromptCache()


def get_cached_prompt(template: BasePromptTemplate, use_cache: bool = True, **kwargs) -> str:
    """캐시를 사용한 프롬프트 생성"""
    if not use_cache:
        return template.format(**kwargs)

    # 캐시 키 생성
    cache_key = f"{id(template)}:{json.dumps(kwargs, sort_keys=True)}"

    # 캐시 확인
    cached = _global_cache.get(cache_key)
    if cached is not None:
        return cached

    # 생성 및 캐시 저장
    result = template.format(**kwargs)
    _global_cache.set(cache_key, result)

    return result


def get_cache_stats() -> Dict[str, Any]:
    """전역 캐시 통계"""
    return _global_cache.get_stats()


def clear_cache() -> None:
    """전역 캐시 초기화"""
    _global_cache.clear()
