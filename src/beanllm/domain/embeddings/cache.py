"""
Embeddings Cache - 임베딩 캐시

Updated to use generic LRUCache with automatic TTL cleanup
"""

from typing import Any, Dict, List, Optional

try:
    from beanllm.utils.cache import LRUCache
    from beanllm.utils.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)

    # Fallback to old implementation if LRUCache not available
    import time
    from collections import OrderedDict

    class LRUCache:
        """Fallback implementation"""

        def __init__(self, max_size: int = 1000, ttl: Optional[int] = None, **kwargs):
            self.cache: OrderedDict = OrderedDict()
            self.ttl = ttl
            self.max_size = max_size

        def get(self, key, default=None):
            if key not in self.cache:
                return default
            value, timestamp = self.cache[key]
            if self.ttl and time.time() - timestamp > self.ttl:
                del self.cache[key]
                return default
            self.cache.move_to_end(key)
            return value

        def set(self, key, value):
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            self.cache[key] = (value, time.time())

        def clear(self):
            self.cache.clear()

        def stats(self):
            return {"size": len(self.cache), "max_size": self.max_size, "ttl": self.ttl}

        def shutdown(self):
            pass


logger = get_logger(__name__)


class EmbeddingCache:
    """
    Embedding 캐시: 같은 텍스트의 임베딩을 재사용하여 비용 절감

    Features (Updated):
        - ✅ Proper LRU eviction (least recently used)
        - ✅ TTL (Time-to-Live) expiration
        - ✅ Automatic background cleanup of expired entries
        - ✅ Thread-safe operations
        - ✅ Cache statistics (hits, misses, evictions)

    Example:
        ```python
        from beanllm.domain.embeddings import Embedding, EmbeddingCache

        emb = Embedding(model="text-embedding-3-small")
        cache = EmbeddingCache(ttl=3600, max_size=10000)  # 1시간 TTL, 10000개 최대

        # 첫 번째: API 호출
        vec1 = await emb.embed(["텍스트"], cache=cache)

        # 두 번째: 캐시에서 가져옴 (API 호출 안 함)
        vec2 = await emb.embed(["텍스트"], cache=cache)

        # 캐시 통계 확인
        stats = cache.stats()
        print(f"Hit rate: {stats['hit_rate']:.2%}")

        # 종료 시 cleanup 스레드 정리 (중요!)
        cache.shutdown()
        ```
    """

    def __init__(
        self, ttl: int = 3600, max_size: int = 10000, cleanup_interval: int = 60
    ):
        """
        Args:
            ttl: 캐시 유지 시간 (초, default: 3600 = 1시간)
            max_size: 최대 캐시 항목 수 (default: 10000)
            cleanup_interval: 자동 정리 주기 (초, default: 60초)
        """
        # Use generic LRUCache with automatic cleanup
        self._cache: LRUCache[str, List[float]] = LRUCache(
            max_size=max_size,
            ttl=ttl,
            cleanup_interval=cleanup_interval,
        )
        self.ttl = ttl
        self.max_size = max_size

    def get(self, text: str) -> Optional[List[float]]:
        """
        캐시에서 임베딩 벡터 가져오기

        Args:
            text: 텍스트 (캐시 키)

        Returns:
            임베딩 벡터 또는 None (캐시 미스 또는 만료)
        """
        return self._cache.get(text)

    def set(self, text: str, vector: List[float]):
        """
        캐시에 임베딩 벡터 저장

        Args:
            text: 텍스트 (캐시 키)
            vector: 임베딩 벡터
        """
        self._cache.set(text, vector)

    def clear(self):
        """캐시 비우기 (모든 항목 삭제)"""
        self._cache.clear()

    def stats(self) -> Dict[str, Any]:
        """
        캐시 통계 반환

        Returns:
            Dictionary with:
                - size: 현재 캐시 항목 수
                - max_size: 최대 캐시 항목 수
                - ttl: TTL (초)
                - hits: 캐시 히트 수
                - misses: 캐시 미스 수
                - hit_rate: 히트율 (0.0 ~ 1.0)
                - evictions: LRU 제거 수
                - expirations: TTL 만료 수
        """
        return self._cache.stats()

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
        except Exception:
            pass
