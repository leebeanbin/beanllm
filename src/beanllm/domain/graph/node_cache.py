"""
NodeCache - 노드 캐시

Updated to use generic LRUCache with TTL and automatic cleanup
"""

import hashlib
import json
from typing import Any, Dict, Optional

try:
    from beanllm.utils.cache import LRUCache
except ImportError:
    # Fallback: simple dict-based cache without TTL
    class LRUCache:
        def __init__(self, max_size: int = 1000, ttl: Optional[int] = None, **kwargs):
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


from beanllm.utils.logger import get_logger

from .graph_state import GraphState

logger = get_logger(__name__)


class NodeCache:
    """
    노드 캐시 (그래프 노드 실행 결과 캐싱)

    Features (Updated):
        - ✅ Proper LRU eviction (least recently used)
        - ✅ TTL (Time-to-Live) expiration
        - ✅ Automatic background cleanup of expired entries
        - ✅ Thread-safe operations
        - ✅ Cache statistics (hits, misses, evictions)

    같은 입력 state에 대해 이전 노드 실행 결과를 재사용하여 성능 향상

    Example:
        ```python
        from beanllm.domain.graph import NodeCache

        # 캐시 생성 (30분 TTL)
        cache = NodeCache(max_size=1000, ttl=1800)

        # 노드 실행 전 캐시 확인
        cached_result = cache.get("process_node", current_state)
        if cached_result is not None:
            # 캐시 히트 - 이전 결과 사용
            return cached_result

        # 캐시 미스 - 노드 실행 후 캐시에 저장
        result = execute_node(current_state)
        cache.set("process_node", current_state, result)

        # 통계 확인
        stats = cache.get_stats()
        print(f"Hit rate: {stats['hit_rate']:.2%}")

        # 종료 시 cleanup 스레드 정리
        cache.shutdown()
        ```
    """

    def __init__(
        self, max_size: int = 1000, ttl: Optional[int] = None, cleanup_interval: int = 60
    ):
        """
        Args:
            max_size: 최대 캐시 크기 (default: 1000)
            ttl: 캐시 유지 시간 초 (default: None = 무제한)
            cleanup_interval: 자동 정리 주기 초 (default: 60초)
        """
        # Use generic LRUCache with automatic cleanup
        self._cache: LRUCache[str, Any] = LRUCache(
            max_size=max_size,
            ttl=ttl,
            cleanup_interval=cleanup_interval,
        )
        self.max_size = max_size
        self.ttl = ttl

    def get_key(self, node_name: str, state: GraphState) -> str:
        """
        캐시 키 생성 (노드 이름 + state 해시)

        Args:
            node_name: 노드 이름
            state: 그래프 state

        Returns:
            캐시 키 문자열
        """
        # state를 JSON으로 직렬화하여 해시
        state_json = json.dumps(state.data, sort_keys=True)
        hash_value = hashlib.md5(state_json.encode()).hexdigest()
        return f"{node_name}:{hash_value}"

    def get(self, node_name: str, state: GraphState) -> Optional[Any]:
        """
        캐시에서 가져오기

        Args:
            node_name: 노드 이름
            state: 그래프 state

        Returns:
            캐시된 결과 또는 None (캐시 미스 또는 만료)
        """
        key = self.get_key(node_name, state)
        result = self._cache.get(key)

        if result is not None:
            logger.debug(f"Cache hit for {node_name}")
        else:
            logger.debug(f"Cache miss for {node_name}")

        return result

    def set(self, node_name: str, state: GraphState, result: Any):
        """
        캐시에 저장

        Args:
            node_name: 노드 이름
            state: 그래프 state
            result: 노드 실행 결과
        """
        key = self.get_key(node_name, state)
        self._cache.set(key, result)
        logger.debug(f"Cached result for {node_name}")

    def clear(self):
        """캐시 초기화 (모든 항목 삭제)"""
        self._cache.clear()

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
