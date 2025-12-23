"""
NodeCache - 노드 캐시
"""

import hashlib
import json
from typing import Any, Dict, Optional

from ...utils.logger import get_logger
from .graph_state import GraphState

logger = get_logger(__name__)


class NodeCache:
    """
    노드 캐시

    같은 입력에 대해 이전 결과 재사용
    """

    def __init__(self, max_size: int = 1000):
        """
        Args:
            max_size: 최대 캐시 크기
        """
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get_key(self, node_name: str, state: GraphState) -> str:
        """캐시 키 생성"""
        # state를 JSON으로 직렬화하여 해시
        state_json = json.dumps(state.data, sort_keys=True)
        hash_value = hashlib.md5(state_json.encode()).hexdigest()
        return f"{node_name}:{hash_value}"

    def get(self, node_name: str, state: GraphState) -> Optional[Any]:
        """캐시에서 가져오기"""
        key = self.get_key(node_name, state)
        if key in self.cache:
            self.hits += 1
            logger.debug(f"Cache hit for {node_name}")
            return self.cache[key]
        else:
            self.misses += 1
            return None

    def set(self, node_name: str, state: GraphState, result: Any):
        """캐시에 저장"""
        # 캐시 크기 제한
        if len(self.cache) >= self.max_size:
            # 가장 오래된 항목 제거 (간단하게 첫 번째 삭제)
            first_key = next(iter(self.cache))
            del self.cache[first_key]

        key = self.get_key(node_name, state)
        self.cache[key] = result
        logger.debug(f"Cached result for {node_name}")

    def clear(self):
        """캐시 초기화"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
        }
