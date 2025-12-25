"""
Embeddings Cache - 임베딩 캐시
"""

import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

try:
    from ...utils.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


class EmbeddingCache:
    """
    Embedding 캐시: 같은 텍스트의 임베딩을 재사용하여 비용 절감

    Example:
        ```python
        from beanllm.domain.embeddings import Embedding, EmbeddingCache

        emb = Embedding(model="text-embedding-3-small")
        cache = EmbeddingCache(ttl=3600)  # 1시간 캐시

        # 첫 번째: API 호출
        vec1 = await emb.embed(["텍스트"], cache=cache)

        # 두 번째: 캐시에서 가져옴 (API 호출 안 함)
        vec2 = await emb.embed(["텍스트"], cache=cache)
        ```
    """

    def __init__(self, ttl: int = 3600, max_size: int = 10000):
        """
        Args:
            ttl: 캐시 유지 시간 (초)
            max_size: 최대 캐시 항목 수
        """
        self.cache: OrderedDict[str, tuple[List[float], float]] = OrderedDict()
        self.ttl = ttl
        self.max_size = max_size

    def get(self, text: str) -> Optional[List[float]]:
        """캐시에서 가져오기"""
        if text not in self.cache:
            return None

        vector, timestamp = self.cache[text]

        # TTL 확인
        if time.time() - timestamp > self.ttl:
            del self.cache[text]
            return None

        # LRU: 사용된 항목을 맨 뒤로
        self.cache.move_to_end(text)
        return vector

    def set(self, text: str, vector: List[float]):
        """캐시에 저장"""
        # 최대 크기 확인
        if len(self.cache) >= self.max_size:
            # 가장 오래된 항목 제거 (LRU)
            self.cache.popitem(last=False)

        self.cache[text] = (vector, time.time())

    def clear(self):
        """캐시 비우기"""
        self.cache.clear()

    def stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl": self.ttl,
        }
