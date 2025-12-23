"""
Prompts Cache - 프롬프트 캐시
"""

import json
from typing import Any, Dict, Optional

from .base import BasePromptTemplate


class PromptCache:
    """프롬프트 캐시 (성능 최적화)"""

    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, str] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[str]:
        """캐시에서 가져오기"""
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def set(self, key: str, value: str) -> None:
        """캐시에 저장"""
        if len(self.cache) >= self.max_size:
            # LRU-like: 첫 번째 항목 제거
            first_key = next(iter(self.cache))
            del self.cache[first_key]

        self.cache[key] = value

    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "max_size": self.max_size,
        }

    def clear(self) -> None:
        """캐시 초기화"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0


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
