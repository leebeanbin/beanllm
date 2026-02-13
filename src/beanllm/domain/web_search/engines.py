"""
Search Engines - 검색 엔진 베이스 및 re-export hub

Base: SearchEngine enum, BaseSearchEngine
Implementations:
- engine_google: GoogleSearch
- engine_bing: BingSearch
- engine_duckduckgo: DuckDuckGoSearch
"""

import time
from abc import ABC
from enum import Enum
from typing import Dict, Optional

from beanllm.domain.web_search.security import validate_url
from beanllm.domain.web_search.types import SearchResponse
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class SearchEngine(Enum):
    """지원하는 검색 엔진"""

    GOOGLE = "google"
    BING = "bing"
    DUCKDUCKGO = "duckduckgo"


class BaseSearchEngine(ABC):
    """
    검색 엔진 베이스 클래스

    Mathematical Foundation:
        Information Retrieval as Function:
        search: Query → [Document]

        Ranked Retrieval:
        search: Query → [(Document, Score)]
        where Score = relevance(Query, Document)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 10,
        timeout: int = 10,
        cache_ttl: int = 3600,
        validate_urls: bool = False,
    ):
        """
        Args:
            api_key: API 키 (필요한 경우)
            max_results: 최대 결과 수
            timeout: 요청 타임아웃 (초)
            cache_ttl: 캐시 유효 시간 (초)
            validate_urls: 검색 결과 URL 검증 여부 (기본: False, SSRF 방지)
        """
        self.api_key = api_key
        self.max_results = max_results
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self.validate_urls = validate_urls
        self._cache: Dict[str, tuple[SearchResponse, float]] = {}

    def search(self, query: str, **kwargs: object) -> SearchResponse:
        """
        검색 실행 (동기)

        Args:
            query: 검색 쿼리
            **kwargs: 엔진별 추가 옵션

        Returns:
            SearchResponse
        """
        raise NotImplementedError

    async def search_async(self, query: str, **kwargs: object) -> SearchResponse:
        """
        검색 실행 (비동기)

        Args:
            query: 검색 쿼리
            **kwargs: 엔진별 추가 옵션

        Returns:
            SearchResponse
        """
        raise NotImplementedError

    def _get_from_cache(self, query: str) -> Optional[SearchResponse]:
        """캐시에서 조회"""
        if query in self._cache:
            response, timestamp = self._cache[query]
            if time.time() - timestamp < self.cache_ttl:
                return response
            else:
                del self._cache[query]
        return None

    def _save_to_cache(self, query: str, response: SearchResponse) -> None:
        """캐시에 저장"""
        self._cache[query] = (response, time.time())

    def _validate_result_url(self, url: str) -> Optional[str]:
        """
        검색 결과 URL 검증 (SSRF 방지)

        Args:
            url: 검증할 URL

        Returns:
            검증된 URL (실패 시 None)
        """
        if not self.validate_urls:
            return url

        try:
            return validate_url(url)
        except ValueError as e:
            # URL 검증 실패 - 로그만 남기고 None 반환
            logger.warning(f"Search result URL validation failed: {url} - {e}")
            return None


# Re-exports from implementation modules
from beanllm.domain.web_search.engine_bing import BingSearch
from beanllm.domain.web_search.engine_duckduckgo import DuckDuckGoSearch
from beanllm.domain.web_search.engine_google import GoogleSearch

__all__ = [
    "SearchEngine",
    "BaseSearchEngine",
    "GoogleSearch",
    "BingSearch",
    "DuckDuckGoSearch",
]
