"""
DuckDuckGo Search Engine - DuckDuckGo 검색 (API 키 불필요).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from beanllm.domain.web_search.engines import BaseSearchEngine
from beanllm.domain.web_search.types import SearchResponse

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None


class DuckDuckGoSearch(BaseSearchEngine):
    """
    DuckDuckGo 검색 (API 키 불필요!)

    Privacy-focused search engine.
    Uses duckduckgo_search library.
    """

    def __init__(self, **kwargs: Any):
        """
        Args:
            **kwargs: BaseSearchEngine 옵션
        """
        super().__init__(api_key=None, **kwargs)

    def search(
        self, query: str, region: str = "wt-wt", safe_search: str = "moderate", **kwargs: Any
    ) -> SearchResponse:
        """
        DuckDuckGo 검색

        Args:
            query: 검색 쿼리
            region: 지역 (wt-wt=전세계, us-en=미국 등)
            safe_search: SafeSearch (on, moderate, off)
            **kwargs: 추가 옵션

        Returns:
            SearchResponse
        """
        from beanllm.domain.web_search.types import SearchResult

        cache_key = f"ddg:{query}:{region}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        start_time = time.time()

        try:
            if DDGS is None:
                raise ImportError("duckduckgo_search not installed")

            with DDGS() as ddgs:
                raw_results = list(
                    ddgs.text(
                        query, region=region, safesearch=safe_search, max_results=self.max_results
                    )
                )

            results = []
            for item in raw_results:
                result_url = item.get("href", "")

                # URL 검증 (SSRF 방지)
                validated_url = self._validate_result_url(result_url)
                if validated_url is None:
                    continue  # Skip invalid URLs

                results.append(
                    SearchResult(
                        title=item.get("title", ""),
                        url=validated_url,
                        snippet=item.get("body", ""),
                        source="duckduckgo",
                        score=1.0,
                        metadata={},
                    )
                )

            search_response = SearchResponse(
                query=query,
                results=results,
                total_results=len(results),
                search_time=time.time() - start_time,
                engine="duckduckgo",
            )

            self._save_to_cache(cache_key, search_response)
            return search_response

        except ImportError:
            return SearchResponse(
                query=query,
                results=[],
                search_time=time.time() - start_time,
                engine="duckduckgo",
                metadata={
                    "error": "duckduckgo_search not installed. pip install duckduckgo-search"
                },
            )
        except Exception as e:
            return SearchResponse(
                query=query,
                results=[],
                search_time=time.time() - start_time,
                engine="duckduckgo",
                metadata={"error": str(e)},
            )

    async def search_async(
        self, query: str, region: str = "wt-wt", safe_search: str = "moderate", **kwargs: Any
    ) -> SearchResponse:
        """비동기 검색 (DDG는 동기 라이브러리이므로 thread pool 사용)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.search, query, region, safe_search)
