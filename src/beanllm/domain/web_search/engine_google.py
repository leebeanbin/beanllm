"""
Google Search Engine - Google Custom Search API 통합.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from beanllm.domain.web_search.engines import BaseSearchEngine
from beanllm.domain.web_search.types import SearchResponse


class GoogleSearch(BaseSearchEngine):
    """
    Google Custom Search API 통합

    Setup:
    1. Google Cloud Console에서 Custom Search API 활성화
    2. API 키 생성
    3. Programmable Search Engine 생성
    4. Search Engine ID 획득
    """

    def __init__(self, api_key: str, search_engine_id: str, **kwargs: Any):
        """
        Args:
            api_key: Google API 키
            search_engine_id: Programmable Search Engine ID
            **kwargs: BaseSearchEngine 옵션
        """
        super().__init__(api_key=api_key, **kwargs)
        self.search_engine_id = search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def search(
        self, query: str, language: str = "en", safe: str = "off", **kwargs: Any
    ) -> SearchResponse:
        """
        Google 검색

        Args:
            query: 검색 쿼리
            language: 언어 (en, ko 등)
            safe: SafeSearch (off, medium, high)
            **kwargs: 추가 파라미터

        Returns:
            SearchResponse
        """
        from beanllm.domain.web_search.types import SearchResult

        # Check cache
        cache_key = f"google:{query}:{language}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        start_time = time.time()

        params = {
            "key": self.api_key,
            "cx": self.search_engine_id,
            "q": query,
            "num": min(self.max_results, 10),  # Google API max is 10
            "lr": f"lang_{language}",
            "safe": safe,
            **kwargs,
        }

        try:
            response = httpx.get(self.base_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            # Parse results
            results = []
            for item in data.get("items", []):
                result_url = item.get("link", "")

                # URL 검증 (SSRF 방지)
                validated_url = self._validate_result_url(result_url)
                if validated_url is None:
                    continue  # Skip invalid URLs

                results.append(
                    SearchResult(
                        title=item.get("title", ""),
                        url=validated_url,
                        snippet=item.get("snippet", ""),
                        source="google",
                        score=1.0,  # Google doesn't provide scores
                        metadata={
                            "display_link": item.get("displayLink", ""),
                            "formatted_url": item.get("formattedUrl", ""),
                        },
                    )
                )

            search_response = SearchResponse(
                query=query,
                results=results,
                total_results=int(data.get("searchInformation", {}).get("totalResults", 0)),
                search_time=time.time() - start_time,
                engine="google",
                metadata={
                    "search_time_google": float(
                        data.get("searchInformation", {}).get("searchTime", 0)
                    )
                },
            )

            # Cache
            self._save_to_cache(cache_key, search_response)

            return search_response

        except httpx.RequestError as e:
            return SearchResponse(
                query=query,
                results=[],
                search_time=time.time() - start_time,
                engine="google",
                metadata={"error": str(e)},
            )

    async def search_async(
        self, query: str, language: str = "en", safe: str = "off", **kwargs: Any
    ) -> SearchResponse:
        """비동기 검색"""
        from beanllm.domain.web_search.types import SearchResult

        cache_key = f"google:{query}:{language}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        start_time = time.time()

        params = {
            "key": self.api_key,
            "cx": self.search_engine_id,
            "q": query,
            "num": min(self.max_results, 10),
            "lr": f"lang_{language}",
            "safe": safe,
            **kwargs,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()

                results = []
                for item in data.get("items", []):
                    result_url = item.get("link", "")

                    # URL 검증 (SSRF 방지)
                    validated_url = self._validate_result_url(result_url)
                    if validated_url is None:
                        continue  # Skip invalid URLs

                    results.append(
                        SearchResult(
                            title=item.get("title", ""),
                            url=validated_url,
                            snippet=item.get("snippet", ""),
                            source="google",
                            score=1.0,
                            metadata={
                                "display_link": item.get("displayLink", ""),
                                "formatted_url": item.get("formattedUrl", ""),
                            },
                        )
                    )

                search_response = SearchResponse(
                    query=query,
                    results=results,
                    total_results=int(data.get("searchInformation", {}).get("totalResults", 0)),
                    search_time=time.time() - start_time,
                    engine="google",
                )

                self._save_to_cache(cache_key, search_response)
                return search_response

            except httpx.HTTPError as e:
                return SearchResponse(
                    query=query,
                    results=[],
                    search_time=time.time() - start_time,
                    engine="google",
                    metadata={"error": str(e)},
                )
