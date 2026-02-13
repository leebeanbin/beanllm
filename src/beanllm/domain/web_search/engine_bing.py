"""
Bing Search Engine - Bing Search API 통합.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, Optional

import httpx

from beanllm.domain.web_search.engines import BaseSearchEngine
from beanllm.domain.web_search.types import SearchResponse


class BingSearch(BaseSearchEngine):
    """
    Bing Search API 통합

    Setup:
    1. Azure Portal에서 Bing Search 리소스 생성
    2. API 키 획득
    """

    def __init__(self, api_key: str, **kwargs: Any):
        """
        Args:
            api_key: Bing Search API 키
            **kwargs: BaseSearchEngine 옵션
        """
        super().__init__(api_key=api_key, **kwargs)
        self.base_url = "https://api.bing.microsoft.com/v7.0/search"

    def search(
        self, query: str, market: str = "en-US", safe_search: str = "Moderate", **kwargs: Any
    ) -> SearchResponse:
        """
        Bing 검색

        Args:
            query: 검색 쿼리
            market: 시장 (en-US, ko-KR 등)
            safe_search: SafeSearch (Off, Moderate, Strict)
            **kwargs: 추가 파라미터

        Returns:
            SearchResponse
        """
        from beanllm.domain.web_search.types import SearchResult

        cache_key = f"bing:{query}:{market}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        start_time = time.time()

        api_key_val = self.api_key or ""
        headers: Dict[str, str] = {"Ocp-Apim-Subscription-Key": api_key_val}
        params = {
            "q": query,
            "count": self.max_results,
            "mkt": market,
            "safeSearch": safe_search,
            **kwargs,
        }

        try:
            response = httpx.get(
                self.base_url, headers=headers, params=params, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            # Parse web pages
            results = []
            for item in data.get("webPages", {}).get("value", []):
                result_url = item.get("url", "")

                # URL 검증 (SSRF 방지)
                validated_url = self._validate_result_url(result_url)
                if validated_url is None:
                    continue  # Skip invalid URLs

                results.append(
                    SearchResult(
                        title=item.get("name", ""),
                        url=validated_url,
                        snippet=item.get("snippet", ""),
                        source="bing",
                        score=1.0,
                        published_date=self._parse_date(item.get("dateLastCrawled")),
                        metadata={
                            "display_url": item.get("displayUrl", ""),
                            "language": item.get("language", ""),
                        },
                    )
                )

            search_response = SearchResponse(
                query=query,
                results=results,
                total_results=data.get("webPages", {}).get("totalEstimatedMatches", 0),
                search_time=time.time() - start_time,
                engine="bing",
            )

            self._save_to_cache(cache_key, search_response)
            return search_response

        except httpx.RequestError as e:
            return SearchResponse(
                query=query,
                results=[],
                search_time=time.time() - start_time,
                engine="bing",
                metadata={"error": str(e)},
            )

    async def search_async(
        self, query: str, market: str = "en-US", safe_search: str = "Moderate", **kwargs: Any
    ) -> SearchResponse:
        """비동기 검색"""
        from beanllm.domain.web_search.types import SearchResult

        cache_key = f"bing:{query}:{market}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        start_time = time.time()

        api_key_val = self.api_key or ""
        headers_async: Dict[str, str] = {"Ocp-Apim-Subscription-Key": api_key_val}
        params = {
            "q": query,
            "count": self.max_results,
            "mkt": market,
            "safeSearch": safe_search,
            **kwargs,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(self.base_url, headers=headers_async, params=params)
                response.raise_for_status()
                data = response.json()

                results = []
                for item in data.get("webPages", {}).get("value", []):
                    result_url = item.get("url", "")

                    # URL 검증 (SSRF 방지)
                    validated_url = self._validate_result_url(result_url)
                    if validated_url is None:
                        continue  # Skip invalid URLs

                    results.append(
                        SearchResult(
                            title=item.get("name", ""),
                            url=validated_url,
                            snippet=item.get("snippet", ""),
                            source="bing",
                            score=1.0,
                            published_date=self._parse_date(item.get("dateLastCrawled")),
                            metadata={
                                "display_url": item.get("displayUrl", ""),
                                "language": item.get("language", ""),
                            },
                        )
                    )

                search_response = SearchResponse(
                    query=query,
                    results=results,
                    total_results=data.get("webPages", {}).get("totalEstimatedMatches", 0),
                    search_time=time.time() - start_time,
                    engine="bing",
                )

                self._save_to_cache(cache_key, search_response)
                return search_response

            except httpx.HTTPError as e:
                return SearchResponse(
                    query=query,
                    results=[],
                    search_time=time.time() - start_time,
                    engine="bing",
                    metadata={"error": str(e)},
                )

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse ISO date string"""
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
