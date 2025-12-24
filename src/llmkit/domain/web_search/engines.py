"""
Search Engines - 검색 엔진 구현체들
"""

import asyncio
import time
from abc import ABC
from datetime import datetime
from enum import Enum
from typing import Dict, Optional

import httpx
import requests

from .types import SearchResponse

# DuckDuckGo는 선택적 의존성
try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None


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
    ):
        """
        Args:
            api_key: API 키 (필요한 경우)
            max_results: 최대 결과 수
            timeout: 요청 타임아웃 (초)
            cache_ttl: 캐시 유효 시간 (초)
        """
        self.api_key = api_key
        self.max_results = max_results
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, tuple[SearchResponse, float]] = {}

    def search(self, query: str, **kwargs) -> SearchResponse:
        """
        검색 실행 (동기)

        Args:
            query: 검색 쿼리
            **kwargs: 엔진별 추가 옵션

        Returns:
            SearchResponse
        """
        raise NotImplementedError

    async def search_async(self, query: str, **kwargs) -> SearchResponse:
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

    def _save_to_cache(self, query: str, response: SearchResponse):
        """캐시에 저장"""
        self._cache[query] = (response, time.time())


class GoogleSearch(BaseSearchEngine):
    """
    Google Custom Search API 통합

    Setup:
    1. Google Cloud Console에서 Custom Search API 활성화
    2. API 키 생성
    3. Programmable Search Engine 생성
    4. Search Engine ID 획득
    """

    def __init__(self, api_key: str, search_engine_id: str, **kwargs):
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
        self, query: str, language: str = "en", safe: str = "off", **kwargs
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
        from .types import SearchResult

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
            response = requests.get(self.base_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            # Parse results
            results = []
            for item in data.get("items", []):
                results.append(
                    SearchResult(
                        title=item.get("title", ""),
                        url=item.get("link", ""),
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

        except requests.RequestException as e:
            return SearchResponse(
                query=query,
                results=[],
                search_time=time.time() - start_time,
                engine="google",
                metadata={"error": str(e)},
            )

    async def search_async(
        self, query: str, language: str = "en", safe: str = "off", **kwargs
    ) -> SearchResponse:
        """비동기 검색"""
        from .types import SearchResult

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
                    results.append(
                        SearchResult(
                            title=item.get("title", ""),
                            url=item.get("link", ""),
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


class BingSearch(BaseSearchEngine):
    """
    Bing Search API 통합

    Setup:
    1. Azure Portal에서 Bing Search 리소스 생성
    2. API 키 획득
    """

    def __init__(self, api_key: str, **kwargs):
        """
        Args:
            api_key: Bing Search API 키
            **kwargs: BaseSearchEngine 옵션
        """
        super().__init__(api_key=api_key, **kwargs)
        self.base_url = "https://api.bing.microsoft.com/v7.0/search"

    def search(
        self, query: str, market: str = "en-US", safe_search: str = "Moderate", **kwargs
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
        from .types import SearchResult

        cache_key = f"bing:{query}:{market}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        start_time = time.time()

        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {
            "q": query,
            "count": self.max_results,
            "mkt": market,
            "safeSearch": safe_search,
            **kwargs,
        }

        try:
            response = requests.get(
                self.base_url, headers=headers, params=params, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            # Parse web pages
            results = []
            for item in data.get("webPages", {}).get("value", []):
                results.append(
                    SearchResult(
                        title=item.get("name", ""),
                        url=item.get("url", ""),
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

        except requests.RequestException as e:
            return SearchResponse(
                query=query,
                results=[],
                search_time=time.time() - start_time,
                engine="bing",
                metadata={"error": str(e)},
            )

    async def search_async(
        self, query: str, market: str = "en-US", safe_search: str = "Moderate", **kwargs
    ) -> SearchResponse:
        """비동기 검색"""
        from .types import SearchResult

        cache_key = f"bing:{query}:{market}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        start_time = time.time()

        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {
            "q": query,
            "count": self.max_results,
            "mkt": market,
            "safeSearch": safe_search,
            **kwargs,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(self.base_url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()

                results = []
                for item in data.get("webPages", {}).get("value", []):
                    results.append(
                        SearchResult(
                            title=item.get("name", ""),
                            url=item.get("url", ""),
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


class DuckDuckGoSearch(BaseSearchEngine):
    """
    DuckDuckGo 검색 (API 키 불필요!)

    Privacy-focused search engine.
    Uses duckduckgo_search library.
    """

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: BaseSearchEngine 옵션
        """
        super().__init__(api_key=None, **kwargs)

    def search(
        self, query: str, region: str = "wt-wt", safe_search: str = "moderate", **kwargs
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
        from .types import SearchResult

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
                results.append(
                    SearchResult(
                        title=item.get("title", ""),
                        url=item.get("href", ""),
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
        self, query: str, region: str = "wt-wt", safe_search: str = "moderate", **kwargs
    ) -> SearchResponse:
        """비동기 검색 (DDG는 동기 라이브러리이므로 thread pool 사용)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.search, query, region, safe_search)
