"""
Web Search Integration

Google, Bing, DuckDuckGo 등 다양한 검색 엔진 통합과
실시간 웹 정보 검색을 제공합니다.

Mathematical Foundations:
=======================

1. TF-IDF (Term Frequency-Inverse Document Frequency):
   TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)

   where:
   TF(t, d) = f_{t,d} / max{f_{t',d} : t' ∈ d}
   IDF(t, D) = log(N / |{d ∈ D : t ∈ d}|)

   N = total documents, f_{t,d} = frequency of term t in document d

2. BM25 Ranking Function:
   score(D, Q) = Σ_{i=1}^n IDF(q_i) × (f(q_i, D) × (k_1 + 1)) /
                                       (f(q_i, D) + k_1 × (1 - b + b × |D| / avgdl))

   where:
   - q_i: query terms
   - f(q_i, D): frequency of q_i in document D
   - |D|: length of document D
   - avgdl: average document length
   - k_1, b: tuning parameters (typically k_1=1.2, b=0.75)

3. PageRank Algorithm:
   PR(p) = (1-d) + d × Σ_{p_i ∈ M(p)} PR(p_i) / L(p_i)

   where:
   - d: damping factor (typically 0.85)
   - M(p): set of pages linking to p
   - L(p_i): number of outbound links from p_i

References:
----------
- Salton, G., & McGill, M. J. (1983). Introduction to Modern Information Retrieval
- Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond
- Page, L., et al. (1998). The PageRank Citation Ranking

Author: LLMKit Team
"""

import re
import time
import asyncio
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import requests
from bs4 import BeautifulSoup
import httpx
from urllib.parse import urlencode, quote_plus


# ============================================================================
# Part 1: Search Result Data Structures
# ============================================================================

@dataclass
class SearchResult:
    """
    검색 결과 하나

    Attributes:
        title: 제목
        url: URL
        snippet: 요약
        source: 출처 (google, bing, duckduckgo 등)
        score: 관련도 점수 (0-1)
        published_date: 발행일 (선택)
        metadata: 추가 메타데이터
    """
    title: str
    url: str
    snippet: str
    source: str = "unknown"
    score: float = 0.0
    published_date: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.source}] {self.title}\n{self.url}\n{self.snippet[:100]}..."


@dataclass
class SearchResponse:
    """
    검색 응답

    Attributes:
        query: 검색 쿼리
        results: 검색 결과 리스트
        total_results: 전체 결과 수 (추정)
        search_time: 검색 소요 시간 (초)
        engine: 사용한 검색 엔진
        metadata: 추가 메타데이터
    """
    query: str
    results: List[SearchResult]
    total_results: Optional[int] = None
    search_time: float = 0.0
    engine: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)


class SearchEngine(Enum):
    """지원하는 검색 엔진"""
    GOOGLE = "google"
    BING = "bing"
    DUCKDUCKGO = "duckduckgo"


# ============================================================================
# Part 2: Base Search Engine
# ============================================================================

class BaseSearchEngine:
    """
    검색 엔진 베이스 클래스

    Mathematical Foundation:
        Information Retrieval as Function:
        search: Query → [Document]

        Ranked Retrieval:
        search: Query → [(Document, Score)]
        where Score = relevance(Query, Document)

        Relevance Metrics:
        - TF-IDF: Term importance in document vs corpus
        - BM25: Probabilistic ranking function
        - PageRank: Link-based authority
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 10,
        timeout: int = 10,
        cache_ttl: int = 3600
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


# ============================================================================
# Part 3: Google Custom Search
# ============================================================================

class GoogleSearch(BaseSearchEngine):
    """
    Google Custom Search API 통합

    Setup:
    1. Google Cloud Console에서 Custom Search API 활성화
    2. API 키 생성
    3. Programmable Search Engine 생성 (https://programmablesearchengine.google.com/)
    4. Search Engine ID 획득

    Mathematical Foundation:
        Google's PageRank Algorithm:

        PR(p_i) = (1-d)/N + d × Σ_{p_j ∈ M(p_i)} PR(p_j) / L(p_j)

        where:
        - N: total number of pages
        - d: damping factor (0.85)
        - M(p_i): pages linking to p_i
        - L(p_j): number of outbound links from p_j

        Iterative Computation:
        PR^(t+1) = (1-d)/N × 1 + d × M^T × PR^(t)

        where M is the transition matrix
    """

    def __init__(
        self,
        api_key: str,
        search_engine_id: str,
        **kwargs
    ):
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
        self,
        query: str,
        language: str = "en",
        safe: str = "off",
        **kwargs
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
            **kwargs
        }

        try:
            response = requests.get(
                self.base_url,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            # Parse results
            results = []
            for item in data.get("items", []):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    source="google",
                    score=1.0,  # Google doesn't provide scores
                    metadata={
                        "display_link": item.get("displayLink", ""),
                        "formatted_url": item.get("formattedUrl", "")
                    }
                ))

            search_response = SearchResponse(
                query=query,
                results=results,
                total_results=int(data.get("searchInformation", {}).get("totalResults", 0)),
                search_time=time.time() - start_time,
                engine="google",
                metadata={
                    "search_time_google": float(data.get("searchInformation", {}).get("searchTime", 0))
                }
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
                metadata={"error": str(e)}
            )

    async def search_async(
        self,
        query: str,
        language: str = "en",
        safe: str = "off",
        **kwargs
    ) -> SearchResponse:
        """비동기 검색"""
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
            **kwargs
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()

                results = []
                for item in data.get("items", []):
                    results.append(SearchResult(
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                        source="google",
                        score=1.0,
                        metadata={
                            "display_link": item.get("displayLink", ""),
                            "formatted_url": item.get("formattedUrl", "")
                        }
                    ))

                search_response = SearchResponse(
                    query=query,
                    results=results,
                    total_results=int(data.get("searchInformation", {}).get("totalResults", 0)),
                    search_time=time.time() - start_time,
                    engine="google"
                )

                self._save_to_cache(cache_key, search_response)
                return search_response

            except httpx.HTTPError as e:
                return SearchResponse(
                    query=query,
                    results=[],
                    search_time=time.time() - start_time,
                    engine="google",
                    metadata={"error": str(e)}
                )


# ============================================================================
# Part 4: Bing Search
# ============================================================================

class BingSearch(BaseSearchEngine):
    """
    Bing Search API 통합

    Setup:
    1. Azure Portal에서 Bing Search 리소스 생성
    2. API 키 획득

    Mathematical Foundation:
        Bing uses proprietary ranking algorithm, but likely based on:

        1. Content Relevance (similar to BM25)
        2. Link Analysis (similar to PageRank)
        3. User Engagement Signals (CTR, dwell time)
        4. Freshness Score

        Combined Score:
        Score = w₁ × ContentRelevance + w₂ × LinkScore +
                w₃ × UserSignals + w₄ × Freshness

        where weights w_i sum to 1
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
        self,
        query: str,
        market: str = "en-US",
        safe_search: str = "Moderate",
        **kwargs
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
            **kwargs
        }

        try:
            response = requests.get(
                self.base_url,
                headers=headers,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            # Parse web pages
            results = []
            for item in data.get("webPages", {}).get("value", []):
                results.append(SearchResult(
                    title=item.get("name", ""),
                    url=item.get("url", ""),
                    snippet=item.get("snippet", ""),
                    source="bing",
                    score=1.0,
                    published_date=self._parse_date(item.get("dateLastCrawled")),
                    metadata={
                        "display_url": item.get("displayUrl", ""),
                        "language": item.get("language", "")
                    }
                ))

            search_response = SearchResponse(
                query=query,
                results=results,
                total_results=data.get("webPages", {}).get("totalEstimatedMatches", 0),
                search_time=time.time() - start_time,
                engine="bing"
            )

            self._save_to_cache(cache_key, search_response)
            return search_response

        except requests.RequestException as e:
            return SearchResponse(
                query=query,
                results=[],
                search_time=time.time() - start_time,
                engine="bing",
                metadata={"error": str(e)}
            )

    async def search_async(
        self,
        query: str,
        market: str = "en-US",
        safe_search: str = "Moderate",
        **kwargs
    ) -> SearchResponse:
        """비동기 검색"""
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
            **kwargs
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(
                    self.base_url,
                    headers=headers,
                    params=params
                )
                response.raise_for_status()
                data = response.json()

                results = []
                for item in data.get("webPages", {}).get("value", []):
                    results.append(SearchResult(
                        title=item.get("name", ""),
                        url=item.get("url", ""),
                        snippet=item.get("snippet", ""),
                        source="bing",
                        score=1.0,
                        published_date=self._parse_date(item.get("dateLastCrawled")),
                        metadata={
                            "display_url": item.get("displayUrl", ""),
                            "language": item.get("language", "")
                        }
                    ))

                search_response = SearchResponse(
                    query=query,
                    results=results,
                    total_results=data.get("webPages", {}).get("totalEstimatedMatches", 0),
                    search_time=time.time() - start_time,
                    engine="bing"
                )

                self._save_to_cache(cache_key, search_response)
                return search_response

            except httpx.HTTPError as e:
                return SearchResponse(
                    query=query,
                    results=[],
                    search_time=time.time() - start_time,
                    engine="bing",
                    metadata={"error": str(e)}
                )

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse ISO date string"""
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except:
            return None


# ============================================================================
# Part 5: DuckDuckGo Search (No API Key Required!)
# ============================================================================

class DuckDuckGoSearch(BaseSearchEngine):
    """
    DuckDuckGo 검색 (API 키 불필요!)

    Privacy-focused search engine.
    Uses duckduckgo_search library.

    Mathematical Foundation:
        DDG doesn't use PageRank or personalization.
        Focus on:
        1. Content Relevance (TF-IDF-like)
        2. Source Authority (curated)
        3. NO user tracking → NO personalization

        Ranking ~ f(content_match, source_trust)
    """

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: BaseSearchEngine 옵션
        """
        super().__init__(api_key=None, **kwargs)

    def search(
        self,
        query: str,
        region: str = "wt-wt",
        safe_search: str = "moderate",
        **kwargs
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
        cache_key = f"ddg:{query}:{region}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        start_time = time.time()

        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                raw_results = list(ddgs.text(
                    query,
                    region=region,
                    safesearch=safe_search,
                    max_results=self.max_results
                ))

            results = []
            for item in raw_results:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("href", ""),
                    snippet=item.get("body", ""),
                    source="duckduckgo",
                    score=1.0,
                    metadata={}
                ))

            search_response = SearchResponse(
                query=query,
                results=results,
                total_results=len(results),
                search_time=time.time() - start_time,
                engine="duckduckgo"
            )

            self._save_to_cache(cache_key, search_response)
            return search_response

        except ImportError:
            return SearchResponse(
                query=query,
                results=[],
                search_time=time.time() - start_time,
                engine="duckduckgo",
                metadata={"error": "duckduckgo_search not installed. pip install duckduckgo-search"}
            )
        except Exception as e:
            return SearchResponse(
                query=query,
                results=[],
                search_time=time.time() - start_time,
                engine="duckduckgo",
                metadata={"error": str(e)}
            )

    async def search_async(
        self,
        query: str,
        region: str = "wt-wt",
        safe_search: str = "moderate",
        **kwargs
    ) -> SearchResponse:
        """비동기 검색 (DDG는 동기 라이브러리이므로 thread pool 사용)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.search,
            query,
            region,
            safe_search
        )


# ============================================================================
# Part 6: Web Scraper (URL에서 콘텐츠 추출)
# ============================================================================

class WebScraper:
    """
    웹 페이지 콘텐츠 추출기

    BeautifulSoup을 사용하여 HTML에서 텍스트 추출
    """

    @staticmethod
    def scrape(url: str, timeout: int = 10) -> Dict[str, Any]:
        """
        URL에서 콘텐츠 추출

        Args:
            url: 대상 URL
            timeout: 타임아웃 (초)

        Returns:
            {
                'title': str,
                'text': str,
                'links': List[str],
                'metadata': dict
            }
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get title
            title = soup.find('title')
            title_text = title.string if title else ""

            # Get text
            text = soup.get_text(separator='\n', strip=True)

            # Get links
            links = [a.get('href') for a in soup.find_all('a', href=True)]

            return {
                'title': title_text,
                'text': text,
                'links': links,
                'metadata': {
                    'url': url,
                    'status_code': response.status_code,
                    'content_type': response.headers.get('Content-Type', '')
                }
            }

        except Exception as e:
            return {
                'title': '',
                'text': '',
                'links': [],
                'metadata': {'error': str(e)}
            }

    @staticmethod
    async def scrape_async(url: str, timeout: int = 10) -> Dict[str, Any]:
        """비동기 스크래핑"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')

                for script in soup(["script", "style"]):
                    script.decompose()

                title = soup.find('title')
                title_text = title.string if title else ""

                text = soup.get_text(separator='\n', strip=True)
                links = [a.get('href') for a in soup.find_all('a', href=True)]

                return {
                    'title': title_text,
                    'text': text,
                    'links': links,
                    'metadata': {
                        'url': url,
                        'status_code': response.status_code,
                        'content_type': response.headers.get('Content-Type', '')
                    }
                }

        except Exception as e:
            return {
                'title': '',
                'text': '',
                'links': [],
                'metadata': {'error': str(e)}
            }


# ============================================================================
# Part 7: Unified Search Interface
# ============================================================================

class WebSearch:
    """
    통합 웹 검색 인터페이스

    여러 검색 엔진을 하나의 인터페이스로 사용
    """

    def __init__(
        self,
        google_api_key: Optional[str] = None,
        google_search_engine_id: Optional[str] = None,
        bing_api_key: Optional[str] = None,
        default_engine: SearchEngine = SearchEngine.DUCKDUCKGO,
        max_results: int = 10
    ):
        """
        Args:
            google_api_key: Google API 키
            google_search_engine_id: Google Search Engine ID
            bing_api_key: Bing API 키
            default_engine: 기본 검색 엔진
            max_results: 최대 결과 수
        """
        self.engines = {}

        # Initialize available engines
        if google_api_key and google_search_engine_id:
            self.engines[SearchEngine.GOOGLE] = GoogleSearch(
                api_key=google_api_key,
                search_engine_id=google_search_engine_id,
                max_results=max_results
            )

        if bing_api_key:
            self.engines[SearchEngine.BING] = BingSearch(
                api_key=bing_api_key,
                max_results=max_results
            )

        # DuckDuckGo always available (no API key needed)
        self.engines[SearchEngine.DUCKDUCKGO] = DuckDuckGoSearch(
            max_results=max_results
        )

        self.default_engine = default_engine
        self.scraper = WebScraper()

    def search(
        self,
        query: str,
        engine: Optional[SearchEngine] = None,
        **kwargs
    ) -> SearchResponse:
        """
        검색 실행

        Args:
            query: 검색 쿼리
            engine: 검색 엔진 (None이면 기본 엔진)
            **kwargs: 엔진별 옵션

        Returns:
            SearchResponse
        """
        engine = engine or self.default_engine

        if engine not in self.engines:
            raise ValueError(f"Search engine '{engine.value}' not configured")

        return self.engines[engine].search(query, **kwargs)

    async def search_async(
        self,
        query: str,
        engine: Optional[SearchEngine] = None,
        **kwargs
    ) -> SearchResponse:
        """비동기 검색"""
        engine = engine or self.default_engine

        if engine not in self.engines:
            raise ValueError(f"Search engine '{engine.value}' not configured")

        return await self.engines[engine].search_async(query, **kwargs)

    def search_and_scrape(
        self,
        query: str,
        max_scrape: int = 3,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        검색 후 상위 결과 스크래핑

        Args:
            query: 검색 쿼리
            max_scrape: 스크래핑할 최대 결과 수
            **kwargs: 검색 옵션

        Returns:
            스크래핑된 콘텐츠 리스트
        """
        search_results = self.search(query, **kwargs)

        scraped = []
        for result in search_results.results[:max_scrape]:
            content = self.scraper.scrape(result.url)
            scraped.append({
                'search_result': result,
                'content': content
            })

        return scraped

    async def search_and_scrape_async(
        self,
        query: str,
        max_scrape: int = 3,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """비동기 검색 및 스크래핑"""
        search_results = await self.search_async(query, **kwargs)

        tasks = [
            self.scraper.scrape_async(result.url)
            for result in search_results.results[:max_scrape]
        ]

        contents = await asyncio.gather(*tasks)

        return [
            {
                'search_result': result,
                'content': content
            }
            for result, content in zip(search_results.results[:max_scrape], contents)
        ]


# ============================================================================
# Convenience Functions
# ============================================================================

def search_web(
    query: str,
    engine: str = "duckduckgo",
    max_results: int = 10,
    **config
) -> SearchResponse:
    """
    간편한 웹 검색 함수

    Args:
        query: 검색 쿼리
        engine: 검색 엔진 ("google", "bing", "duckduckgo")
        max_results: 최대 결과 수
        **config: 엔진별 설정 (api_key 등)

    Returns:
        SearchResponse

    Example:
        >>> results = search_web("machine learning", engine="duckduckgo")
        >>> for result in results:
        ...     print(result.title, result.url)
    """
    engine_enum = SearchEngine(engine)

    searcher = WebSearch(
        google_api_key=config.get("google_api_key"),
        google_search_engine_id=config.get("google_search_engine_id"),
        bing_api_key=config.get("bing_api_key"),
        default_engine=engine_enum,
        max_results=max_results
    )

    return searcher.search(query)
