"""
Web Search Facade - 기존 Web Search API를 위한 Facade
책임: 하위 호환성 유지, 내부적으로는 Handler/Service 사용
SOLID 원칙:
- Facade 패턴: 복잡한 내부 구조를 단순한 인터페이스로
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..domain.web_search import SearchEngine, SearchResponse, WebScraper
from ..utils.logger import get_logger

logger = get_logger(__name__)


class WebSearch:
    """
    통합 웹 검색 인터페이스 (Facade 패턴)

    기존 API를 유지하면서 내부적으로는 Handler/Service 사용

    Example:
        ```python
        from beanllm import WebSearch, SearchEngine

        web = WebSearch(
            google_api_key="...",
            google_search_engine_id="...",
            default_engine=SearchEngine.DUCKDUCKGO
        )

        results = await web.search_async("machine learning")
        ```
    """

    def __init__(
        self,
        google_api_key: Optional[str] = None,
        google_search_engine_id: Optional[str] = None,
        bing_api_key: Optional[str] = None,
        default_engine: SearchEngine = SearchEngine.DUCKDUCKGO,
        max_results: int = 10,
    ):
        """
        Args:
            google_api_key: Google API 키
            google_search_engine_id: Google Search Engine ID
            bing_api_key: Bing API 키
            default_engine: 기본 검색 엔진
            max_results: 최대 결과 수
        """
        self.google_api_key = google_api_key
        self.google_search_engine_id = google_search_engine_id
        self.bing_api_key = bing_api_key
        self.default_engine = default_engine
        self.max_results = max_results
        self.scraper = WebScraper()

        # Handler/Service 초기화 (의존성 주입)
        self._init_services()

    def _init_services(self) -> None:
        """Service 및 Handler 초기화 (의존성 주입) - DI Container 사용"""
        from ..utils.di_container import get_container

        container = get_container()
        handler_factory = container.handler_factory
        self._web_search_handler = handler_factory.create_web_search_handler()

    def search(self, query: str, engine: Optional[SearchEngine] = None, **kwargs) -> SearchResponse:
        """
        검색 실행

        내부적으로 Handler를 사용하여 처리 (기존 web_search.py의 WebSearch.search() 정확히 마이그레이션)

        Args:
            query: 검색 쿼리
            engine: 검색 엔진 (None이면 기본 엔진)
            **kwargs: 엔진별 옵션

        Returns:
            SearchResponse
        """
        # 동기 메서드이지만 내부적으로는 비동기 사용
        # 기존 web_search.py는 동기였지만, 새로운 구조에서는 비동기 사용
        import asyncio

        response = asyncio.run(
            self._web_search_handler.handle_search(
                query=query,
                engine=engine.value if engine else self.default_engine.value,
                max_results=self.max_results,
                google_api_key=self.google_api_key,
                google_search_engine_id=self.google_search_engine_id,
                bing_api_key=self.bing_api_key,
                **kwargs,
            )
        )

        # WebSearchResponse를 SearchResponse로 변환 (기존 API 유지)
        from ..domain.web_search import SearchResponse as DomainSearchResponse

        return DomainSearchResponse(
            query=response.query,
            results=response.results,
            total_results=response.total_results,
            search_time=response.search_time,
            engine=response.engine,
            metadata=response.metadata,
        )

    async def search_async(
        self, query: str, engine: Optional[SearchEngine] = None, **kwargs
    ) -> SearchResponse:
        """
        비동기 검색

        내부적으로 Handler를 사용하여 처리

        Args:
            query: 검색 쿼리
            engine: 검색 엔진 (None이면 기본 엔진)
            **kwargs: 엔진별 옵션

        Returns:
            SearchResponse
        """
        # Handler를 통한 처리
        response = await self._web_search_handler.handle_search(
            query=query,
            engine=engine.value if engine else self.default_engine.value,
            max_results=self.max_results,
            google_api_key=self.google_api_key,
            google_search_engine_id=self.google_search_engine_id,
            bing_api_key=self.bing_api_key,
            **kwargs,
        )

        # WebSearchResponse를 SearchResponse로 변환 (기존 API 유지)
        from ..domain.web_search import SearchResponse as DomainSearchResponse

        return DomainSearchResponse(
            query=response.query,
            results=response.results,
            total_results=response.total_results,
            search_time=response.search_time,
            engine=response.engine,
            metadata=response.metadata,
        )

    def search_and_scrape(self, query: str, max_scrape: int = 3, **kwargs) -> List[Dict[str, Any]]:
        """
        검색 후 상위 결과 스크래핑

        내부적으로 Handler를 사용하여 처리 (기존 web_search.py의 WebSearch.search_and_scrape() 정확히 마이그레이션)

        Args:
            query: 검색 쿼리
            max_scrape: 스크래핑할 최대 결과 수
            **kwargs: 검색 옵션

        Returns:
            스크래핑된 콘텐츠 리스트
        """
        # 동기 메서드이지만 내부적으로는 비동기 사용
        import asyncio

        return asyncio.run(
            self._web_search_handler.handle_search_and_scrape(
                query=query,
                engine=self.default_engine.value,
                max_results=self.max_results,
                max_scrape=max_scrape,
                google_api_key=self.google_api_key,
                google_search_engine_id=self.google_search_engine_id,
                bing_api_key=self.bing_api_key,
                **kwargs,
            )
        )

    async def search_and_scrape_async(
        self, query: str, max_scrape: int = 3, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        비동기 검색 및 스크래핑

        내부적으로 Handler를 사용하여 처리

        Args:
            query: 검색 쿼리
            max_scrape: 스크래핑할 최대 결과 수
            **kwargs: 검색 옵션

        Returns:
            스크래핑된 콘텐츠 리스트
        """
        # Handler를 통한 처리
        return await self._web_search_handler.handle_search_and_scrape(
            query=query,
            engine=self.default_engine.value,
            max_results=self.max_results,
            max_scrape=max_scrape,
            google_api_key=self.google_api_key,
            google_search_engine_id=self.google_search_engine_id,
            bing_api_key=self.bing_api_key,
            **kwargs,
        )


# 편의 함수
def search_web(
    query: str, engine: str = "duckduckgo", max_results: int = 10, **config
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
        max_results=max_results,
    )

    return searcher.search(query)
