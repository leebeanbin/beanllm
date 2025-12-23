"""
WebSearchServiceImpl - Web Search 서비스 구현체
SOLID 원칙:
- SRP: Web Search 비즈니스 로직만 담당
- DIP: 인터페이스에 의존 (의존성 주입)
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Dict, List

from ...domain.web_search import (
    BingSearch,
    DuckDuckGoSearch,
    GoogleSearch,
    SearchEngine,
    WebScraper,
)
from ...dto.request.web_search_request import WebSearchRequest
from ...dto.response.web_search_response import WebSearchResponse
from ...utils.logger import get_logger
from ..web_search_service import IWebSearchService

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class WebSearchServiceImpl(IWebSearchService):
    """
    Web Search 서비스 구현체

    책임:
    - Web Search 비즈니스 로직만
    - 검증 없음 (Handler에서 처리)
    - 에러 처리 없음 (Handler에서 처리)

    SOLID:
    - SRP: Web Search 비즈니스 로직만
    - DIP: 인터페이스에 의존 (의존성 주입)
    """

    def __init__(self) -> None:
        """의존성 주입을 통한 생성자"""
        pass

    async def search(self, request: WebSearchRequest) -> WebSearchResponse:
        """
        웹 검색 실행 (기존 web_search.py의 WebSearch.search() 정확히 마이그레이션)

        Args:
            request: Web Search 요청 DTO

        Returns:
            WebSearchResponse: Web Search 응답 DTO
        """
        # 엔진 결정 (기존과 동일)
        engine_enum = SearchEngine(request.engine) if request.engine else SearchEngine.DUCKDUCKGO

        # 엔진 인스턴스 생성 (기존과 동일)
        engine_instance = None
        if engine_enum == SearchEngine.GOOGLE:
            if not request.google_api_key or not request.google_search_engine_id:
                raise ValueError("Google API key and search engine ID are required")
            engine_instance = GoogleSearch(
                api_key=request.google_api_key,
                search_engine_id=request.google_search_engine_id,
                max_results=request.max_results,
            )
        elif engine_enum == SearchEngine.BING:
            if not request.bing_api_key:
                raise ValueError("Bing API key is required")
            engine_instance = BingSearch(
                api_key=request.bing_api_key, max_results=request.max_results
            )
        elif engine_enum == SearchEngine.DUCKDUCKGO:
            engine_instance = DuckDuckGoSearch(max_results=request.max_results)

        if not engine_instance:
            raise ValueError(f"Search engine '{engine_enum.value}' not configured")

        # 검색 실행 (기존과 동일)
        # 엔진별 옵션 준비
        search_kwargs = {}
        if engine_enum == SearchEngine.GOOGLE:
            if request.language:
                search_kwargs["language"] = request.language
            if request.safe:
                search_kwargs["safe"] = request.safe
        elif engine_enum == SearchEngine.BING:
            if request.market:
                search_kwargs["market"] = request.market
            if request.safe_search:
                search_kwargs["safe_search"] = request.safe_search
        elif engine_enum == SearchEngine.DUCKDUCKGO:
            if request.region:
                search_kwargs["region"] = request.region
            if request.safe_search:
                search_kwargs["safe_search"] = request.safe_search

        search_kwargs.update(request.extra_params or {})

        # 비동기 검색 실행 (기존과 동일)
        search_response = await engine_instance.search_async(request.query, **search_kwargs)

        # WebSearchResponse로 변환
        return WebSearchResponse(
            query=search_response.query,
            results=search_response.results,
            total_results=search_response.total_results,
            search_time=search_response.search_time,
            engine=search_response.engine,
            metadata=search_response.metadata,
        )

    async def search_and_scrape(self, request: WebSearchRequest) -> List[Dict[str, Any]]:
        """
        검색 후 상위 결과 스크래핑 (기존 web_search.py의 WebSearch.search_and_scrape_async() 정확히 마이그레이션)

        Args:
            request: Web Search 요청 DTO

        Returns:
            스크래핑된 콘텐츠 리스트
        """
        # 검색 실행 (기존과 동일)
        search_response = await self.search(request)

        # 스크래핑 (기존과 동일)
        scraper = WebScraper()
        tasks = [
            scraper.scrape_async(result.url)
            for result in search_response.results[: request.max_scrape]
        ]

        contents = await asyncio.gather(*tasks)

        # 결과 조합 (기존과 동일)
        return [
            {"search_result": result, "content": content}
            for result, content in zip(search_response.results[: request.max_scrape], contents)
        ]
