"""
WebSearchHandler - Web Search 요청 처리 (Controller 역할)
책임 분리:
- 모든 if-else/try-catch 처리
- 입력 검증
- DTO 변환
- 결과 출력
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..decorators.error_handler import handle_errors
from ..decorators.logger import log_handler_call
from ..decorators.validation import validate_input
from ..dto.request.web_search_request import WebSearchRequest
from ..dto.response.web_search_response import WebSearchResponse
from ..service.web_search_service import IWebSearchService


class WebSearchHandler:
    """
    Web Search 요청 처리 Handler

    책임:
    - 입력 검증 (if-else)
    - 에러 처리 (try-catch)
    - DTO 변환
    - Service 호출
    - 비즈니스 로직 없음
    """

    def __init__(self, web_search_service: IWebSearchService) -> None:
        """
        의존성 주입

        Args:
            web_search_service: Web Search 서비스 (인터페이스에 의존 - DIP)
        """
        self._web_search_service = web_search_service

    @log_handler_call
    @handle_errors(error_message="Web search failed")
    @validate_input(
        required_params=["query"],
        param_types={"query": str, "engine": str, "max_results": int},
    )
    async def handle_search(
        self,
        query: str,
        engine: Optional[str] = None,
        max_results: int = 10,
        google_api_key: Optional[str] = None,
        google_search_engine_id: Optional[str] = None,
        bing_api_key: Optional[str] = None,
        language: Optional[str] = None,
        safe: Optional[str] = None,
        market: Optional[str] = None,
        safe_search: Optional[str] = None,
        region: Optional[str] = None,
        **kwargs: Any,
    ) -> WebSearchResponse:
        """
        Web Search 실행 요청 처리 (모든 검증 및 에러 처리 포함)

        Args:
            query: 검색 쿼리
            engine: 검색 엔진
            max_results: 최대 결과 수
            google_api_key: Google API 키
            google_search_engine_id: Google Search Engine ID
            bing_api_key: Bing API 키
            language: 언어 (Google용)
            safe: SafeSearch (Google용)
            market: 시장 (Bing용)
            safe_search: SafeSearch (Bing/DuckDuckGo용)
            region: 지역 (DuckDuckGo용)
            **kwargs: 추가 파라미터

        Returns:
            WebSearchResponse: Web Search 응답

        책임:
            - 입력 검증 (decorator로 처리)
            - 에러 처리 (decorator로 처리)
            - DTO 변환
            - Service 호출
        """
        # DTO 생성
        request = WebSearchRequest(
            query=query,
            engine=engine,
            max_results=max_results,
            google_api_key=google_api_key,
            google_search_engine_id=google_search_engine_id,
            bing_api_key=bing_api_key,
            language=language,
            safe=safe,
            market=market,
            safe_search=safe_search,
            region=region,
            extra_params=kwargs,
        )

        # Service 호출 (에러 처리는 decorator가 담당)
        return await self._web_search_service.search(request)

    @log_handler_call
    @handle_errors(error_message="Web search and scrape failed")
    @validate_input(
        required_params=["query"],
        param_types={"query": str, "max_scrape": int},
    )
    async def handle_search_and_scrape(
        self,
        query: str,
        engine: Optional[str] = None,
        max_results: int = 10,
        max_scrape: int = 3,
        google_api_key: Optional[str] = None,
        google_search_engine_id: Optional[str] = None,
        bing_api_key: Optional[str] = None,
        language: Optional[str] = None,
        safe: Optional[str] = None,
        market: Optional[str] = None,
        safe_search: Optional[str] = None,
        region: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        검색 후 상위 결과 스크래핑 요청 처리

        Args:
            query: 검색 쿼리
            engine: 검색 엔진
            max_results: 최대 결과 수
            max_scrape: 스크래핑할 최대 결과 수
            google_api_key: Google API 키
            google_search_engine_id: Google Search Engine ID
            bing_api_key: Bing API 키
            language: 언어 (Google용)
            safe: SafeSearch (Google용)
            market: 시장 (Bing용)
            safe_search: SafeSearch (Bing/DuckDuckGo용)
            region: 지역 (DuckDuckGo용)
            **kwargs: 추가 파라미터

        Returns:
            스크래핑된 콘텐츠 리스트
        """
        # DTO 생성
        request = WebSearchRequest(
            query=query,
            engine=engine,
            max_results=max_results,
            max_scrape=max_scrape,
            google_api_key=google_api_key,
            google_search_engine_id=google_search_engine_id,
            bing_api_key=bing_api_key,
            language=language,
            safe=safe,
            market=market,
            safe_search=safe_search,
            region=region,
            extra_params=kwargs,
        )

        # Service 호출 (에러 처리는 decorator가 담당)
        return await self._web_search_service.search_and_scrape(request)
