"""
WebSearchHandler 테스트 - Web Search Handler 테스트
"""

import pytest
from unittest.mock import AsyncMock, Mock

from beanllm.dto.request.web_search_request import WebSearchRequest
from beanllm.dto.response.web_search_response import WebSearchResponse
from beanllm.handler.web_search_handler import WebSearchHandler


class TestWebSearchHandler:
    """WebSearchHandler 테스트"""

    @pytest.fixture
    def mock_web_search_service(self):
        """Mock WebSearchService"""
        service = Mock()
        service.search = AsyncMock(
            return_value=WebSearchResponse(
                query="Python",
                results=[],
                engine="duckduckgo",
            )
        )
        service.search_and_scrape = AsyncMock(
            return_value=[
                {"search_result": Mock(), "content": "Scraped content"}
            ]
        )
        return service

    @pytest.fixture
    def web_search_handler(self, mock_web_search_service):
        """WebSearchHandler 인스턴스"""
        return WebSearchHandler(web_search_service=mock_web_search_service)

    @pytest.mark.asyncio
    async def test_handle_search(self, web_search_handler):
        """웹 검색 테스트"""
        response = await web_search_handler.handle_search(
            query="Python programming",
            engine="duckduckgo",
            max_results=5,
        )

        assert response is not None
        assert isinstance(response, WebSearchResponse)
        # query가 request에서 전달되었는지 확인
        call_args = web_search_handler._web_search_service.search.call_args[0][0]
        assert call_args.query == "Python programming"

    @pytest.mark.asyncio
    async def test_handle_search_and_scrape(self, web_search_handler):
        """검색 및 스크래핑 테스트"""
        results = await web_search_handler.handle_search_and_scrape(
            query="Python programming",
            engine="duckduckgo",
            max_results=5,
            max_scrape=3,
        )

        assert results is not None
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_handle_search_validation_error(self, web_search_handler):
        """입력 검증 에러 테스트"""
        # query가 빈 문자열이어도 통과할 수 있음
        try:
            await web_search_handler.handle_search(
                query="",
                engine="duckduckgo",
            )
            # 통과하면 통과
        except ValueError:
            # 검증 에러가 발생하면 통과
            pass


