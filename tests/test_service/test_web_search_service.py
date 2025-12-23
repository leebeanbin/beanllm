"""
WebSearchService 테스트 - Web Search 서비스 구현체 테스트
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from llmkit.dto.request.web_search_request import WebSearchRequest
from llmkit.dto.response.web_search_response import WebSearchResponse
from llmkit.domain.web_search import SearchResult, SearchEngine
from llmkit.service.impl.web_search_service_impl import WebSearchServiceImpl


class TestWebSearchService:
    """WebSearchService 테스트"""

    @pytest.fixture
    def web_search_service(self):
        """WebSearchService 인스턴스"""
        return WebSearchServiceImpl()

    @pytest.fixture
    def mock_search_result(self):
        """Mock 검색 결과"""
        result = Mock(spec=SearchResult)
        result.title = "Test Result"
        result.url = "https://example.com"
        result.snippet = "Test snippet"
        return result

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires actual search engine API keys")
    async def test_search_duckduckgo(self, web_search_service):
        """DuckDuckGo 검색 테스트 (실제 API 호출)"""
        request = WebSearchRequest(
            query="Python programming",
            engine="duckduckgo",
            max_results=5,
        )

        response = await web_search_service.search(request)

        assert response is not None
        assert isinstance(response, WebSearchResponse)
        assert response.query == "Python programming"
        assert len(response.results) > 0

    @pytest.mark.asyncio
    async def test_search_google_missing_api_key(self, web_search_service):
        """Google 검색 - API 키 없음 테스트"""
        request = WebSearchRequest(
            query="Python programming",
            engine="google",
            max_results=5,
        )

        with pytest.raises(ValueError, match="Google API key"):
            await web_search_service.search(request)

    @pytest.mark.asyncio
    async def test_search_bing_missing_api_key(self, web_search_service):
        """Bing 검색 - API 키 없음 테스트"""
        request = WebSearchRequest(
            query="Python programming",
            engine="bing",
            max_results=5,
        )

        with pytest.raises(ValueError, match="Bing API key"):
            await web_search_service.search(request)

    @pytest.mark.asyncio
    async def test_search_google_with_api_key(self, web_search_service):
        """Google 검색 - API 키 포함 테스트"""
        # Mock GoogleSearch
        from llmkit.domain.web_search import GoogleSearch

        mock_engine = Mock(spec=GoogleSearch)
        mock_search_response = Mock()
        mock_search_response.query = "Python programming"
        mock_search_response.results = [Mock(spec=SearchResult)]
        mock_search_response.total_results = 1000
        mock_search_response.search_time = 0.5
        mock_search_response.engine = "google"
        mock_search_response.metadata = {}

        mock_engine.search_async = AsyncMock(return_value=mock_search_response)

        # GoogleSearch 생성자를 Mock
        with patch(
            "llmkit.service.impl.web_search_service_impl.GoogleSearch", return_value=mock_engine
        ):
            request = WebSearchRequest(
                query="Python programming",
                engine="google",
                max_results=5,
                google_api_key="test_key",
                google_search_engine_id="test_id",
            )

            response = await web_search_service.search(request)

            assert response is not None
            assert response.query == "Python programming"
            assert response.engine == "google"

    @pytest.mark.asyncio
    async def test_search_bing_with_api_key(self, web_search_service):
        """Bing 검색 - API 키 포함 테스트"""
        # Mock BingSearch
        from llmkit.domain.web_search import BingSearch

        mock_engine = Mock(spec=BingSearch)
        mock_search_response = Mock()
        mock_search_response.query = "Python programming"
        mock_search_response.results = [Mock(spec=SearchResult)]
        mock_search_response.total_results = 1000
        mock_search_response.search_time = 0.5
        mock_search_response.engine = "bing"
        mock_search_response.metadata = {}

        mock_engine.search_async = AsyncMock(return_value=mock_search_response)

        # BingSearch 생성자를 Mock
        with patch(
            "llmkit.service.impl.web_search_service_impl.BingSearch", return_value=mock_engine
        ):
            request = WebSearchRequest(
                query="Python programming",
                engine="bing",
                max_results=5,
                bing_api_key="test_key",
            )

            response = await web_search_service.search(request)

            assert response is not None
            assert response.query == "Python programming"
            assert response.engine == "bing"

    @pytest.mark.asyncio
    async def test_search_extra_params(self, web_search_service):
        """추가 파라미터 포함 검색 테스트"""
        # Mock DuckDuckGoSearch
        from llmkit.domain.web_search import DuckDuckGoSearch

        mock_engine = Mock(spec=DuckDuckGoSearch)
        mock_search_response = Mock()
        mock_search_response.query = "Python programming"
        mock_search_response.results = []
        mock_search_response.total_results = 0
        mock_search_response.search_time = 0.0
        mock_search_response.engine = "duckduckgo"
        mock_search_response.metadata = {}

        mock_engine.search_async = AsyncMock(return_value=mock_search_response)

        with patch(
            "llmkit.service.impl.web_search_service_impl.DuckDuckGoSearch", return_value=mock_engine
        ):
            request = WebSearchRequest(
                query="Python programming",
                engine="duckduckgo",
                max_results=5,
                extra_params={"param1": "value1"},
            )

            response = await web_search_service.search(request)

            assert response is not None
            # search_async가 extra_params로 호출되었는지 확인
            call_kwargs = mock_engine.search_async.call_args[1]
            assert call_kwargs.get("param1") == "value1"

    @pytest.mark.asyncio
    async def test_search_and_scrape(self, web_search_service):
        """검색 및 스크래핑 테스트"""
        # Mock search 결과
        mock_result = Mock(spec=SearchResult)
        mock_result.url = "https://example.com"

        mock_search_response = WebSearchResponse(
            query="Python programming",
            results=[mock_result],
            total_results=1,
            search_time=0.5,
            engine="duckduckgo",
        )

        # search 메서드를 Mock
        web_search_service.search = AsyncMock(return_value=mock_search_response)

        # Mock WebScraper
        mock_scraper = Mock()
        mock_scraper.scrape_async = AsyncMock(return_value="Scraped content")

        with patch(
            "llmkit.service.impl.web_search_service_impl.WebScraper", return_value=mock_scraper
        ):
            request = WebSearchRequest(
                query="Python programming",
                engine="duckduckgo",
                max_results=5,
                max_scrape=1,
            )

            results = await web_search_service.search_and_scrape(request)

            assert results is not None
            assert len(results) == 1
            assert "search_result" in results[0]
            assert "content" in results[0]
            assert results[0]["content"] == "Scraped content"

    @pytest.mark.asyncio
    async def test_search_and_scrape_multiple(self, web_search_service):
        """여러 결과 스크래핑 테스트"""
        # Mock search 결과
        mock_result1 = Mock(spec=SearchResult)
        mock_result1.url = "https://example1.com"
        mock_result2 = Mock(spec=SearchResult)
        mock_result2.url = "https://example2.com"

        mock_search_response = WebSearchResponse(
            query="Python programming",
            results=[mock_result1, mock_result2],
            total_results=2,
            search_time=0.5,
            engine="duckduckgo",
        )

        web_search_service.search = AsyncMock(return_value=mock_search_response)

        # Mock WebScraper
        mock_scraper = Mock()
        mock_scraper.scrape_async = AsyncMock(side_effect=["Content 1", "Content 2"])

        with patch(
            "llmkit.service.impl.web_search_service_impl.WebScraper", return_value=mock_scraper
        ):
            request = WebSearchRequest(
                query="Python programming",
                engine="duckduckgo",
                max_results=5,
                max_scrape=2,
            )

            results = await web_search_service.search_and_scrape(request)

            assert results is not None
            assert len(results) == 2
            assert results[0]["content"] == "Content 1"
            assert results[1]["content"] == "Content 2"


