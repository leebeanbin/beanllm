"""
Web Search Facade 테스트
"""
import pytest
from unittest.mock import Mock, patch, MagicMock

try:
    from llmkit.facade.web_search_facade import WebSearch
    from llmkit.domain.web_search import SearchEngine, SearchResponse
    FACADE_AVAILABLE = True
except ImportError:
    FACADE_AVAILABLE = False


@pytest.mark.skipif(not FACADE_AVAILABLE, reason="WebSearch Facade not available")
class TestWebSearch:
    @pytest.fixture
    def web_search(self):
        with patch("llmkit.utils.di_container.get_container") as mock_get_container:
            mock_handler = MagicMock()
            mock_response = SearchResponse(
                query="test query",
                results=[],
                total_results=0,
                engine=SearchEngine.DUCKDUCKGO.value
            )
            async def mock_handle_search(*args, **kwargs):
                return mock_response
            mock_handler.handle_search = MagicMock(side_effect=mock_handle_search)

            mock_handler_factory = Mock()
            mock_handler_factory.create_web_search_handler.return_value = mock_handler

            mock_container = Mock()
            mock_container.handler_factory = mock_handler_factory
            mock_get_container.return_value = mock_container

            web = WebSearch(default_engine=SearchEngine.DUCKDUCKGO)
            return web

    def test_search(self, web_search):
        result = web_search.search("machine learning")
        assert isinstance(result, SearchResponse)
        assert result.query == "test query"
        assert web_search._web_search_handler.handle_search.called

    @pytest.mark.asyncio
    async def test_search_async(self, web_search):
        result = await web_search.search_async("machine learning")
        assert isinstance(result, SearchResponse)
        assert result.query == "test query"
        assert web_search._web_search_handler.handle_search.called


