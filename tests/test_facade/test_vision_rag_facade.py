"""
Vision RAG Facade 테스트
"""
import pytest
from unittest.mock import Mock, patch, MagicMock

try:
    from llmkit.facade.vision_rag_facade import VisionRAG
    from llmkit.domain.vector_stores.base import BaseVectorStore
    FACADE_AVAILABLE = True
except ImportError:
    FACADE_AVAILABLE = False


@pytest.mark.skipif(not FACADE_AVAILABLE, reason="VisionRAG Facade not available")
class TestVisionRAG:
    @pytest.fixture
    def mock_vector_store(self):
        store = Mock(spec=BaseVectorStore)
        store.similarity_search = Mock(return_value=[])
        return store

    @pytest.fixture
    def vision_rag(self, mock_vector_store):
        with patch('llmkit.facade.vision_rag_facade.HandlerFactory') as mock_factory:
            mock_handler = MagicMock()
            # query는 직접 값을 반환 (str 또는 tuple)
            async def mock_handle_query(*args, **kwargs):
                # include_sources에 따라 반환 타입이 달라짐
                include_sources = kwargs.get('include_sources', False)
                if include_sources:
                    return ("Vision RAG answer", [])
                return "Vision RAG answer"
            from unittest.mock import AsyncMock
            mock_handler.handle_query = AsyncMock(side_effect=mock_handle_query)
            
            mock_handler_factory = Mock()
            mock_handler_factory.create_vision_rag_handler.return_value = mock_handler
            mock_factory.return_value = mock_handler_factory
            
            rag = VisionRAG(vector_store=mock_vector_store)
            rag._vision_rag_handler = mock_handler
            return rag

    def test_query(self, vision_rag):
        result = vision_rag.query("Show me images of cats")
        assert isinstance(result, str)
        assert result == "Vision RAG answer"
        assert vision_rag._vision_rag_handler.handle_query.called

    def test_query_with_sources(self, vision_rag):
        result = vision_rag.query("Show me images of cats", include_sources=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], list)


