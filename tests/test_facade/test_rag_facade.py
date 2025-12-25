"""
RAG Facade 테스트 - RAG 인터페이스 테스트
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock

try:
    from beanllm.facade.rag_facade import RAGChain
    from beanllm.domain.vector_stores.base import BaseVectorStore

    FACADE_AVAILABLE = True
except ImportError:
    FACADE_AVAILABLE = False


@pytest.mark.skipif(not FACADE_AVAILABLE, reason="RAGChain not available")
class TestRAGFacade:
    """RAGChain Facade 테스트"""

    @pytest.fixture
    def mock_vector_store(self):
        """Mock VectorStore"""
        store = Mock(spec=BaseVectorStore)
        store.similarity_search = Mock(return_value=[])
        return store

    @pytest.fixture
    def rag_chain(self, mock_vector_store):
        """RAGChain 인스턴스"""
        patcher = patch("beanllm.utils.di_container.get_container")
        mock_get_container = patcher.start()

        mock_handler = MagicMock()
        mock_response = Mock()
        mock_response.answer = "Test answer"
        mock_response.sources = []

        async def mock_handle_query(*args, **kwargs):
            return mock_response

        mock_handler.handle_query = AsyncMock(side_effect=mock_handle_query)

        mock_handler_factory = Mock()
        mock_handler_factory.create_rag_handler.return_value = mock_handler

        mock_service_factory = Mock()

        mock_container = Mock()
        mock_container.handler_factory = mock_handler_factory
        mock_container.get_service_factory.return_value = mock_service_factory
        mock_container.get_handler_factory.return_value = mock_handler_factory
        mock_get_container.return_value = mock_container

        rag = RAGChain(vector_store=mock_vector_store)

        yield rag

        patcher.stop()

    def test_query(self, rag_chain, mock_vector_store):
        """RAG 질의 테스트"""
        result = rag_chain.query("What is AI?")

        assert isinstance(result, str)
        assert result == "Test answer"
        assert rag_chain._rag_handler.handle_query.called

    def test_query_with_sources(self, rag_chain, mock_vector_store):
        """출처 포함 질의 테스트"""
        result = rag_chain.query("What is AI?", include_sources=True)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], list)


