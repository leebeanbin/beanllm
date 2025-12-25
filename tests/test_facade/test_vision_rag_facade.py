"""
Vision RAG Facade 테스트
"""
import pytest
from unittest.mock import Mock, patch, MagicMock

try:
    from beanllm.facade.vision_rag_facade import VisionRAG
    from beanllm.domain.vector_stores.base import BaseVectorStore
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
        patcher = patch("beanllm.utils.di_container.get_container")
        mock_get_container = patcher.start()

        from beanllm.dto.response.vision_rag_response import VisionRAGResponse
        from beanllm.dto.response.chat_response import ChatResponse
        from unittest.mock import AsyncMock

        # Mock vision RAG handler
        mock_vision_rag_handler = MagicMock()
        async def mock_handle_query(*args, **kwargs):
            include_sources = kwargs.get('include_sources', False)
            return VisionRAGResponse(
                answer="Vision RAG answer",
                sources=[] if include_sources else None
            )
        mock_vision_rag_handler.handle_query = AsyncMock(side_effect=mock_handle_query)

        # Mock chat handler (for Client used by VisionRAG)
        mock_chat_handler = MagicMock()
        async def mock_handle_chat(*args, **kwargs):
            return ChatResponse(content="Vision RAG answer", model="gpt-4o", provider="openai")
        mock_chat_handler.handle_chat = AsyncMock(side_effect=mock_handle_chat)

        # Mock handler factory
        mock_handler_factory = Mock()
        mock_handler_factory.create_vision_rag_handler.return_value = mock_vision_rag_handler
        mock_handler_factory.create_chat_handler.return_value = mock_chat_handler

        # Mock service factory
        mock_service_factory = Mock()
        mock_chat_service = Mock()
        mock_service_factory.create_chat_service.return_value = mock_chat_service

        # Mock container
        mock_container = Mock()
        mock_container.handler_factory = mock_handler_factory
        mock_container.get_service_factory.return_value = mock_service_factory
        mock_get_container.return_value = mock_container

        rag = VisionRAG(vector_store=mock_vector_store)

        yield rag

        patcher.stop()

    def test_query(self, vision_rag):
        result = vision_rag.query("Show me images of cats")
        assert isinstance(result, str)
        assert result == "Vision RAG answer"

    def test_query_with_sources(self, vision_rag):
        result = vision_rag.query("Show me images of cats", include_sources=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], list)


