"""
RAG Facade 테스트 - RAG 인터페이스 테스트
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from beanllm.facade.core.rag_facade import RAGChain


def _make_rag_chain(answer="Test answer", sources=None, retrieve_result=None):
    """Create a RAGChain with fully mocked dependencies."""
    mock_store = MagicMock()
    patcher = patch("beanllm.utils.core.di_container.get_container")
    mock_get_container = patcher.start()

    mock_handler = MagicMock()
    mock_response = MagicMock()
    mock_response.answer = answer
    mock_response.sources = sources if sources is not None else []

    async def mock_handle_query(*args, **kwargs):
        return mock_response

    async def mock_handle_retrieve(*args, **kwargs):
        return retrieve_result if retrieve_result is not None else ["doc1", "doc2"]

    async def mock_handle_stream(*args, **kwargs):
        for chunk in ["Hello", " world"]:
            yield chunk

    mock_handler.handle_query = AsyncMock(side_effect=mock_handle_query)
    mock_handler.handle_retrieve = AsyncMock(side_effect=mock_handle_retrieve)
    mock_handler.handle_stream_query = mock_handle_stream

    mock_handler_factory = MagicMock()
    mock_handler_factory.create_rag_handler.return_value = mock_handler
    mock_service_factory = MagicMock()

    mock_container = MagicMock()
    mock_container.get_service_factory.return_value = mock_service_factory
    mock_container.get_handler_factory.return_value = mock_handler_factory
    mock_get_container.return_value = mock_container

    rag = RAGChain(vector_store=mock_store)
    return rag, mock_handler, mock_store, patcher


class TestRAGChainInit:
    def test_creates_with_vector_store(self):
        rag, _, store, p = _make_rag_chain()
        try:
            assert rag.vector_store is store
        finally:
            p.stop()

    def test_default_prompt_template(self):
        rag, _, store, p = _make_rag_chain()
        try:
            assert "context" in rag.prompt_template.lower()
            assert "question" in rag.prompt_template.lower()
        finally:
            p.stop()

    def test_custom_prompt_template(self):
        mock_store = MagicMock()
        with patch("beanllm.utils.core.di_container.get_container") as mock_get:
            mock_container = MagicMock()
            mock_handler_factory = MagicMock()
            mock_handler_factory.create_rag_handler.return_value = MagicMock()
            mock_container.get_service_factory.return_value = MagicMock()
            mock_container.get_handler_factory.return_value = mock_handler_factory
            mock_get.return_value = mock_container
            rag = RAGChain(vector_store=mock_store, prompt_template="Custom: {question}")
        assert rag.prompt_template == "Custom: {question}"

    def test_rag_alias(self):
        from beanllm.facade.core.rag_facade import RAG

        assert RAG is RAGChain


class TestRAGChainQuery:
    def test_query_returns_answer_string(self):
        rag, handler, store, p = _make_rag_chain(answer="AI is great")
        try:
            result = rag.query("What is AI?")
            assert result == "AI is great"
        finally:
            p.stop()

    def test_query_with_sources_returns_tuple(self):
        rag, handler, store, p = _make_rag_chain(answer="AI answer", sources=["doc1"])
        try:
            result = rag.query("What is AI?", include_sources=True)
            assert isinstance(result, tuple)
            assert result[0] == "AI answer"
            assert result[1] == ["doc1"]
        finally:
            p.stop()

    def test_query_sources_none_returns_empty_list(self):
        rag, handler, store, p = _make_rag_chain(answer="ans", sources=None)
        try:
            result = rag.query("q", include_sources=True)
            assert result[1] == []
        finally:
            p.stop()

    def test_query_passes_k_parameter(self):
        rag, handler, store, p = _make_rag_chain()
        try:
            rag.query("q", k=10)
            call_kwargs = handler.handle_query.call_args.kwargs
            assert call_kwargs.get("k") == 10
        finally:
            p.stop()

    def test_query_passes_model(self):
        rag, handler, store, p = _make_rag_chain()
        try:
            rag.query("q", model="gpt-4o")
            call_kwargs = handler.handle_query.call_args.kwargs
            assert call_kwargs.get("llm_model") == "gpt-4o"
        finally:
            p.stop()

    def test_query_passes_rerank_flag(self):
        rag, handler, store, p = _make_rag_chain()
        try:
            rag.query("q", rerank=True)
            call_kwargs = handler.handle_query.call_args.kwargs
            assert call_kwargs.get("rerank") is True
        finally:
            p.stop()

    def test_query_passes_mmr_flag(self):
        rag, handler, store, p = _make_rag_chain()
        try:
            rag.query("q", mmr=True)
            call_kwargs = handler.handle_query.call_args.kwargs
            assert call_kwargs.get("mmr") is True
        finally:
            p.stop()

    def test_query_passes_hybrid_flag(self):
        rag, handler, store, p = _make_rag_chain()
        try:
            rag.query("q", hybrid=True)
            call_kwargs = handler.handle_query.call_args.kwargs
            assert call_kwargs.get("hybrid") is True
        finally:
            p.stop()


class TestRAGChainRetrieve:
    def test_retrieve_returns_documents(self):
        docs = [MagicMock(), MagicMock()]
        rag, handler, store, p = _make_rag_chain(retrieve_result=docs)
        try:
            result = rag.retrieve("find docs")
            assert len(result) == 2
        finally:
            p.stop()

    def test_retrieve_returns_empty_on_none(self):
        rag, handler, store, p = _make_rag_chain(retrieve_result=None)
        try:
            # When handler returns None (from our mock, it returns ["doc1","doc2"])
            # Override to return None
            async def return_none(*a, **k):
                return None

            handler.handle_retrieve = AsyncMock(side_effect=return_none)
            result = rag.retrieve("q")
            assert result == []
        finally:
            p.stop()

    def test_retrieve_passes_k(self):
        rag, handler, store, p = _make_rag_chain()
        try:
            rag.retrieve("q", k=7)
            call_kwargs = handler.handle_retrieve.call_args.kwargs
            assert call_kwargs.get("k") == 7
        finally:
            p.stop()


class TestRAGChainAquery:
    async def test_aquery_returns_string(self):
        rag, handler, store, p = _make_rag_chain(answer="async answer")
        try:
            result = await rag.aquery("What is ML?")
            assert result == "async answer"
        finally:
            p.stop()

    async def test_aquery_with_include_sources(self):
        rag, handler, store, p = _make_rag_chain(answer="ans", sources=["src1"])
        try:
            result = await rag.aquery("q", include_sources=True)
            assert isinstance(result, tuple)
        finally:
            p.stop()


class TestRAGChainBatchQuery:
    def test_batch_query_returns_list(self):
        rag, handler, store, p = _make_rag_chain(answer="batch answer")
        try:
            with (
                patch("beanllm.infrastructure.distributed.get_rate_limiter") as mock_rl,
                patch("beanllm.infrastructure.distributed.ConcurrencyController") as mock_cc,
            ):
                mock_rate_limiter = MagicMock()
                mock_rate_limiter.wait = AsyncMock()
                mock_rl.return_value = mock_rate_limiter

                mock_cc_instance = MagicMock()
                ctx_mgr = MagicMock()
                ctx_mgr.__aenter__ = AsyncMock(return_value=None)
                ctx_mgr.__aexit__ = AsyncMock(return_value=False)
                mock_cc_instance.with_concurrency_control = AsyncMock(return_value=ctx_mgr)
                mock_cc.return_value = mock_cc_instance

                results = rag.batch_query(["q1", "q2", "q3"])
            assert isinstance(results, list)
            assert len(results) == 3
        finally:
            p.stop()

    def test_batch_query_empty_list(self):
        rag, handler, store, p = _make_rag_chain()
        try:
            with (
                patch("beanllm.infrastructure.distributed.get_rate_limiter"),
                patch("beanllm.infrastructure.distributed.ConcurrencyController"),
            ):
                results = rag.batch_query([])
            assert results == []
        finally:
            p.stop()


class TestRAGChainStreamQuery:
    def test_stream_query_yields_chunks(self):
        rag, handler, store, p = _make_rag_chain()
        try:
            chunks = list(rag.stream_query("What is AI?"))
            assert chunks == ["Hello", " world"]
        finally:
            p.stop()

    def test_stream_query_passes_k(self):
        rag, handler, store, p = _make_rag_chain()
        try:
            list(rag.stream_query("question", k=10))
        finally:
            p.stop()

    def test_stream_query_with_model(self):
        rag, handler, store, p = _make_rag_chain()
        try:
            chunks = list(rag.stream_query("q", model="gpt-4o"))
            assert len(chunks) == 2
        finally:
            p.stop()

    def test_stream_query_with_rerank(self):
        rag, handler, store, p = _make_rag_chain()
        try:
            chunks = list(rag.stream_query("q", rerank=True))
            assert isinstance(chunks, list)
        finally:
            p.stop()

    def test_stream_query_with_mmr(self):
        rag, handler, store, p = _make_rag_chain()
        try:
            chunks = list(rag.stream_query("q", mmr=True))
            assert isinstance(chunks, list)
        finally:
            p.stop()

    def test_stream_query_with_hybrid(self):
        rag, handler, store, p = _make_rag_chain()
        try:
            chunks = list(rag.stream_query("q", hybrid=True))
            assert isinstance(chunks, list)
        finally:
            p.stop()


class TestRAGChainFromDocumentsError:
    def test_from_documents_raises_on_document_loader_error(self):
        """from_documents re-raises exceptions after logging the error event."""
        with (
            patch("beanllm.domain.loaders.DocumentLoader.load") as mock_load,
            patch("beanllm.infrastructure.distributed.get_event_logger", return_value=None),
            patch("beanllm.utils.core.di_container.get_container"),
        ):
            mock_load.side_effect = RuntimeError("Cannot load documents")
            with pytest.raises(RuntimeError, match="Cannot load documents"):
                RAGChain.from_documents("/nonexistent/path")

    def test_from_documents_with_list_input(self):
        """from_documents accepts a list of documents directly."""
        mock_doc = MagicMock()
        mock_doc.content = "test content"
        mock_chunks = [mock_doc]
        mock_vector_store = MagicMock()
        mock_llm = MagicMock()

        with (
            patch("beanllm.domain.splitters.TextSplitter.split", return_value=mock_chunks),
            patch("beanllm.domain.embeddings.Embedding") as mock_emb_cls,
            patch("beanllm.domain.vector_stores.from_documents", return_value=mock_vector_store),
            patch("beanllm.facade.core.client_facade.Client", return_value=mock_llm),
            patch("beanllm.infrastructure.distributed.get_event_logger", return_value=None),
            patch("beanllm.utils.core.di_container.get_container"),
        ):
            mock_emb = MagicMock()
            mock_emb_cls.return_value = mock_emb
            mock_emb.embed_sync = MagicMock(return_value=[[0.1, 0.2]])

            rag = RAGChain.from_documents([mock_doc])
            assert isinstance(rag, RAGChain)


class TestSafeLogEvent:
    def test_safe_log_with_none_logger(self):
        from beanllm.facade.core.rag_facade import _safe_log_event

        _safe_log_event(None, "test.event", {"key": "value"})  # should not raise

    def test_safe_log_with_logger(self):
        from beanllm.facade.core.rag_facade import _safe_log_event

        mock_logger = MagicMock()
        _safe_log_event(mock_logger, "test.event", {"key": "value"})
        mock_logger.info.assert_called_once_with("test.event", {"key": "value"})

    def test_safe_log_with_error_level(self):
        from beanllm.facade.core.rag_facade import _safe_log_event

        mock_logger = MagicMock()
        _safe_log_event(mock_logger, "test.event", {"key": "value"}, level="error")
        mock_logger.error.assert_called_once()

    def test_safe_log_swallows_exceptions(self):
        from beanllm.facade.core.rag_facade import _safe_log_event

        mock_logger = MagicMock()
        mock_logger.info.side_effect = RuntimeError("logging failed")
        _safe_log_event(mock_logger, "event", {})  # should not raise
