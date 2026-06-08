"""Tests for facade/ml/vision_rag_facade.py — VisionRAG, MultimodalRAG, create_vision_rag."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from beanllm.facade.ml.vision_rag_facade import MultimodalRAG, VisionRAG, create_vision_rag


def _make_vision_rag(answer="Vision answer", sources=None, retrieve_results=None, answers=None):
    """Create VisionRAG with fully mocked dependencies."""
    mock_store = MagicMock()
    mock_embedding = MagicMock()
    mock_llm = MagicMock()
    mock_llm.model = "gpt-4o"

    mock_handler = MagicMock()
    mock_retrieve_response = MagicMock()
    mock_retrieve_response.results = retrieve_results if retrieve_results is not None else []

    mock_query_response = MagicMock()
    mock_query_response.answer = answer
    mock_query_response.sources = sources if sources is not None else []

    mock_batch_response = MagicMock()
    mock_batch_response.answers = answers if answers is not None else []

    mock_handler.handle_retrieve = AsyncMock(return_value=mock_retrieve_response)
    mock_handler.handle_query = AsyncMock(return_value=mock_query_response)
    mock_handler.handle_batch_query = AsyncMock(return_value=mock_batch_response)

    container_patcher = patch("beanllm.utils.core.di_container.get_container")
    handler_patcher = patch("beanllm.facade.ml.vision_rag_facade.VisionRAGHandler")

    mock_get_container = container_patcher.start()
    MockHandler = handler_patcher.start()

    mock_container = MagicMock()
    mock_get_container.return_value = mock_container
    MockHandler.return_value = mock_handler

    rag = VisionRAG(
        vector_store=mock_store,
        vision_embedding=mock_embedding,
        llm=mock_llm,
    )
    return rag, mock_handler, mock_store, [container_patcher, handler_patcher]


def _stop(patchers):
    for p in patchers:
        p.stop()


# ---------------------------------------------------------------------------
# VisionRAG.__init__
# ---------------------------------------------------------------------------


class TestVisionRAGInit:
    def test_stores_vector_store(self):
        rag, _, store, p = _make_vision_rag()
        try:
            assert rag.vector_store is store
        finally:
            _stop(p)

    def test_stores_custom_embedding(self):
        mock_emb = MagicMock()
        with (
            patch("beanllm.utils.core.di_container.get_container") as mc,
            patch("beanllm.facade.ml.vision_rag_facade.VisionRAGHandler"),
        ):
            mc.return_value = MagicMock()
            rag = VisionRAG(vector_store=MagicMock(), vision_embedding=mock_emb, llm=MagicMock())
        assert rag.vision_embedding is mock_emb

    def test_default_prompt_template_contains_context(self):
        rag, _, _, p = _make_vision_rag()
        try:
            assert "{context}" in rag.prompt_template
            assert "{question}" in rag.prompt_template
        finally:
            _stop(p)

    def test_custom_prompt_template(self):
        with (
            patch("beanllm.utils.core.di_container.get_container") as mc,
            patch("beanllm.facade.ml.vision_rag_facade.VisionRAGHandler"),
        ):
            mc.return_value = MagicMock()
            rag = VisionRAG(
                vector_store=MagicMock(),
                vision_embedding=MagicMock(),
                llm=MagicMock(),
                prompt_template="Custom: {question}",
            )
        assert rag.prompt_template == "Custom: {question}"

    def test_handler_initialized(self):
        rag, handler, _, p = _make_vision_rag()
        try:
            assert rag._vision_rag_handler is handler
        finally:
            _stop(p)


# ---------------------------------------------------------------------------
# VisionRAG.retrieve
# ---------------------------------------------------------------------------


class TestVisionRAGRetrieve:
    def test_retrieve_returns_results(self):
        docs = [MagicMock(), MagicMock()]
        rag, handler, _, p = _make_vision_rag(retrieve_results=docs)
        try:
            result = rag.retrieve("cats")
            assert len(result) == 2
        finally:
            _stop(p)

    def test_retrieve_returns_empty_when_no_results(self):
        rag, _, _, p = _make_vision_rag(retrieve_results=[])
        try:
            result = rag.retrieve("nothing")
            assert result == []
        finally:
            _stop(p)

    def test_retrieve_passes_k(self):
        rag, handler, _, p = _make_vision_rag()
        try:
            rag.retrieve("q", k=8)
            call_kwargs = handler.handle_retrieve.call_args.kwargs
            assert call_kwargs.get("k") == 8
        finally:
            _stop(p)

    def test_retrieve_passes_query(self):
        rag, handler, _, p = _make_vision_rag()
        try:
            rag.retrieve("test query")
            call_kwargs = handler.handle_retrieve.call_args.kwargs
            assert call_kwargs.get("query") == "test query"
        finally:
            _stop(p)


# ---------------------------------------------------------------------------
# VisionRAG.query
# ---------------------------------------------------------------------------


class TestVisionRAGQuery:
    def test_query_returns_string(self):
        rag, _, _, p = _make_vision_rag(answer="This is a cat image.")
        try:
            result = rag.query("What animal is this?")
            assert result == "This is a cat image."
        finally:
            _stop(p)

    def test_query_with_sources_returns_tuple(self):
        rag, _, _, p = _make_vision_rag(answer="Cat", sources=["img1.jpg"])
        try:
            result = rag.query("What is this?", include_sources=True)
            assert isinstance(result, tuple)
            assert result[0] == "Cat"
            assert result[1] == ["img1.jpg"]
        finally:
            _stop(p)

    def test_query_sources_none_returns_empty_list(self):
        rag, _, _, p = _make_vision_rag(answer="ans", sources=None)
        try:
            result = rag.query("q", include_sources=True)
            assert result[1] == []
        finally:
            _stop(p)

    def test_query_passes_k(self):
        rag, handler, _, p = _make_vision_rag()
        try:
            rag.query("q", k=10)
            call_kwargs = handler.handle_query.call_args.kwargs
            assert call_kwargs.get("k") == 10
        finally:
            _stop(p)

    def test_query_passes_include_images(self):
        rag, handler, _, p = _make_vision_rag()
        try:
            rag.query("q", include_images=False)
            call_kwargs = handler.handle_query.call_args.kwargs
            assert call_kwargs.get("include_images") is False
        finally:
            _stop(p)

    def test_query_without_sources_returns_string(self):
        rag, _, _, p = _make_vision_rag(answer="Just text")
        try:
            result = rag.query("q", include_sources=False)
            assert isinstance(result, str)
        finally:
            _stop(p)


# ---------------------------------------------------------------------------
# VisionRAG.batch_query
# ---------------------------------------------------------------------------


class TestVisionRAGBatchQuery:
    def test_batch_query_returns_list(self):
        rag, _, _, p = _make_vision_rag(answers=["a1", "a2", "a3"])
        try:
            result = rag.batch_query(["q1", "q2", "q3"])
            assert result == ["a1", "a2", "a3"]
        finally:
            _stop(p)

    def test_batch_query_empty_answers(self):
        rag, _, _, p = _make_vision_rag(answers=[])
        try:
            result = rag.batch_query(["q"])
            assert result == []
        finally:
            _stop(p)

    def test_batch_query_passes_questions(self):
        rag, handler, _, p = _make_vision_rag()
        try:
            rag.batch_query(["what?", "how?"])
            call_kwargs = handler.handle_batch_query.call_args.kwargs
            assert call_kwargs.get("questions") == ["what?", "how?"]
        finally:
            _stop(p)

    def test_batch_query_passes_k(self):
        rag, handler, _, p = _make_vision_rag()
        try:
            rag.batch_query(["q"], k=6)
            call_kwargs = handler.handle_batch_query.call_args.kwargs
            assert call_kwargs.get("k") == 6
        finally:
            _stop(p)


# ---------------------------------------------------------------------------
# MultimodalRAG init
# ---------------------------------------------------------------------------


class TestMultimodalRAGInit:
    def test_is_subclass_of_vision_rag(self):
        assert issubclass(MultimodalRAG, VisionRAG)

    def test_creates_with_vector_store(self):
        with (
            patch("beanllm.utils.core.di_container.get_container") as mc,
            patch("beanllm.facade.ml.vision_rag_facade.VisionRAGHandler"),
        ):
            mc.return_value = MagicMock()
            rag = MultimodalRAG(
                vector_store=MagicMock(),
                vision_embedding=MagicMock(),
                llm=MagicMock(),
            )
        assert rag is not None


# ---------------------------------------------------------------------------
# create_vision_rag routes correctly
# ---------------------------------------------------------------------------


class TestCreateVisionRAG:
    def test_routes_list_to_multimodal(self):
        with patch.object(MultimodalRAG, "from_sources", return_value=MagicMock()) as mock_fs:
            create_vision_rag(["path1", "path2"])
            mock_fs.assert_called_once()

    def test_routes_single_source_to_vision_rag(self):
        with patch.object(VisionRAG, "from_images", return_value=MagicMock()) as mock_fi:
            create_vision_rag("path1")
            mock_fi.assert_called_once()
