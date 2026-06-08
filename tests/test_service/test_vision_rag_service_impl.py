"""Tests for service/impl/ml/vision_rag_service_impl.py"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from beanllm.dto.request.ml.vision_rag_request import VisionRAGRequest
from beanllm.dto.response.ml.vision_rag_response import VisionRAGResponse
from beanllm.service.impl.ml.vision_rag_service_impl import VisionRAGServiceImpl


def _make_service(chat_service=None, vision_embedding=None, prompt_template=None):
    mock_vs = MagicMock()
    return (
        VisionRAGServiceImpl(
            vector_store=mock_vs,
            vision_embedding=vision_embedding,
            chat_service=chat_service,
            prompt_template=prompt_template,
        ),
        mock_vs,
    )


def _make_result(content="doc content"):
    result = MagicMock()
    result.document = MagicMock()
    result.document.content = content
    return result


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestVisionRAGServiceImplInit:
    def test_default_prompt_template_used_when_none(self):
        svc, _ = _make_service()
        assert "{context}" in svc._prompt_template
        assert "{question}" in svc._prompt_template

    def test_custom_prompt_template_stored(self):
        svc, _ = _make_service(prompt_template="Custom: {context} Q: {question}")
        assert svc._prompt_template == "Custom: {context} Q: {question}"

    def test_vector_store_stored(self):
        svc, vs = _make_service()
        assert svc._vector_store is vs

    def test_optional_deps_default_none(self):
        svc, _ = _make_service()
        assert svc._chat_service is None
        assert svc._vision_embedding is None


# ---------------------------------------------------------------------------
# retrieve
# ---------------------------------------------------------------------------


class TestVisionRAGServiceImplRetrieve:
    async def test_retrieve_calls_similarity_search(self):
        svc, mock_vs = _make_service()
        mock_vs.similarity_search.return_value = [_make_result()]
        request = VisionRAGRequest(query="cat images", k=3)

        with patch(
            "beanllm.infrastructure.distributed.pipeline_decorators.with_distributed_features",
            lambda **kw: lambda f: f,  # passthrough decorator
        ):
            response = await svc.retrieve(request)

        mock_vs.similarity_search.assert_called_once_with("cat images", k=3)

    async def test_retrieve_returns_vision_rag_response(self):
        svc, mock_vs = _make_service()
        mock_vs.similarity_search.return_value = []
        request = VisionRAGRequest(query="test")

        with patch(
            "beanllm.infrastructure.distributed.pipeline_decorators.with_distributed_features",
            lambda **kw: lambda f: f,
        ):
            response = await svc.retrieve(request)

        assert isinstance(response, VisionRAGResponse)
        assert response.results == []

    async def test_retrieve_uses_empty_string_when_query_is_none(self):
        svc, mock_vs = _make_service()
        mock_vs.similarity_search.return_value = []
        request = VisionRAGRequest(query=None)

        with patch(
            "beanllm.infrastructure.distributed.pipeline_decorators.with_distributed_features",
            lambda **kw: lambda f: f,
        ):
            await svc.retrieve(request)

        mock_vs.similarity_search.assert_called_once_with("", k=4)


# ---------------------------------------------------------------------------
# _build_context
# ---------------------------------------------------------------------------


class TestBuildContext:
    def test_text_only_context(self):
        svc, _ = _make_service()
        results = [_make_result("first doc"), _make_result("second doc")]
        # include_images=False → text only
        context = svc._build_context(results, include_images=False)
        assert isinstance(context, str)
        assert "first doc" in context
        assert "second doc" in context

    def test_text_only_when_image_document_unavailable(self):
        svc, _ = _make_service()
        results = [_make_result("doc A")]
        # vision_loaders won't be importable in test environment
        with patch.dict("sys.modules", {"beanllm.vision_loaders": None}):
            context = svc._build_context(results, include_images=True)
        assert isinstance(context, (str, list))

    def test_empty_results_returns_empty_string(self):
        svc, _ = _make_service()
        context = svc._build_context([], include_images=False)
        assert context == ""

    def test_context_includes_index_numbers(self):
        svc, _ = _make_service()
        results = [_make_result("alpha"), _make_result("beta")]
        context = svc._build_context(results, include_images=False)
        assert "[1]" in context
        assert "[2]" in context

    def test_multimodal_context_with_non_image_docs(self):
        import sys
        import types

        svc, _ = _make_service()
        result = _make_result("text content")

        # Create a fake vision_loaders module with a real class our doc is not an instance of
        class ImageDocument:
            pass

        fake_module = types.ModuleType("beanllm.vision_loaders")
        fake_module.ImageDocument = ImageDocument
        with patch.dict(sys.modules, {"beanllm.vision_loaders": fake_module}):
            context = svc._build_context([result], include_images=True)
        # result.document is a MagicMock, NOT an ImageDocument instance → text path
        assert isinstance(context, list)
        assert len(context) == 1
        assert context[0]["type"] == "text"


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


class TestVisionRAGServiceImplQuery:
    def _make_chat_service(self, answer="LLM answer"):
        chat_service = AsyncMock()
        response = MagicMock()
        response.content = answer
        chat_service.chat = AsyncMock(return_value=response)
        return chat_service

    async def test_query_text_only_returns_answer(self):
        chat_svc = self._make_chat_service("Great answer")
        svc, mock_vs = _make_service(chat_service=chat_svc)
        mock_vs.similarity_search.return_value = [_make_result("relevant doc")]

        request = VisionRAGRequest(
            question="What is AI?",
            include_images=False,
            include_sources=False,
        )

        with patch(
            "beanllm.infrastructure.distributed.pipeline_decorators.with_distributed_features",
            lambda **kw: lambda f: f,
        ):
            response = await svc.query(request)

        assert response.answer == "Great answer"
        assert response.sources is None

    async def test_query_includes_sources_when_requested(self):
        chat_svc = self._make_chat_service("Answer with sources")
        svc, mock_vs = _make_service(chat_service=chat_svc)
        mock_result = _make_result("doc")
        mock_vs.similarity_search.return_value = [mock_result]

        request = VisionRAGRequest(
            question="explain?",
            include_images=False,
            include_sources=True,
        )

        with patch(
            "beanllm.infrastructure.distributed.pipeline_decorators.with_distributed_features",
            lambda **kw: lambda f: f,
        ):
            response = await svc.query(request)

        assert response.sources is not None
        assert len(response.sources) == 1

    async def test_query_raises_when_no_chat_service(self):
        svc, mock_vs = _make_service(chat_service=None)
        mock_vs.similarity_search.return_value = []
        request = VisionRAGRequest(question="test?", include_images=False)

        with patch(
            "beanllm.infrastructure.distributed.pipeline_decorators.with_distributed_features",
            lambda **kw: lambda f: f,
        ):
            with pytest.raises(ValueError, match="chat_service"):
                await svc.query(request)

    async def test_query_uses_query_field_when_question_is_none(self):
        chat_svc = self._make_chat_service()
        svc, mock_vs = _make_service(chat_service=chat_svc)
        mock_vs.similarity_search.return_value = []
        request = VisionRAGRequest(query="fallback query", question=None, include_images=False)

        with patch(
            "beanllm.infrastructure.distributed.pipeline_decorators.with_distributed_features",
            lambda **kw: lambda f: f,
        ):
            response = await svc.query(request)

        assert response.answer is not None


# ---------------------------------------------------------------------------
# batch_query
# ---------------------------------------------------------------------------


class TestVisionRAGServiceImplBatchQuery:
    def _make_service_with_chat(self, answer="batch answer"):
        chat_svc = AsyncMock()
        resp = MagicMock()
        resp.content = answer
        chat_svc.chat = AsyncMock(return_value=resp)
        svc, mock_vs = _make_service(chat_service=chat_svc)
        mock_vs.similarity_search.return_value = []
        return svc

    async def test_batch_query_raises_when_no_questions(self):
        svc = self._make_service_with_chat()
        request = VisionRAGRequest(questions=None)
        with pytest.raises(ValueError, match="questions"):
            await svc.batch_query(request)

    async def test_batch_query_raises_when_questions_empty(self):
        svc = self._make_service_with_chat()
        request = VisionRAGRequest(questions=[])
        with pytest.raises(ValueError, match="questions"):
            await svc.batch_query(request)

    async def test_batch_query_sequential_returns_answers(self):
        svc = self._make_service_with_chat("answer X")
        questions = ["q1", "q2", "q3"]
        request = VisionRAGRequest(questions=questions, include_images=False)

        with patch(
            "beanllm.infrastructure.distributed.pipeline_decorators.with_distributed_features",
            lambda **kw: lambda f: f,
        ):
            response = await svc.batch_query(request)

        assert isinstance(response, VisionRAGResponse)
        assert response.answers is not None
        assert len(response.answers) == 3

    async def test_batch_query_each_answer_from_llm(self):
        call_count = [0]

        chat_svc = AsyncMock()

        async def side_effect(req):
            call_count[0] += 1
            m = MagicMock()
            m.content = f"answer {call_count[0]}"
            return m

        chat_svc.chat = AsyncMock(side_effect=side_effect)
        svc, mock_vs = _make_service(chat_service=chat_svc)
        mock_vs.similarity_search.return_value = []

        request = VisionRAGRequest(questions=["q1", "q2"], include_images=False)

        with patch(
            "beanllm.infrastructure.distributed.pipeline_decorators.with_distributed_features",
            lambda **kw: lambda f: f,
        ):
            response = await svc.batch_query(request)

        assert response.answers == ["answer 1", "answer 2"]
