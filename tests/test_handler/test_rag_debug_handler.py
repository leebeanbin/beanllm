"""
RAGDebugHandler 테스트
"""

from unittest.mock import AsyncMock

import pytest

from beanllm.dto.request.ml.rag_debug_request import (
    AnalyzeEmbeddingsRequest,
    StartDebugSessionRequest,
    TuneParametersRequest,
    ValidateChunksRequest,
)
from beanllm.dto.response.ml.rag_debug_response import (
    AnalyzeEmbeddingsResponse,
    DebugSessionResponse,
    TuneParametersResponse,
    ValidateChunksResponse,
)
from beanllm.handler.advanced.rag_debug_handler import RAGDebugHandler
from beanllm.service.rag_debug_service import IRAGDebugService


def _make_session_response() -> DebugSessionResponse:
    return DebugSessionResponse(
        session_id="sess-1",
        session_name="test-session",
        vector_store_id="vs-1",
        num_documents=100,
        num_embeddings=100,
        embedding_dim=384,
        status="active",
        created_at="2026-01-01T00:00:00",
    )


def _make_analyze_response() -> AnalyzeEmbeddingsResponse:
    return AnalyzeEmbeddingsResponse(
        session_id="sess-1",
        method="umap",
        num_clusters=3,
        cluster_labels=[0, 1, 2, 0, 1],
        cluster_sizes={0: 2, 1: 2, 2: 1},
        outliers=[],
    )


def _make_validate_response() -> ValidateChunksResponse:
    return ValidateChunksResponse(
        session_id="sess-1",
        total_chunks=100,
        valid_chunks=95,
        issues=[{"type": "size", "chunk_id": "c1"}],
        size_distribution={"small": 5, "medium": 80, "large": 15},
        recommendations=["Consider splitting large chunks"],
    )


def _make_tune_response() -> TuneParametersResponse:
    return TuneParametersResponse(
        session_id="sess-1",
        parameters={"top_k": 5, "threshold": 0.8},
        test_results=[{"query": "test", "score": 0.9}],
        avg_score=0.9,
        recommendations=["Current parameters are optimal"],
    )


class TestRAGDebugHandler:
    @pytest.fixture
    def mock_service(self) -> AsyncMock:
        service = AsyncMock(spec=IRAGDebugService)
        service.start_session.return_value = _make_session_response()
        service.analyze_embeddings.return_value = _make_analyze_response()
        service.validate_chunks.return_value = _make_validate_response()
        service.tune_parameters.return_value = _make_tune_response()
        service.export_report.return_value = {"session_id": "sess-1", "status": "exported"}
        return service

    @pytest.fixture
    def handler(self, mock_service: AsyncMock) -> RAGDebugHandler:
        return RAGDebugHandler(service=mock_service)

    async def test_handle_start_session(self, handler: RAGDebugHandler) -> None:
        request = StartDebugSessionRequest(
            vector_store_id="vs-1",
            config={"vector_store": "chroma"},
        )
        result = await handler.handle_start_session(request)
        assert isinstance(result, DebugSessionResponse)
        assert result.session_id == "sess-1"

    async def test_handle_start_session_no_vector_store_raises(
        self, handler: RAGDebugHandler
    ) -> None:
        request = StartDebugSessionRequest(
            vector_store_id="",
            config={"vector_store": "chroma"},
        )
        with pytest.raises(Exception):
            await handler.handle_start_session(request)

    async def test_handle_start_session_no_config_raises(self, handler: RAGDebugHandler) -> None:
        request = StartDebugSessionRequest(
            vector_store_id="vs-1",
            config={},  # missing vector_store key
        )
        with pytest.raises(Exception):
            await handler.handle_start_session(request)

    async def test_handle_analyze_embeddings(self, handler: RAGDebugHandler) -> None:
        request = AnalyzeEmbeddingsRequest(session_id="sess-1")
        result = await handler.handle_analyze_embeddings(request)
        assert isinstance(result, AnalyzeEmbeddingsResponse)
        assert result.num_clusters == 3

    async def test_handle_analyze_embeddings_no_session_raises(
        self, handler: RAGDebugHandler
    ) -> None:
        request = AnalyzeEmbeddingsRequest(session_id="")
        with pytest.raises(Exception):
            await handler.handle_analyze_embeddings(request)

    async def test_handle_validate_chunks(self, handler: RAGDebugHandler) -> None:
        request = ValidateChunksRequest(session_id="sess-1")
        result = await handler.handle_validate_chunks(request)
        assert isinstance(result, ValidateChunksResponse)
        assert result.total_chunks == 100
        assert result.valid_chunks == 95

    async def test_handle_validate_chunks_no_session_raises(self, handler: RAGDebugHandler) -> None:
        request = ValidateChunksRequest(session_id="")
        with pytest.raises(Exception):
            await handler.handle_validate_chunks(request)

    async def test_handle_tune_parameters(self, handler: RAGDebugHandler) -> None:
        request = TuneParametersRequest(
            session_id="sess-1",
            parameters={"top_k": 5, "threshold": 0.8},
            test_queries=["What is AI?"],
        )
        result = await handler.handle_tune_parameters(request)
        assert isinstance(result, TuneParametersResponse)
        assert result.avg_score == 0.9

    async def test_handle_tune_parameters_no_session_raises(self, handler: RAGDebugHandler) -> None:
        request = TuneParametersRequest(
            session_id="",
            parameters={"top_k": 5},
        )
        with pytest.raises(Exception):
            await handler.handle_tune_parameters(request)

    async def test_handle_tune_parameters_no_params_raises(self, handler: RAGDebugHandler) -> None:
        request = TuneParametersRequest(
            session_id="sess-1",
            parameters={},
        )
        with pytest.raises(Exception):
            await handler.handle_tune_parameters(request)

    async def test_handle_export_report(self, handler: RAGDebugHandler) -> None:
        result = await handler.handle_export_report("sess-1")
        assert isinstance(result, dict)
        assert "session_id" in result

    async def test_handle_export_report_no_session_raises(self, handler: RAGDebugHandler) -> None:
        with pytest.raises(Exception):
            await handler.handle_export_report("")
