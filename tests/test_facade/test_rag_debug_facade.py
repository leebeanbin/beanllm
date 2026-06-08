"""
RAGDebug Facade 테스트
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from beanllm.dto.response.ml.rag_debug_response import (
    AnalyzeEmbeddingsResponse,
    DebugSessionResponse,
    TuneParametersResponse,
    ValidateChunksResponse,
)
from beanllm.facade.advanced.rag_debug_facade import RAGDebug
from beanllm.handler.advanced.rag_debug_handler import RAGDebugHandler


def _make_handler() -> MagicMock:
    handler = AsyncMock(spec=RAGDebugHandler)
    handler.handle_start_session.return_value = DebugSessionResponse(
        session_id="sess-1",
        session_name="test-session",
        vector_store_id="vs-1",
        num_documents=100,
        num_embeddings=100,
        embedding_dim=384,
        status="active",
        created_at="2026-01-01T00:00:00",
    )
    handler.handle_analyze_embeddings.return_value = AnalyzeEmbeddingsResponse(
        session_id="sess-1",
        method="umap",
        num_clusters=3,
        cluster_labels=[0, 1, 2],
        cluster_sizes={0: 30, 1: 40, 2: 30},
        outliers=[],
        silhouette_score=0.72,
    )
    handler.handle_validate_chunks.return_value = ValidateChunksResponse(
        session_id="sess-1",
        total_chunks=100,
        valid_chunks=96,
        issues=[],
        size_distribution={"small": 5, "medium": 80, "large": 15},
    )
    handler.handle_tune_parameters.return_value = TuneParametersResponse(
        session_id="sess-1",
        parameters={"top_k": 7, "threshold": 0.75},
        test_results=[{"query": "test", "score": 0.91}],
        avg_score=0.91,
    )
    handler.handle_export_report.return_value = {
        "session_id": "sess-1",
        "status": "exported",
        "path": "/tmp/report.json",
    }
    return handler


class TestRAGDebugFacade:
    @pytest.fixture
    def debug(self) -> RAGDebug:
        mock_vs = MagicMock()
        mock_vs.collection_name = "test_collection"
        debug = object.__new__(RAGDebug)
        debug.vector_store = mock_vs
        debug.session_name = "test-session"
        debug.session_id = None
        debug._handler = _make_handler()
        return debug

    async def test_start_session(self, debug: RAGDebug) -> None:
        result = await debug.start()
        assert isinstance(result, DebugSessionResponse)
        assert result.session_id == "sess-1"
        assert debug.session_id == "sess-1"

    async def test_analyze_embeddings(self, debug: RAGDebug) -> None:
        await debug.start()
        result = await debug.analyze_embeddings(method="umap", n_clusters=3)
        assert isinstance(result, AnalyzeEmbeddingsResponse)
        assert result.num_clusters == 3
        assert result.silhouette_score == 0.72

    async def test_validate_chunks(self, debug: RAGDebug) -> None:
        await debug.start()
        result = await debug.validate_chunks()
        assert isinstance(result, ValidateChunksResponse)
        assert result.total_chunks == 100
        assert result.valid_chunks == 96

    async def test_tune_parameters(self, debug: RAGDebug) -> None:
        await debug.start()
        result = await debug.tune_parameters(
            parameters={"top_k": 7, "threshold": 0.75},
            test_queries=["What is RAG?"],
        )
        assert isinstance(result, TuneParametersResponse)
        assert result.avg_score == 0.91

    async def test_export_report(self, debug: RAGDebug) -> None:
        await debug.start()
        result = await debug.export_report(output_dir="/tmp")
        assert isinstance(result, dict)
        assert len(result) > 0

    async def test_debug_has_handler(self, debug: RAGDebug) -> None:
        assert debug._handler is not None

    async def test_session_id_set_after_start(self, debug: RAGDebug) -> None:
        assert debug.session_id is None
        await debug.start()
        assert debug.session_id == "sess-1"
