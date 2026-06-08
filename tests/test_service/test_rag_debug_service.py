"""
RAGDebugService 테스트 - 디버그 세션, 임베딩 분석, 청크 검증, 파라미터 튜닝
"""

from unittest.mock import MagicMock

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
from beanllm.service.impl.advanced.rag_debug_service_impl import RAGDebugServiceImpl


@pytest.fixture
def service() -> RAGDebugServiceImpl:
    return RAGDebugServiceImpl()


@pytest.fixture
def mock_vector_store() -> MagicMock:
    vs = MagicMock()
    vs._documents = []
    vs._embeddings = []
    return vs


@pytest.fixture
def mock_vector_store_with_data() -> MagicMock:
    vs = MagicMock()
    vs._documents = [MagicMock(content="doc1", metadata={}), MagicMock(content="doc2", metadata={})]
    vs._embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    return vs


class TestStartSession:
    @pytest.mark.asyncio
    async def test_start_session_basic(
        self, service: RAGDebugServiceImpl, mock_vector_store: MagicMock
    ) -> None:
        request = StartDebugSessionRequest(
            vector_store_id="vs-1",
            config={"vector_store": mock_vector_store},
        )
        response = await service.start_session(request)
        assert isinstance(response, DebugSessionResponse)
        assert response.session_id is not None
        assert response.vector_store_id == "vs-1"

    @pytest.mark.asyncio
    async def test_start_session_stores_session(
        self, service: RAGDebugServiceImpl, mock_vector_store: MagicMock
    ) -> None:
        request = StartDebugSessionRequest(
            vector_store_id="vs-2",
            config={"vector_store": mock_vector_store},
        )
        response = await service.start_session(request)
        assert response.session_id in service._sessions

    @pytest.mark.asyncio
    async def test_start_session_with_name(
        self, service: RAGDebugServiceImpl, mock_vector_store: MagicMock
    ) -> None:
        request = StartDebugSessionRequest(
            vector_store_id="vs-3",
            session_name="my-debug-session",
            config={"vector_store": mock_vector_store},
        )
        response = await service.start_session(request)
        assert response.session_name == "my-debug-session"

    @pytest.mark.asyncio
    async def test_start_session_no_vector_store_raises(self, service: RAGDebugServiceImpl) -> None:
        request = StartDebugSessionRequest(
            vector_store_id="vs-bad",
            config={},
        )
        with pytest.raises(ValueError, match="vector_store"):
            await service.start_session(request)

    @pytest.mark.asyncio
    async def test_start_session_status_active(
        self, service: RAGDebugServiceImpl, mock_vector_store: MagicMock
    ) -> None:
        request = StartDebugSessionRequest(
            vector_store_id="vs-4",
            config={"vector_store": mock_vector_store},
        )
        response = await service.start_session(request)
        assert response.status == "active"


class TestAnalyzeEmbeddings:
    @pytest.fixture
    def session_id(
        self,
        service: RAGDebugServiceImpl,
        mock_vector_store: MagicMock,
    ):
        import asyncio

        request = StartDebugSessionRequest(
            vector_store_id="vs-emb",
            config={"vector_store": mock_vector_store},
        )
        resp = asyncio.get_event_loop().run_until_complete(service.start_session(request))
        return resp.session_id

    @pytest.mark.asyncio
    async def test_analyze_embeddings_empty_store_raises(
        self, service: RAGDebugServiceImpl, mock_vector_store: MagicMock
    ) -> None:
        start_req = StartDebugSessionRequest(
            vector_store_id="vs-emb-1",
            config={"vector_store": mock_vector_store},
        )
        start_resp = await service.start_session(start_req)

        analyze_req = AnalyzeEmbeddingsRequest(
            session_id=start_resp.session_id,
            method="umap",
            n_clusters=3,
        )
        with pytest.raises(ValueError, match="No embeddings"):
            await service.analyze_embeddings(analyze_req)

    @pytest.mark.asyncio
    async def test_analyze_embeddings_nonexistent_session(
        self, service: RAGDebugServiceImpl
    ) -> None:
        request = AnalyzeEmbeddingsRequest(
            session_id="nonexistent-session",
            method="umap",
        )
        with pytest.raises(ValueError, match="Session not found"):
            await service.analyze_embeddings(request)

    @pytest.mark.asyncio
    async def test_analyze_embeddings_nonexistent_session_raises(
        self, service: RAGDebugServiceImpl
    ) -> None:
        analyze_req = AnalyzeEmbeddingsRequest(
            session_id="totally-nonexistent",
            method="tsne",
            n_clusters=2,
        )
        with pytest.raises(ValueError, match="Session not found"):
            await service.analyze_embeddings(analyze_req)


class TestValidateChunks:
    @pytest.mark.asyncio
    async def test_validate_chunks_empty_store_raises(
        self, service: RAGDebugServiceImpl, mock_vector_store: MagicMock
    ) -> None:
        start_req = StartDebugSessionRequest(
            vector_store_id="vs-val",
            config={"vector_store": mock_vector_store},
        )
        start_resp = await service.start_session(start_req)

        validate_req = ValidateChunksRequest(
            session_id=start_resp.session_id,
            check_size=True,
            check_overlap=True,
        )
        with pytest.raises(ValueError, match="No documents"):
            await service.validate_chunks(validate_req)

    @pytest.mark.asyncio
    async def test_validate_chunks_nonexistent_session(self, service: RAGDebugServiceImpl) -> None:
        request = ValidateChunksRequest(
            session_id="nonexistent",
            check_size=True,
        )
        with pytest.raises(ValueError, match="Session not found"):
            await service.validate_chunks(request)

    @pytest.mark.asyncio
    async def test_validate_chunks_nonexistent_session_raises(
        self, service: RAGDebugServiceImpl
    ) -> None:
        validate_req = ValidateChunksRequest(
            session_id="nonexistent-val",
            check_size=True,
            check_metadata=True,
            check_duplicates=True,
        )
        with pytest.raises(ValueError, match="Session not found"):
            await service.validate_chunks(validate_req)


class TestTuneParameters:
    @pytest.mark.asyncio
    async def test_tune_parameters_basic(
        self, service: RAGDebugServiceImpl, mock_vector_store: MagicMock
    ) -> None:
        start_req = StartDebugSessionRequest(
            vector_store_id="vs-tune",
            config={"vector_store": mock_vector_store},
        )
        start_resp = await service.start_session(start_req)

        tune_req = TuneParametersRequest(
            session_id=start_resp.session_id,
            parameters={"chunk_size": 500, "chunk_overlap": 50},
            test_queries=["What is AI?", "Explain machine learning"],
        )
        response = await service.tune_parameters(tune_req)
        assert isinstance(response, TuneParametersResponse)
        assert response.session_id == start_resp.session_id

    @pytest.mark.asyncio
    async def test_tune_parameters_nonexistent_session(self, service: RAGDebugServiceImpl) -> None:
        request = TuneParametersRequest(
            session_id="nonexistent",
            parameters={"chunk_size": 500},
        )
        with pytest.raises(ValueError, match="Session not found"):
            await service.tune_parameters(request)

    @pytest.mark.asyncio
    async def test_tune_parameters_empty_queries(
        self, service: RAGDebugServiceImpl, mock_vector_store: MagicMock
    ) -> None:
        start_req = StartDebugSessionRequest(
            vector_store_id="vs-tune-2",
            config={"vector_store": mock_vector_store},
        )
        start_resp = await service.start_session(start_req)

        tune_req = TuneParametersRequest(
            session_id=start_resp.session_id,
            parameters={"k": 5},
            test_queries=[],
        )
        response = await service.tune_parameters(tune_req)
        assert isinstance(response, TuneParametersResponse)


class TestExportReport:
    @pytest.mark.asyncio
    async def test_export_report_basic(
        self, service: RAGDebugServiceImpl, mock_vector_store: MagicMock
    ) -> None:
        start_req = StartDebugSessionRequest(
            vector_store_id="vs-exp",
            config={"vector_store": mock_vector_store},
        )
        start_resp = await service.start_session(start_req)

        report = await service.export_report(start_resp.session_id)
        assert isinstance(report, dict)
        # Report contains session info nested under "session" key or at top-level
        assert "session" in report or "session_id" in report

    @pytest.mark.asyncio
    async def test_export_report_nonexistent_session(self, service: RAGDebugServiceImpl) -> None:
        with pytest.raises(ValueError, match="Session not found"):
            await service.export_report("nonexistent-session")
