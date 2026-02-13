"""
RAGDebugHandler - RAG 디버깅 Handler
SOLID 원칙:
- SRP: 검증 및 에러 처리만 담당
- DIP: 인터페이스에 의존
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from beanllm.decorators.error_handler import handle_errors
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
from beanllm.handler.base_handler import BaseHandler
from beanllm.utils.logging import get_logger

if TYPE_CHECKING:
    from beanllm.service.rag_debug_service import IRAGDebugService

logger = get_logger(__name__)


class RAGDebugHandler(BaseHandler["IRAGDebugService"]):
    """
    RAG 디버깅 Handler

    책임:
    - 요청 검증
    - 에러 처리 (@handle_errors 데코레이터)
    - 응답 포매팅
    """

    def __init__(self, service: "IRAGDebugService") -> None:
        """
        Args:
            service: RAG Debug 서비스
        """
        super().__init__(service)

    @handle_errors(error_message="Failed to start debug session")
    async def handle_start_session(self, request: StartDebugSessionRequest) -> DebugSessionResponse:
        """
        디버그 세션 시작 처리

        Args:
            request: 세션 시작 요청

        Returns:
            DebugSessionResponse: 세션 정보

        Raises:
            ValueError: Validation 실패 시
        """
        if not request.vector_store_id:
            raise ValueError("vector_store_id is required")

        if not request.config or "vector_store" not in request.config:
            raise ValueError("vector_store must be provided in config")

        return await self._service.start_session(request)

    @handle_errors(error_message="Failed to analyze embeddings")
    async def handle_analyze_embeddings(
        self, request: AnalyzeEmbeddingsRequest
    ) -> AnalyzeEmbeddingsResponse:
        """
        Embedding 분석 처리

        Args:
            request: Embedding 분석 요청

        Returns:
            AnalyzeEmbeddingsResponse: 분석 결과

        Raises:
            ValueError: Validation 실패 시
        """
        if not request.session_id:
            raise ValueError("session_id is required")

        if request.method not in ["umap", "tsne"]:
            raise ValueError("method must be 'umap' or 'tsne'")

        if request.n_clusters <= 0:
            raise ValueError("n_clusters must be positive")

        return await self._service.analyze_embeddings(request)

    @handle_errors(error_message="Failed to validate chunks")
    async def handle_validate_chunks(
        self, request: ValidateChunksRequest
    ) -> ValidateChunksResponse:
        """
        청크 검증 처리

        Args:
            request: 청크 검증 요청

        Returns:
            ValidateChunksResponse: 검증 결과

        Raises:
            ValueError: Validation 실패 시
        """
        if not request.session_id:
            raise ValueError("session_id is required")

        if request.size_threshold <= 0:
            raise ValueError("size_threshold must be positive")

        return await self._service.validate_chunks(request)

    @handle_errors(error_message="Failed to tune parameters")
    async def handle_tune_parameters(
        self, request: TuneParametersRequest
    ) -> TuneParametersResponse:
        """
        파라미터 튜닝 처리

        Args:
            request: 파라미터 튜닝 요청

        Returns:
            TuneParametersResponse: 튜닝 결과

        Raises:
            ValueError: Validation 실패 시
        """
        if not request.session_id:
            raise ValueError("session_id is required")

        if not request.parameters:
            raise ValueError("parameters dictionary is required")

        top_k = request.parameters.get("top_k")
        if top_k is not None and isinstance(top_k, (int, float)) and top_k <= 0:
            raise ValueError("top_k must be positive")

        score_threshold = request.parameters.get("score_threshold")
        if (
            score_threshold is not None
            and isinstance(score_threshold, (int, float))
            and not 0 <= score_threshold <= 1
        ):
            raise ValueError("score_threshold must be between 0 and 1")

        return await self._service.tune_parameters(request)

    @handle_errors(error_message="Failed to export report")
    async def handle_export_report(self, session_id: str) -> Dict[str, Any]:
        """
        리포트 내보내기 처리

        Args:
            session_id: 세션 ID

        Returns:
            Dict: 리포트 데이터

        Raises:
            ValueError: Validation 실패 시
        """
        if not session_id:
            raise ValueError("session_id is required")

        return await self._service.export_report(session_id)
