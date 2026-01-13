"""
IRAGDebugService - RAG 디버깅 서비스 인터페이스
SOLID 원칙:
- ISP: RAG 디버깅 관련 메서드만 포함
- DIP: 인터페이스에 의존
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ..dto.request.ml.rag_debug_request import (
    AnalyzeEmbeddingsRequest,
    StartDebugSessionRequest,
    TuneParametersRequest,
    ValidateChunksRequest,
)
from ..dto.response.ml.rag_debug_response import (
    AnalyzeEmbeddingsResponse,
    DebugSessionResponse,
    TuneParametersResponse,
    ValidateChunksResponse,
)


class IRAGDebugService(ABC):
    """
    RAG 디버깅 서비스 인터페이스

    책임:
    - RAG 파이프라인 디버깅 비즈니스 로직 정의
    - Embedding 분석, 청크 검증, 파라미터 튜닝 등

    SOLID:
    - ISP: RAG 디버깅 관련 메서드만
    - DIP: 구현체가 아닌 인터페이스에 의존
    """

    @abstractmethod
    async def start_session(
        self, request: StartDebugSessionRequest
    ) -> DebugSessionResponse:
        """
        디버그 세션 시작

        Args:
            request: 세션 시작 요청 DTO

        Returns:
            DebugSessionResponse: 세션 정보 응답
        """
        pass

    @abstractmethod
    async def analyze_embeddings(
        self, request: AnalyzeEmbeddingsRequest
    ) -> AnalyzeEmbeddingsResponse:
        """
        Embedding 분석 (UMAP/t-SNE, 클러스터링)

        Args:
            request: Embedding 분석 요청 DTO

        Returns:
            AnalyzeEmbeddingsResponse: 분석 결과 응답
        """
        pass

    @abstractmethod
    async def validate_chunks(
        self, request: ValidateChunksRequest
    ) -> ValidateChunksResponse:
        """
        청크 검증 (크기, 중복, 메타데이터)

        Args:
            request: 청크 검증 요청 DTO

        Returns:
            ValidateChunksResponse: 검증 결과 응답
        """
        pass

    @abstractmethod
    async def tune_parameters(
        self, request: TuneParametersRequest
    ) -> TuneParametersResponse:
        """
        파라미터 실시간 튜닝

        Args:
            request: 파라미터 튜닝 요청 DTO

        Returns:
            TuneParametersResponse: 튜닝 결과 응답
        """
        pass

    @abstractmethod
    async def export_report(self, session_id: str) -> Dict[str, Any]:
        """
        디버그 리포트 내보내기

        Args:
            session_id: 세션 ID

        Returns:
            Dict: 리포트 데이터
        """
        pass
