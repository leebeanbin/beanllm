"""
RAGDebug Facade - RAG 디버깅을 위한 간단한 공개 API
책임: 사용하기 쉬운 인터페이스 제공, 내부적으로는 Handler/Service 사용
SOLID 원칙:
- Facade 패턴: 복잡한 내부 구조를 단순한 인터페이스로
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

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
from beanllm.utils.logging import get_logger

if TYPE_CHECKING:
    from beanllm.domain.vector_stores import BaseVectorStore
    from beanllm.handler.advanced.rag_debug_handler import RAGDebugHandler

logger = get_logger(__name__)


class RAGDebug:
    """
    RAG 디버깅 Facade

    사용하기 쉬운 공개 API를 제공하면서 내부적으로는 Handler/Service 사용

    Example:
        ```python
        # 디버그 세션 시작
        debug = RAGDebug(vector_store)

        # Embedding 분석
        analysis = await debug.analyze_embeddings(method="umap", n_clusters=5)

        # 청크 검증
        validation = await debug.validate_chunks()

        # 파라미터 튜닝
        tuning = await debug.tune_parameters(
            {"top_k": 10, "score_threshold": 0.7},
            test_queries=["query1", "query2"]
        )

        # 리포트 내보내기
        report = await debug.export_report("output/")
        ```
    """

    def __init__(
        self,
        vector_store: "BaseVectorStore",
        session_name: Optional[str] = None,
    ) -> None:
        """
        Args:
            vector_store: 디버깅할 VectorStore
            session_name: 세션 이름 (optional)
        """
        self.vector_store = vector_store
        self.session_name = session_name
        self.session_id: Optional[str] = None

        # Handler 초기화 (의존성 주입)
        self._init_handler()

    def _init_handler(self) -> None:
        """Handler 초기화 (DI Container 사용)"""
        from beanllm.utils.di_container import get_container

        container = get_container()
        service_factory = container.get_service_factory()
        handler_factory = container.get_handler_factory(service_factory)

        # RAGDebugHandler 생성
        self._handler: "RAGDebugHandler" = handler_factory.create_rag_debug_handler()

    async def start(self) -> DebugSessionResponse:
        """
        디버그 세션 시작

        Returns:
            DebugSessionResponse: 세션 정보

        Raises:
            RuntimeError: 세션 시작 실패 시
        """
        logger.info("Starting RAG debug session")

        request = StartDebugSessionRequest(
            vector_store_id=str(id(self.vector_store)),
            session_name=self.session_name,
            config={"vector_store": self.vector_store},
        )

        response = await self._handler.handle_start_session(request)
        self.session_id = response.session_id

        logger.info(
            f"Debug session started: {self.session_id}, "
            f"{response.num_documents} docs, {response.num_embeddings} embeddings"
        )

        return response

    async def analyze_embeddings(
        self,
        method: str = "umap",
        n_clusters: int = 5,
        detect_outliers: bool = True,
        sample_size: Optional[int] = None,
    ) -> AnalyzeEmbeddingsResponse:
        """
        Embedding 분석 (UMAP/t-SNE, 클러스터링)

        Args:
            method: 차원 축소 방법 ("umap" or "tsne")
            n_clusters: 클러스터 수
            detect_outliers: 이상치 탐지 여부
            sample_size: 샘플 크기 (None이면 전체)

        Returns:
            AnalyzeEmbeddingsResponse: 분석 결과

        Raises:
            RuntimeError: 세션이 시작되지 않았거나 분석 실패 시
        """
        if not self.session_id:
            raise RuntimeError("Session not started. Call start() first.")

        logger.info(f"Analyzing embeddings: method={method}, n_clusters={n_clusters}")

        request = AnalyzeEmbeddingsRequest(
            session_id=self.session_id,
            method=method,
            n_clusters=n_clusters,
            detect_outliers=detect_outliers,
            sample_size=sample_size,
        )

        response = await self._handler.handle_analyze_embeddings(request)

        logger.info(
            f"Analysis completed: {response.num_clusters} clusters, "
            f"{len(response.outliers)} outliers, "
            f"silhouette={response.silhouette_score:.4f}"
        )

        return response

    async def validate_chunks(
        self,
        size_threshold: int = 2000,
        check_size: bool = True,
        check_overlap: bool = True,
        check_metadata: bool = True,
        check_duplicates: bool = True,
    ) -> ValidateChunksResponse:
        """
        청크 검증 (크기, 중복, 메타데이터)

        Args:
            size_threshold: 최대 청크 크기
            check_size: 크기 검증 여부
            check_overlap: Overlap 검증 여부
            check_metadata: 메타데이터 검증 여부
            check_duplicates: 중복 검증 여부

        Returns:
            ValidateChunksResponse: 검증 결과

        Raises:
            RuntimeError: 세션이 시작되지 않았거나 검증 실패 시
        """
        if not self.session_id:
            raise RuntimeError("Session not started. Call start() first.")

        logger.info("Validating chunks")

        request = ValidateChunksRequest(
            session_id=self.session_id,
            check_size=check_size,
            check_overlap=check_overlap,
            check_metadata=check_metadata,
            check_duplicates=check_duplicates,
            size_threshold=size_threshold,
        )

        response = await self._handler.handle_validate_chunks(request)

        logger.info(
            f"Validation completed: {response.total_chunks} total, "
            f"{response.valid_chunks} valid, {len(response.issues)} issues"
        )

        return response

    async def tune_parameters(
        self,
        parameters: Dict[str, Any],
        test_queries: Optional[List[str]] = None,
    ) -> TuneParametersResponse:
        """
        파라미터 실시간 튜닝

        Args:
            parameters: 테스트할 파라미터
                예: {"top_k": 10, "score_threshold": 0.7}
            test_queries: 테스트 쿼리 목록

        Returns:
            TuneParametersResponse: 튜닝 결과

        Raises:
            RuntimeError: 세션이 시작되지 않았거나 튜닝 실패 시
        """
        if not self.session_id:
            raise RuntimeError("Session not started. Call start() first.")

        logger.info(f"Tuning parameters: {parameters}")

        request = TuneParametersRequest(
            session_id=self.session_id,
            parameters=parameters,
            test_queries=test_queries or [],
        )

        response = await self._handler.handle_tune_parameters(request)

        logger.info(
            f"Tuning completed: avg_score={response.avg_score:.4f}, "
            f"recommendations={len(response.recommendations)}"
        )

        return response

    async def export_report(
        self, output_dir: str, formats: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        디버그 리포트 내보내기

        Args:
            output_dir: 출력 디렉토리
            formats: 내보낼 포맷 목록 (None이면 ["json", "markdown", "html"])

        Returns:
            Dict[str, str]: 포맷별 파일 경로

        Raises:
            RuntimeError: 세션이 시작되지 않았거나 내보내기 실패 시
        """
        if not self.session_id:
            raise RuntimeError("Session not started. Call start() first.")

        logger.info(f"Exporting report to: {output_dir}")

        # Get report data
        report_data = await self._handler.handle_export_report(self.session_id)

        # Export to files
        from beanllm.domain.rag_debug import DebugReportExporter

        results = DebugReportExporter.create_full_report(
            session_data=report_data, output_dir=output_dir, formats=formats
        )

        logger.info(f"Report exported: {len(results)} files created")

        return results

    async def run_full_analysis(
        self,
        analyze_embeddings: bool = True,
        validate_chunks: bool = True,
        tune_parameters: bool = False,
        tuning_params: Optional[Dict[str, Any]] = None,
        test_queries: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        전체 분석 실행 (원스톱)

        Args:
            analyze_embeddings: Embedding 분석 실행 여부
            validate_chunks: 청크 검증 실행 여부
            tune_parameters: 파라미터 튜닝 실행 여부
            tuning_params: 튜닝할 파라미터 (tune_parameters=True일 때)
            test_queries: 테스트 쿼리 (tune_parameters=True일 때)

        Returns:
            Dict: 전체 분석 결과

        Example:
            ```python
            results = await debug.run_full_analysis(
                analyze_embeddings=True,
                validate_chunks=True,
                tune_parameters=True,
                tuning_params={"top_k": 10},
                test_queries=["test query"]
            )
            ```
        """
        logger.info("Running full RAG debug analysis")

        # Start session
        session_info = await self.start()

        results = {
            "session": session_info,
        }

        # Analyze embeddings
        if analyze_embeddings:
            logger.info("Step 1/3: Analyzing embeddings...")
            results["embedding_analysis"] = await self.analyze_embeddings()

        # Validate chunks
        if validate_chunks:
            logger.info("Step 2/3: Validating chunks...")
            results["chunk_validation"] = await self.validate_chunks()

        # Tune parameters
        if tune_parameters:
            logger.info("Step 3/3: Tuning parameters...")
            if not tuning_params:
                raise ValueError("tuning_params required when tune_parameters=True")
            results["parameter_tuning"] = await self.tune_parameters(
                parameters=tuning_params, test_queries=test_queries
            )

        logger.info("Full analysis completed")

        return results
