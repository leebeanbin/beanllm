"""
RAGDebugServiceImpl - RAG 디버깅 서비스 구현체
SOLID 원칙:
- SRP: RAG 디버깅 비즈니스 로직만 담당
- DIP: 인터페이스에 의존
"""

from __future__ import annotations

from typing import Any, Dict

from beanllm.domain.rag_debug import (
    ChunkValidator,
    DebugSession,
    EmbeddingAnalyzer,
    ParameterTuner,
)
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
from beanllm.service.rag_debug_service import IRAGDebugService
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class RAGDebugServiceImpl(IRAGDebugService):
    """
    RAG 디버깅 서비스 구현체

    책임:
    - RAG 디버깅 비즈니스 로직
    - DebugSession 관리
    - Domain logic orchestration
    """

    def __init__(self) -> None:
        """Initialize service with session storage"""
        # Session storage: session_id -> DebugSession
        self._sessions: Dict[str, DebugSession] = {}
        logger.info("RAGDebugService initialized")

    async def start_session(self, request: StartDebugSessionRequest) -> DebugSessionResponse:
        """
        디버그 세션 시작

        Args:
            request: 세션 시작 요청

        Returns:
            DebugSessionResponse: 세션 정보
        """
        logger.info(f"Starting debug session for vector_store: {request.vector_store_id}")

        # Get VectorStore from registry or provided instance
        # For now, we expect vector_store to be passed in config
        vector_store = request.config.get("vector_store")
        if not vector_store:
            raise ValueError("vector_store must be provided in config")

        # Create DebugSession
        session = DebugSession(
            vector_store=vector_store,
            session_name=request.session_name,
        )

        # Store session
        self._sessions[session.session_id] = session

        # Get metadata
        metadata = session.get_metadata()

        # Create response
        response = DebugSessionResponse(
            session_id=session.session_id,
            session_name=session.session_name,
            vector_store_id=request.vector_store_id,
            num_documents=metadata["num_documents"],
            num_embeddings=metadata["num_embeddings"],
            embedding_dim=metadata["embedding_dim"],
            status="active",
            created_at=metadata["created_at"],
            metadata=metadata,
        )

        logger.info(f"Debug session started: {session.session_id}")
        return response

    async def analyze_embeddings(
        self, request: AnalyzeEmbeddingsRequest
    ) -> AnalyzeEmbeddingsResponse:
        """
        Embedding 분석 (UMAP/t-SNE, clustering)

        Args:
            request: Embedding 분석 요청

        Returns:
            AnalyzeEmbeddingsResponse: 분석 결과
        """
        logger.info(
            f"Analyzing embeddings for session: {request.session_id}, "
            f"method={request.method}, n_clusters={request.n_clusters}"
        )

        # Get session
        session = self._sessions.get(request.session_id)
        if not session:
            raise ValueError(f"Session not found: {request.session_id}")

        # Get embeddings
        embeddings = session.get_embeddings()

        if not embeddings:
            raise ValueError("No embeddings found in VectorStore")

        # Sample if requested
        if request.sample_size and request.sample_size < len(embeddings):
            import random

            indices = random.sample(range(len(embeddings)), request.sample_size)
            embeddings = [embeddings[i] for i in indices]
            logger.info(f"Sampled {request.sample_size} embeddings from {len(embeddings)}")

        # Analyze embeddings
        analyzer = EmbeddingAnalyzer()
        analysis = analyzer.analyze(
            embeddings=embeddings,
            method=request.method,
            n_clusters=request.n_clusters,
            detect_outliers=request.detect_outliers,
        )

        # Cache results
        session.cache_result("embedding_analysis", analysis)

        # Create response
        response = AnalyzeEmbeddingsResponse(
            session_id=request.session_id,
            method=analysis["method"],
            num_clusters=analysis["cluster_stats"]["n_clusters"],
            cluster_labels=analysis["labels"],
            cluster_sizes=analysis["cluster_stats"]["cluster_sizes"],
            outliers=analysis["outliers"],
            reduced_embeddings=analysis["reduced_embeddings"],
            silhouette_score=analysis["silhouette_score"],
            metadata=analysis["cluster_stats"],
        )

        logger.info(
            f"Embedding analysis completed: {response.num_clusters} clusters, "
            f"{len(response.outliers)} outliers"
        )
        return response

    async def validate_chunks(self, request: ValidateChunksRequest) -> ValidateChunksResponse:
        """
        청크 검증 (크기, 중복, 메타데이터)

        Args:
            request: 청크 검증 요청

        Returns:
            ValidateChunksResponse: 검증 결과
        """
        logger.info(f"Validating chunks for session: {request.session_id}")

        # Get session
        session = self._sessions.get(request.session_id)
        if not session:
            raise ValueError(f"Session not found: {request.session_id}")

        # Get documents
        documents = session.get_documents()

        if not documents:
            raise ValueError("No documents found in VectorStore")

        # Validate chunks
        validator = ChunkValidator(
            min_chunk_size=100,
            max_chunk_size=request.size_threshold,
            overlap_threshold=0.9,
        )

        validation_result = validator.validate_all(documents)

        # Cache results
        session.cache_result("chunk_validation", validation_result)

        # Create response
        response = ValidateChunksResponse(
            session_id=request.session_id,
            total_chunks=validation_result["total_chunks"],
            valid_chunks=validation_result["valid_chunks"],
            issues=validation_result["size_issues"] + validation_result["metadata_issues"],
            size_distribution=validation_result["size_distribution"],
            overlap_stats=validation_result["overlap_stats"],
            duplicate_chunks=validation_result["duplicate_chunks"],
            recommendations=validation_result["recommendations"],
        )

        logger.info(
            f"Chunk validation completed: {response.total_chunks} total, "
            f"{response.valid_chunks} valid, {len(response.issues)} issues"
        )
        return response

    async def tune_parameters(self, request: TuneParametersRequest) -> TuneParametersResponse:
        """
        파라미터 실시간 튜닝

        Args:
            request: 파라미터 튜닝 요청

        Returns:
            TuneParametersResponse: 튜닝 결과
        """
        logger.info(
            f"Tuning parameters for session: {request.session_id}, " f"params={request.parameters}"
        )

        # Get session
        session = self._sessions.get(request.session_id)
        if not session:
            raise ValueError(f"Session not found: {request.session_id}")

        # Get baseline params (from cache or default)
        baseline_params = session.get_cached_result("baseline_params") or {
            "top_k": 4,
            "score_threshold": 0.0,
        }

        # Create tuner
        tuner = ParameterTuner(vector_store=session.vector_store, baseline_params=baseline_params)

        # Test parameters with test queries
        test_results = []
        if request.test_queries:
            for query in request.test_queries:
                result = tuner.compare_with_baseline(query, request.parameters)
                test_results.append(result)

        # Compute average score
        avg_score = 0.0
        if test_results:
            avg_score = sum(r["new"]["avg_score"] for r in test_results) / len(test_results)

        # Compare with baseline
        baseline_score = 0.0
        if test_results:
            baseline_score = sum(r["baseline"]["avg_score"] for r in test_results) / len(
                test_results
            )

        comparison = {
            "baseline_score": baseline_score,
            "new_score": avg_score,
            "improvement_pct": (
                (avg_score - baseline_score) / baseline_score * 100 if baseline_score > 0 else 0.0
            ),
        }

        # Generate recommendations
        recommendations = []
        if comparison["improvement_pct"] > 5:
            recommendations.append("✅ New parameters show significant improvement!")
        elif comparison["improvement_pct"] < -5:
            recommendations.append("⚠️  New parameters perform worse than baseline. Keep baseline.")
        else:
            recommendations.append("💡 Marginal difference. A/B testing recommended.")

        # Cache results
        session.cache_result("parameter_tuning", request.parameters)

        # Create response
        response = TuneParametersResponse(
            session_id=request.session_id,
            parameters=request.parameters,
            test_results=test_results,
            avg_score=avg_score,
            comparison_with_baseline=comparison,
            recommendations=recommendations,
        )

        logger.info(
            f"Parameter tuning completed: avg_score={avg_score:.4f}, "
            f"improvement={comparison['improvement_pct']:.2f}%"
        )
        return response

    async def export_report(self, session_id: str) -> Dict[str, Any]:
        """
        디버그 리포트 내보내기

        Args:
            session_id: 세션 ID

        Returns:
            Dict: 리포트 데이터
        """
        logger.info(f"Exporting report for session: {session_id}")

        # Get session
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        # Collect all cached results
        report_data = {
            "session": session.to_dict(),
            "metadata": session.get_metadata(),
            "embedding_analysis": session.get_cached_result("embedding_analysis"),
            "chunk_validation": session.get_cached_result("chunk_validation"),
            "parameter_tuning": session.get_cached_result("parameter_tuning"),
        }

        logger.info("Report data collected")
        return report_data
