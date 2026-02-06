"""
RAG Debug - RAG 파이프라인 디버깅 및 개선 도구

기존 도구:
- DebugSession: 세션 관리 및 데이터 수집
- EmbeddingAnalyzer: UMAP/t-SNE 차원 축소, HDBSCAN 클러스터링
- ChunkValidator: 청크 검증 (크기, 중복, 메타데이터, overlap)
- SimilarityTester: 쿼리 시뮬레이션 및 검색 전략 비교
- ParameterTuner: 실시간 파라미터 튜닝
- DebugReportExporter: 리포트 내보내기 (JSON, Markdown, HTML)

청킹 개선 도구:
- ChunkingExperimenter: 청킹 전략 실험, A/B 테스트, 피드백 기반 개선

통합 개선 루프:
- RAGImprovementLoop: 청킹 → 평가 → 피드백 → 개선의 전체 사이클 관리

Example:
    ```python
    from beanllm.domain.rag_debug import RAGImprovementLoop

    # 통합 개선 루프 생성
    loop = RAGImprovementLoop(
        documents=documents,
        test_queries=queries,
        embedding_function=embed_fn
    )

    # 1. 초기 실험
    loop.run_initial_experiments()

    # 2. RAG 평가
    result = loop.evaluate_pipeline(
        query="What is RAG?",
        response="RAG is...",
        contexts=["context..."]
    )

    # 3. Human 피드백
    loop.add_human_feedback(query="What is RAG?", rating=0.8)

    # 4. 자동 개선 사이클
    improved = loop.run_full_cycle(max_iterations=3)
    print(f"Improvement: {improved['total_improvement']:.2%}")

    # 5. 리포트
    print(loop.export_full_report())
    ```
"""

from beanllm.domain.rag_debug.chunk_validator import ChunkValidator
from beanllm.domain.rag_debug.chunking_experimenter import (
    ChunkFeedback,
    ChunkingExperimenter,
    ChunkingResult,
)
from beanllm.domain.rag_debug.debug_session import DebugSession
from beanllm.domain.rag_debug.embedding_analyzer import EmbeddingAnalyzer
from beanllm.domain.rag_debug.export import DebugReportExporter
from beanllm.domain.rag_debug.improvement_loop import (
    ImprovementCycle,
    ImprovementPlan,
    RAGImprovementLoop,
)
from beanllm.domain.rag_debug.parameter_tuner import ParameterTuner
from beanllm.domain.rag_debug.similarity_tester import SimilarityTester

__all__ = [
    # 기존 도구
    "DebugSession",
    "EmbeddingAnalyzer",
    "ChunkValidator",
    "SimilarityTester",
    "ParameterTuner",
    "DebugReportExporter",
    # 청킹 개선 도구
    "ChunkingExperimenter",
    "ChunkingResult",
    "ChunkFeedback",
    # 통합 개선 루프
    "RAGImprovementLoop",
    "ImprovementCycle",
    "ImprovementPlan",
]
