"""
Chunking Experimenter - 청킹 전략 실험 및 개선 도구 (coordinator).

Delegates to experiment_runner, experiment_feedback, experiment_report.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

try:
    from beanllm.utils.logging import get_logger
except ImportError:
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)

from beanllm.domain.rag_debug.experiment_feedback import (
    ChunkFeedback,
    add_feedback as _add_feedback_impl,
    get_feedback_summary as _get_feedback_summary_impl,
    improve_from_feedback as _improve_from_feedback_impl,
)
from beanllm.domain.rag_debug.experiment_report import get_comparison_report as _get_comparison_report_impl
from beanllm.domain.rag_debug.experiment_runner import (
    ChunkingResult,
    build_grid_configs as _build_grid_configs,
    run_experiment as _run_experiment_impl,
)

logger = get_logger(__name__)

__all__ = ["ChunkingExperimenter", "ChunkingResult", "ChunkFeedback"]


class ChunkingExperimenter:
    """청킹 전략 실험 및 개선 도구 (coordinator)."""

    def __init__(
        self,
        documents: List[str],
        test_queries: List[str],
        embedding_function: Optional[Callable[[str], List[float]]] = None,
        ground_truth: Optional[Dict[str, List[int]]] = None,
        **kwargs: Any,
    ):
        """
        ChunkingExperimenter 초기화

        Args:
            documents: 원본 문서 목록
            test_queries: 테스트용 쿼리 목록
            embedding_function: 임베딩 함수 (str -> List[float])
            ground_truth: 각 쿼리의 정답 청크 인덱스
                예: {"query1": [0, 2, 5], "query2": [1, 3]}
            **kwargs: 추가 옵션
        """
        self.documents = documents
        self.test_queries = test_queries
        self.embedding_function = embedding_function
        self.ground_truth = ground_truth or {}
        self.kwargs = kwargs

        # 실험 결과 저장
        self._results: List[ChunkingResult] = []
        self._feedbacks: List[ChunkFeedback] = []
        self._current_chunks: Dict[str, List[str]] = {}  # strategy_name -> chunks

        logger.info(
            f"ChunkingExperimenter initialized: {len(documents)} docs, {len(test_queries)} queries"
        )

    def run_experiment(
        self, config: Dict[str, Any], strategy_name: Optional[str] = None
    ) -> ChunkingResult:
        """Run a single chunking strategy experiment."""
        result = _run_experiment_impl(
            self.documents,
            self.test_queries,
            config,
            self.embedding_function,
            self.ground_truth,
            strategy_name,
        )
        name = result.strategy_name
        self._results.append(result)
        self._current_chunks[name] = result.chunks
        return result

    def compare_strategies(self, configs: List[Dict[str, Any]]) -> List[ChunkingResult]:
        """Compare multiple chunking strategies."""
        results = []
        for i, config in enumerate(configs):
            name = config.get("name", f"strategy_{i}")
            result = self.run_experiment(config, strategy_name=name)
            results.append(result)
        results.sort(key=lambda r: r.avg_retrieval_score, reverse=True)
        return results

    def grid_search(
        self,
        splitter_type: str = "recursive",
        chunk_sizes: Optional[List[int]] = None,
        chunk_overlaps: Optional[List[int]] = None,
        **fixed_params: Any,
    ) -> List[ChunkingResult]:
        """Grid search over chunk parameters."""
        if chunk_sizes is None:
            chunk_sizes = [256, 512, 1000, 2000]
        if chunk_overlaps is None:
            chunk_overlaps = [0, 50, 100, 200]
        configs = _build_grid_configs(
            splitter_type=splitter_type,
            chunk_sizes=chunk_sizes,
            chunk_overlaps=chunk_overlaps,
            **fixed_params,
        )
        logger.info(f"Grid search: {len(configs)} configurations")
        return self.compare_strategies(configs)

    def find_best_strategy(self) -> Optional[Dict[str, Any]]:
        """
        최적 전략 찾기

        Returns:
            최적 전략 정보 (없으면 None)
        """
        if not self._results:
            logger.warning("No experiment results. Run experiments first.")
            return None

        best = max(self._results, key=lambda r: r.avg_retrieval_score)

        return {
            "strategy": best.strategy_name,
            "config": best.strategy_config,
            "score": best.avg_retrieval_score,
            "chunk_count": best.chunk_count,
            "avg_chunk_size": best.avg_chunk_size,
        }

    def add_feedback(
        self,
        query: str,
        chunk_id: str,
        rating: float,
        feedback_type: str = "relevance",
        comment: Optional[str] = None,
    ) -> None:
        """Add chunk feedback."""
        _add_feedback_impl(
            self._feedbacks,
            self._current_chunks,
            query,
            chunk_id,
            rating,
            feedback_type=feedback_type,
            comment=comment,
        )

    def get_feedback_summary(self) -> Dict[str, Any]:
        """Summarize feedbacks."""
        return _get_feedback_summary_impl(self._feedbacks)

    def improve_from_feedback(self, min_rating_threshold: float = 0.3) -> Dict[str, Any]:
        """Get improvement suggestions from feedback."""
        return _improve_from_feedback_impl(
            self._feedbacks,
            self.find_best_strategy,
            min_rating_threshold=min_rating_threshold,
        )

    def get_comparison_report(self) -> str:
        """Build comparison report."""
        return _get_comparison_report_impl(
            self._results,
            self.find_best_strategy,
            self.get_feedback_summary,
        )

    def __repr__(self) -> str:
        return (
            f"ChunkingExperimenter("
            f"docs={len(self.documents)}, "
            f"queries={len(self.test_queries)}, "
            f"experiments={len(self._results)})"
        )
