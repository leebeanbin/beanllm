"""
RAG Improvement Loop - 통합 RAG 개선 피드백 루프

청킹 → 평가 → 피드백 → 개선의 지속적인 사이클을 관리합니다.
Coordinator: delegates to loop_phases, loop_cycle, loop_report.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

try:
    from beanllm.utils.logging import get_logger
except ImportError:

    def get_logger(name: str) -> logging.Logger:  # type: ignore[misc]
        return logging.getLogger(name)


from beanllm.domain.rag_debug.chunking_experimenter import (
    ChunkingExperimenter,
    ChunkingResult,
)
from beanllm.domain.rag_debug.loop_cycle import (
    ImprovementCycle,
    ImprovementPlan,
    get_current_score as _get_current_score_impl,
    run_full_cycle as _run_full_cycle_impl,
    run_improvement_cycle_step as _run_improvement_cycle_step_impl,
)
from beanllm.domain.rag_debug.loop_phases import (
    add_comparison_feedback as _add_comparison_feedback_impl,
    add_human_feedback as _add_human_feedback_impl,
    get_improvement_plan as _get_improvement_plan_impl,
    run_initial_experiments as _run_initial_experiments_impl,
)
from beanllm.domain.rag_debug.loop_report import (
    export_full_report as _export_full_report_impl,
)

try:
    from beanllm.domain.evaluation.unified_evaluator import (
        ImprovementSuggestion,
        UnifiedEvaluator,
    )
except ImportError:
    UnifiedEvaluator = None  # type: ignore
    ImprovementSuggestion = None  # type: ignore


logger = get_logger(__name__)

__all__ = ["RAGImprovementLoop", "ImprovementCycle", "ImprovementPlan"]


class RAGImprovementLoop:
    """통합 RAG 개선 피드백 루프 (coordinator). Logic in loop_phases, loop_cycle, loop_report."""

    def __init__(
        self,
        documents: List[str],
        test_queries: List[str],
        embedding_function: Optional[Callable[[str], List[float]]] = None,
        llm_judge: Optional[Callable[[str, str, List[str]], float]] = None,
        persist_path: Optional[str] = None,
        human_weight: float = 0.5,
        auto_weight: float = 0.5,
        **kwargs: Any,
    ):
        self.documents = documents
        self.test_queries = test_queries
        self.embedding_function = embedding_function
        self.persist_path = persist_path

        # ChunkingExperimenter 초기화
        self.chunking_experimenter = ChunkingExperimenter(
            documents=documents,
            test_queries=test_queries,
            embedding_function=embedding_function,
            **kwargs,
        )

        # UnifiedEvaluator 초기화
        if UnifiedEvaluator is not None:
            self.evaluator = UnifiedEvaluator(
                auto_metrics=["faithfulness", "relevance", "context_precision"],
                human_weight=human_weight,
                auto_weight=auto_weight,
                persist_path=persist_path,
                llm_judge=llm_judge,
                embedding_function=embedding_function,
            )
        else:
            self.evaluator = None
            logger.warning("UnifiedEvaluator not available")

        # 개선 사이클 히스토리
        self._cycles: List[ImprovementCycle] = []
        self._current_best_config: Optional[Dict[str, Any]] = None
        self._baseline_score: float = 0.0

        logger.info(
            f"RAGImprovementLoop initialized: {len(documents)} docs, {len(test_queries)} queries"
        )

    def run_initial_experiments(
        self,
        configs: Optional[List[Dict[str, Any]]] = None,
        use_grid_search: bool = False,
    ) -> List[ChunkingResult]:
        """Run initial chunking experiments and set baseline."""
        results, best_config, baseline = _run_initial_experiments_impl(
            self.chunking_experimenter, configs=configs, use_grid_search=use_grid_search
        )
        self._current_best_config = best_config
        self._baseline_score = baseline
        return results

    # ==================== 2단계: 파이프라인 평가 ====================

    def evaluate_pipeline(
        self,
        query: str,
        response: str,
        contexts: List[str],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        RAG 파이프라인 평가

        Args:
            query: 질문
            response: 응답
            contexts: 검색된 컨텍스트

        Returns:
            평가 결과
        """
        if self.evaluator is None:
            logger.warning("Evaluator not available, returning empty results")
            return {"auto_scores": {}, "unified_score": 0.0}

        # 자동 평가
        auto_scores = self.evaluator.evaluate_auto(
            query=query,
            response=response,
            contexts=contexts,
            metrics=metrics,
        )

        unified_score = self.evaluator.get_unified_score(query)

        return {
            "query": query,
            "auto_scores": auto_scores,
            "unified_score": unified_score or sum(auto_scores.values()) / len(auto_scores)
            if auto_scores
            else 0.0,
        }

    def batch_evaluate(
        self,
        qa_pairs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        여러 QA 쌍 일괄 평가

        Args:
            qa_pairs: [{"query": str, "response": str, "contexts": List[str]}]

        Returns:
            일괄 평가 결과
        """
        results = []
        for qa in qa_pairs:
            result = self.evaluate_pipeline(
                query=qa["query"],
                response=qa["response"],
                contexts=qa.get("contexts", []),
            )
            results.append(result)

        avg_score = sum(r["unified_score"] for r in results) / len(results) if results else 0.0

        return {
            "total": len(results),
            "results": results,
            "avg_unified_score": avg_score,
        }

    def add_human_feedback(
        self,
        query: str,
        rating: float,
        feedback_type: str = "overall",
        comment: Optional[str] = None,
        chunk_id: Optional[str] = None,
    ) -> None:
        """Add human feedback."""
        _add_human_feedback_impl(
            self.evaluator,
            self.chunking_experimenter,
            query,
            rating,
            feedback_type=feedback_type,
            comment=comment,
            chunk_id=chunk_id,
        )

    def add_comparison_feedback(
        self,
        query: str,
        response_a: str,
        response_b: str,
        winner: str,
    ) -> None:
        """Add A/B comparison feedback."""
        _add_comparison_feedback_impl(self.evaluator, query, response_a, response_b, winner)

    def get_improvement_plan(self) -> List[ImprovementPlan]:
        """Build improvement plan from evaluator and chunking suggestions."""
        return _get_improvement_plan_impl(
            self.evaluator, self.chunking_experimenter, ImprovementPlan
        )

    def run_improvement_cycle(
        self,
        improvement_plan: Optional[ImprovementPlan] = None,
    ) -> ImprovementCycle:
        """Run a single improvement cycle."""
        return _run_improvement_cycle_step_impl(self, improvement_plan)

    def run_full_cycle(
        self,
        max_iterations: int = 3,
        target_improvement: float = 0.2,
    ) -> Dict[str, Any]:
        """Run full improvement cycle until target or max iterations."""
        return _run_full_cycle_impl(self, max_iterations, target_improvement)

    def _get_current_score(self) -> float:
        """Current combined score."""
        return _get_current_score_impl(self.chunking_experimenter, self.evaluator)

    def export_full_report(self, format: str = "markdown") -> str:
        """Export full improvement report."""

        def current_score_fn() -> float:
            return self._get_current_score()

        def chunking_report_fn() -> str:
            return self.chunking_experimenter.get_comparison_report()

        evaluator_report_fn = self.evaluator.export_report if self.evaluator else None

        return _export_full_report_impl(
            format=format,
            cycles=self._cycles,
            baseline_score=self._baseline_score,
            get_current_score_fn=current_score_fn,
            chunking_report_fn=chunking_report_fn,
            evaluator_report_fn=evaluator_report_fn,
            get_improvement_plan_fn=self.get_improvement_plan,
            current_best_config=self._current_best_config,
        )

    def detect_drift(self) -> Optional[Dict[str, Any]]:
        """성능 저하 감지"""
        if self.evaluator:
            return self.evaluator.detect_drift()
        return None

    def get_status(self) -> Dict[str, Any]:
        """현재 상태 조회"""
        return {
            "documents": len(self.documents),
            "test_queries": len(self.test_queries),
            "experiments_run": len(self.chunking_experimenter._results),
            "improvement_cycles": len(self._cycles),
            "baseline_score": self._baseline_score,
            "current_score": self._get_current_score(),
            "best_config": self._current_best_config,
        }

    def __repr__(self) -> str:
        return (
            f"RAGImprovementLoop("
            f"docs={len(self.documents)}, "
            f"queries={len(self.test_queries)}, "
            f"cycles={len(self._cycles)})"
        )
