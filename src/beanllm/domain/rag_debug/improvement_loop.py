"""
RAG Improvement Loop - 통합 RAG 개선 피드백 루프

청킹 → 평가 → 피드백 → 개선의 지속적인 사이클을 관리합니다.

주요 기능:
1. ChunkingExperimenter + UnifiedEvaluator 통합
2. 자동 개선 사이클 실행
3. 개선 전후 비교 분석
4. 최적 파이프라인 추천

Example:
    ```python
    from beanllm.domain.rag_debug import RAGImprovementLoop

    # 개선 루프 생성
    loop = RAGImprovementLoop(
        documents=documents,
        test_queries=queries,
        embedding_function=embed_fn
    )

    # 1단계: 초기 실험
    initial_results = loop.run_initial_experiments([
        {"type": "recursive", "chunk_size": 500},
        {"type": "recursive", "chunk_size": 1000},
    ])

    # 2단계: RAG 파이프라인 평가
    eval_results = loop.evaluate_pipeline(
        query="What is RAG?",
        response="RAG is...",
        contexts=["Retrieved context..."]
    )

    # 3단계: Human 피드백 수집
    loop.add_human_feedback(
        query="What is RAG?",
        rating=0.7,
        comment="Good but incomplete"
    )

    # 4단계: 자동 개선 제안
    suggestions = loop.get_improvement_plan()

    # 5단계: 개선 사이클 실행
    improved_results = loop.run_improvement_cycle()

    # 리포트 생성
    report = loop.export_full_report()
    ```
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
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

try:
    from beanllm.domain.evaluation.unified_evaluator import (
        ImprovementSuggestion,
        UnifiedEvaluator,
    )
except ImportError:
    UnifiedEvaluator = None  # type: ignore
    ImprovementSuggestion = None  # type: ignore


logger = get_logger(__name__)


@dataclass
class ImprovementCycle:
    """개선 사이클 기록"""

    cycle_number: int
    timestamp: datetime
    chunking_result: Optional[ChunkingResult]
    eval_score_before: float
    eval_score_after: float
    improvement: float
    strategy_used: str
    changes_made: List[str]


@dataclass
class ImprovementPlan:
    """개선 계획"""

    priority: str  # "high", "medium", "low"
    area: str  # "chunking", "retrieval", "generation", "prompt"
    issue: str
    action: str
    expected_improvement: float
    config_changes: Dict[str, Any] = field(default_factory=dict)


class RAGImprovementLoop:
    """
    통합 RAG 개선 피드백 루프

    청킹 실험, 평가, 피드백을 통합하여 RAG 파이프라인의
    지속적인 개선을 지원합니다.

    Workflow:
        1. 초기 청킹 전략 실험 (ChunkingExperimenter)
        2. RAG 파이프라인 평가 (UnifiedEvaluator)
        3. Human/Auto 피드백 수집
        4. 개선 계획 생성
        5. 개선 사이클 실행
        6. 결과 비교 및 리포트

    Example:
        ```python
        loop = RAGImprovementLoop(
            documents=docs,
            test_queries=queries,
            embedding_function=embed_fn
        )

        # 전체 개선 사이클 실행
        result = loop.run_full_cycle(max_iterations=3)
        print(f"Final improvement: {result['total_improvement']:.2%}")
        ```
    """

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
        """
        RAGImprovementLoop 초기화

        Args:
            documents: 원본 문서 목록
            test_queries: 테스트 쿼리 목록
            embedding_function: 임베딩 함수
            llm_judge: LLM 기반 평가 함수
            persist_path: 결과 저장 경로
            human_weight: Human 피드백 가중치
            auto_weight: 자동 평가 가중치
        """
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

    # ==================== 1단계: 초기 실험 ====================

    def run_initial_experiments(
        self,
        configs: Optional[List[Dict[str, Any]]] = None,
        use_grid_search: bool = False,
    ) -> List[ChunkingResult]:
        """
        초기 청킹 전략 실험 실행

        Args:
            configs: 테스트할 설정 목록 (None이면 기본 설정)
            use_grid_search: 그리드 서치 사용 여부

        Returns:
            실험 결과 목록
        """
        if configs is None:
            # 기본 설정
            configs = [
                {
                    "type": "recursive",
                    "chunk_size": 500,
                    "chunk_overlap": 50,
                    "name": "recursive_500",
                },
                {
                    "type": "recursive",
                    "chunk_size": 1000,
                    "chunk_overlap": 100,
                    "name": "recursive_1000",
                },
                {
                    "type": "recursive",
                    "chunk_size": 1500,
                    "chunk_overlap": 150,
                    "name": "recursive_1500",
                },
            ]

        if use_grid_search:
            results = self.chunking_experimenter.grid_search(
                splitter_type="recursive",
                chunk_sizes=[256, 512, 1000],
                chunk_overlaps=[0, 50, 100],
            )
        else:
            results = self.chunking_experimenter.compare_strategies(configs)

        # 베이스라인 설정
        best = self.chunking_experimenter.find_best_strategy()
        if best:
            self._current_best_config = best["config"]
            self._baseline_score = best["score"]

        logger.info(
            f"Initial experiments complete: {len(results)} strategies tested, "
            f"baseline score: {self._baseline_score:.4f}"
        )

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

    # ==================== 3단계: 피드백 수집 ====================

    def add_human_feedback(
        self,
        query: str,
        rating: float,
        feedback_type: str = "overall",
        comment: Optional[str] = None,
        chunk_id: Optional[str] = None,
    ):
        """
        Human 피드백 추가

        Args:
            query: 쿼리
            rating: 평점 (0.0-1.0)
            feedback_type: 피드백 유형
            comment: 코멘트
            chunk_id: 청크 ID (청킹 피드백용)
        """
        # UnifiedEvaluator에 피드백
        if self.evaluator:
            self.evaluator.collect_human_feedback(
                query=query,
                rating=rating,
                feedback_type=feedback_type,
                comment=comment,
            )

        # ChunkingExperimenter에 피드백 (청크 ID가 있으면)
        if chunk_id:
            self.chunking_experimenter.add_feedback(
                query=query,
                chunk_id=chunk_id,
                rating=rating,
                feedback_type=feedback_type,
                comment=comment,
            )

        logger.info(f"Human feedback added: query='{query[:30]}...', rating={rating:.2f}")

    def add_comparison_feedback(
        self,
        query: str,
        response_a: str,
        response_b: str,
        winner: str,
    ):
        """
        A/B 비교 피드백 추가

        Args:
            query: 쿼리
            response_a: 응답 A
            response_b: 응답 B
            winner: 선택 ("A", "B", "TIE")
        """
        if self.evaluator:
            self.evaluator.collect_comparison_feedback(
                query=query,
                response_a=response_a,
                response_b=response_b,
                winner=winner,
            )

    # ==================== 4단계: 개선 계획 ====================

    def get_improvement_plan(self) -> List[ImprovementPlan]:
        """
        피드백 기반 개선 계획 생성

        Returns:
            개선 계획 목록 (우선순위순)
        """
        plans: List[ImprovementPlan] = []

        # UnifiedEvaluator에서 제안 가져오기
        if self.evaluator:
            suggestions = self.evaluator.get_improvement_suggestions()
            for s in suggestions:
                plans.append(
                    ImprovementPlan(
                        priority=s.priority,
                        area=s.category,
                        issue=s.issue,
                        action=s.suggestion,
                        expected_improvement=s.expected_improvement,
                    )
                )

        # ChunkingExperimenter에서 제안 가져오기
        chunking_suggestions = self.chunking_experimenter.improve_from_feedback()
        for suggestion in chunking_suggestions.get("suggestions", []):
            plans.append(
                ImprovementPlan(
                    priority="medium",
                    area="chunking",
                    issue="청킹 개선 필요",
                    action=suggestion,
                    expected_improvement=0.15,
                    config_changes=chunking_suggestions.get("recommended_configs", [{}])[0]
                    if chunking_suggestions.get("recommended_configs")
                    else {},
                )
            )

        # 우선순위 정렬
        priority_order = {"high": 0, "medium": 1, "low": 2}
        plans.sort(key=lambda p: priority_order.get(p.priority, 2))

        return plans

    # ==================== 5단계: 개선 사이클 ====================

    def run_improvement_cycle(
        self,
        improvement_plan: Optional[ImprovementPlan] = None,
    ) -> ImprovementCycle:
        """
        단일 개선 사이클 실행

        Args:
            improvement_plan: 적용할 개선 계획 (None이면 자동 선택)

        Returns:
            개선 사이클 결과
        """
        cycle_number = len(self._cycles) + 1

        # 개선 전 점수
        score_before = self._get_current_score()

        # 개선 계획 선택
        if improvement_plan is None:
            plans = self.get_improvement_plan()
            if not plans:
                logger.info("No improvement plans available")
                return ImprovementCycle(
                    cycle_number=cycle_number,
                    timestamp=datetime.now(timezone.utc),
                    chunking_result=None,
                    eval_score_before=score_before,
                    eval_score_after=score_before,
                    improvement=0.0,
                    strategy_used="none",
                    changes_made=[],
                )
            improvement_plan = plans[0]  # 최우선 계획

        # 개선 적용
        changes_made = []
        chunking_result = None

        if improvement_plan.area == "chunking" and improvement_plan.config_changes:
            # 새로운 청킹 설정으로 실험
            new_config = improvement_plan.config_changes
            chunking_result = self.chunking_experimenter.run_experiment(
                config=new_config,
                strategy_name=f"improved_cycle_{cycle_number}",
            )
            changes_made.append(f"Chunking config: {new_config}")
            self._current_best_config = new_config

        # 개선 후 점수
        score_after = self._get_current_score()

        # 사이클 기록
        cycle = ImprovementCycle(
            cycle_number=cycle_number,
            timestamp=datetime.now(timezone.utc),
            chunking_result=chunking_result,
            eval_score_before=score_before,
            eval_score_after=score_after,
            improvement=score_after - score_before,
            strategy_used=improvement_plan.action[:50],
            changes_made=changes_made,
        )

        self._cycles.append(cycle)

        logger.info(
            f"Improvement cycle {cycle_number} complete: "
            f"{score_before:.4f} → {score_after:.4f} "
            f"(+{cycle.improvement:.4f})"
        )

        return cycle

    def run_full_cycle(
        self,
        max_iterations: int = 3,
        target_improvement: float = 0.2,
    ) -> Dict[str, Any]:
        """
        전체 개선 사이클 실행

        Args:
            max_iterations: 최대 반복 횟수
            target_improvement: 목표 개선율

        Returns:
            최종 결과
        """
        initial_score = self._get_current_score()
        total_improvement = 0.0

        for i in range(max_iterations):
            cycle = self.run_improvement_cycle()
            total_improvement += cycle.improvement

            # 목표 달성 확인
            if total_improvement >= target_improvement:
                logger.info(f"Target improvement reached: {total_improvement:.4f}")
                break

            # 더 이상 개선이 없으면 종료
            if cycle.improvement < 0.01:
                logger.info("No significant improvement, stopping")
                break

        final_score = self._get_current_score()

        return {
            "initial_score": initial_score,
            "final_score": final_score,
            "total_improvement": total_improvement,
            "cycles_run": len(self._cycles),
            "best_config": self._current_best_config,
            "cycles": self._cycles,
        }

    def _get_current_score(self) -> float:
        """현재 점수 가져오기"""
        # ChunkingExperimenter의 최고 점수
        best = self.chunking_experimenter.find_best_strategy()
        chunking_score = best["score"] if best else 0.0

        # UnifiedEvaluator의 평균 점수
        eval_score = 0.0
        if self.evaluator:
            summary = self.evaluator.get_evaluation_summary()
            eval_score = summary.get("unified_score", {}).get("avg", 0.0)

        # 두 점수 조합 (둘 다 있으면 평균, 하나만 있으면 그것 사용)
        if chunking_score > 0 and eval_score > 0:
            return (chunking_score + eval_score) / 2
        elif chunking_score > 0:
            return chunking_score
        elif eval_score > 0:
            return eval_score
        return 0.0

    # ==================== 6단계: 리포트 ====================

    def export_full_report(self, format: str = "markdown") -> str:
        """
        전체 개선 리포트 내보내기

        Args:
            format: 출력 형식 ("markdown", "json")

        Returns:
            리포트 문자열
        """
        if format == "markdown":
            lines = [
                "# RAG Improvement Report",
                "",
                f"**Generated**: {datetime.now(timezone.utc).isoformat()}",
                "",
                "## Summary",
                f"- Total improvement cycles: {len(self._cycles)}",
                f"- Baseline score: {self._baseline_score:.4f}",
                f"- Current score: {self._get_current_score():.4f}",
                "",
            ]

            # 청킹 실험 결과
            lines.append("## Chunking Experiments")
            lines.append(self.chunking_experimenter.get_comparison_report())
            lines.append("")

            # 평가 결과
            if self.evaluator:
                lines.append("## Evaluation Results")
                lines.append(self.evaluator.export_report())
                lines.append("")

            # 개선 사이클 히스토리
            if self._cycles:
                lines.append("## Improvement History")
                lines.append("")
                lines.append("| Cycle | Before | After | Improvement | Strategy |")
                lines.append("|-------|--------|-------|-------------|----------|")
                for c in self._cycles:
                    lines.append(
                        f"| {c.cycle_number} | {c.eval_score_before:.4f} | "
                        f"{c.eval_score_after:.4f} | {c.improvement:+.4f} | "
                        f"{c.strategy_used[:30]}... |"
                    )

            # 최종 추천
            lines.append("")
            lines.append("## Final Recommendations")
            if self._current_best_config:
                lines.append(f"**Best Chunking Config**: {self._current_best_config}")

            plans = self.get_improvement_plan()
            if plans:
                lines.append("")
                lines.append("**Remaining Improvements**:")
                for p in plans[:3]:
                    lines.append(f"- [{p.priority}] {p.area}: {p.action[:50]}...")

            return "\n".join(lines)

        # JSON 형식
        import json

        return json.dumps(
            {
                "baseline_score": self._baseline_score,
                "current_score": self._get_current_score(),
                "cycles": len(self._cycles),
                "best_config": self._current_best_config,
            },
            indent=2,
        )

    # ==================== 유틸리티 ====================

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
