"""
Unified Evaluator - Human-in-the-Loop + Automatic Evaluation 통합 시스템

자동 평가와 Human 피드백을 통합하여 RAG 시스템의 지속적인 개선을 지원합니다.

주요 기능:
1. 자동 평가 (RAGAS 스타일 메트릭)
2. Human-in-the-Loop 피드백 수집
3. 피드백 기반 자동 개선 제안
4. 평가 결과 저장 및 분석
5. Drift 감지 및 알림

Example:
    ```python
    from beanllm.domain.evaluation import UnifiedEvaluator

    # 통합 평가기 생성
    evaluator = UnifiedEvaluator(
        auto_metrics=["faithfulness", "relevance", "context_precision"],
        human_weight=0.6,  # 60% human, 40% auto
        persist_path="./eval_history"
    )

    # 자동 평가
    auto_result = evaluator.evaluate_auto(
        query="What is RAG?",
        response="RAG is Retrieval-Augmented Generation...",
        contexts=["RAG combines retrieval with generation..."]
    )

    # Human 피드백 수집
    evaluator.collect_human_feedback(
        query="What is RAG?",
        response="RAG is...",
        rating=0.8,
        feedback_type="relevance"
    )

    # 통합 점수 계산
    unified_score = evaluator.get_unified_score(query="What is RAG?")

    # 개선 제안
    suggestions = evaluator.get_improvement_suggestions()
    ```
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from beanllm.domain.evaluation.unified import (
    DriftDetector,
    HumanFeedbackManager,
    ImprovementAnalyzer,
)
from beanllm.domain.evaluation.unified_auto_metrics import AutoMetricsMixin
from beanllm.domain.evaluation.unified_models import EvalRecord, ImprovementSuggestion
from beanllm.domain.evaluation.unified_persistence import PersistenceMixin

try:
    from beanllm.utils.logging import get_logger
except ImportError:

    def get_logger(name: str) -> logging.Logger:  # type: ignore[misc]
        return logging.getLogger(name)


logger = get_logger(__name__)

# Re-export for backward compatibility
__all__ = ["EvalRecord", "ImprovementSuggestion", "UnifiedEvaluator"]


class UnifiedEvaluator(AutoMetricsMixin, PersistenceMixin):
    """
    Human-in-the-Loop + Automatic Evaluation 통합 시스템

    자동 평가와 Human 피드백을 결합하여 RAG 품질을 종합적으로 평가하고,
    개선 방향을 제안합니다.

    Features:
    - 자동 평가: Faithfulness, Relevance, Context Precision 등
    - Human 피드백: Rating, Comparison, Correction
    - 통합 점수: 가중 평균 기반
    - 개선 제안: 낮은 점수 패턴 분석
    - 히스토리 저장: JSON 파일 기반 영속화

    Example:
        ```python
        evaluator = UnifiedEvaluator(
            auto_metrics=["faithfulness", "relevance"],
            human_weight=0.6,
            persist_path="./evaluations"
        )

        # 평가 실행
        result = evaluator.evaluate(
            query="What is Python?",
            response="Python is a programming language.",
            contexts=["Python is versatile..."]
        )

        # 피드백 추가
        evaluator.collect_human_feedback(
            query="What is Python?",
            rating=0.9,
            comment="Good answer"
        )

        # 개선 제안 받기
        suggestions = evaluator.get_improvement_suggestions()
        ```
    """

    # 지원하는 자동 평가 메트릭
    SUPPORTED_METRICS = {
        "faithfulness": "응답이 컨텍스트에 기반하는지 (환각 감지)",
        "relevance": "응답이 질문과 관련있는지",
        "context_precision": "검색된 컨텍스트의 정확도",
        "context_recall": "필요한 정보가 검색되었는지",
        "coherence": "응답의 논리적 일관성",
        "completeness": "응답의 완전성",
    }

    def __init__(
        self,
        auto_metrics: Optional[List[str]] = None,
        human_weight: float = 0.5,
        auto_weight: float = 0.5,
        persist_path: Optional[str] = None,
        llm_judge: Optional[Callable[[str, str, List[str]], float]] = None,
        embedding_function: Optional[Callable[[str], List[float]]] = None,
        drift_threshold: float = 0.2,
        **kwargs: Any,
    ):
        """
        UnifiedEvaluator 초기화

        Args:
            auto_metrics: 사용할 자동 평가 메트릭 목록
            human_weight: Human 피드백 가중치 (0.0-1.0)
            auto_weight: 자동 평가 가중치 (0.0-1.0)
            persist_path: 평가 기록 저장 경로
            llm_judge: LLM 기반 평가 함수 (선택)
            embedding_function: 임베딩 함수 (선택)
            drift_threshold: Drift 감지 임계값
            **kwargs: 추가 옵션
        """
        self.auto_metrics = auto_metrics or ["faithfulness", "relevance"]
        self.human_weight = human_weight
        self.auto_weight = auto_weight
        self.persist_path = Path(persist_path) if persist_path else None
        self.llm_judge = llm_judge
        self.embedding_function = embedding_function
        self.drift_threshold = drift_threshold
        self.kwargs = kwargs

        # 가중치 정규화
        total_weight = self.human_weight + self.auto_weight
        if total_weight > 0:
            self.human_weight /= total_weight
            self.auto_weight /= total_weight

        # 평가 기록 저장소
        self._records: Dict[str, EvalRecord] = {}
        self._query_to_record: Dict[str, str] = {}  # query -> record_id

        # Compose modules
        self._human_feedback_manager = HumanFeedbackManager(self.human_weight)
        self._improvement_analyzer = ImprovementAnalyzer()
        self._drift_detector = DriftDetector(drift_threshold)

        # 히스토리 로드
        if self.persist_path:
            self.persist_path.mkdir(parents=True, exist_ok=True)
            self._load_history()

        logger.info(
            f"UnifiedEvaluator initialized: "
            f"metrics={self.auto_metrics}, "
            f"human_weight={self.human_weight:.2f}, "
            f"auto_weight={self.auto_weight:.2f}"
        )

    def _generate_record_id(self, query: str) -> str:
        """쿼리 기반 레코드 ID 생성"""
        import hashlib

        return hashlib.md5(query.encode()).hexdigest()[:12]

    def _get_or_create_record(
        self, query: str, response: str = "", contexts: Optional[List[str]] = None
    ) -> EvalRecord:
        """레코드 가져오기 또는 생성"""
        record_id = self._generate_record_id(query)

        if record_id not in self._records:
            self._records[record_id] = EvalRecord(
                record_id=record_id,
                query=query,
                response=response,
                contexts=contexts or [],
            )
            self._query_to_record[query] = record_id

        return self._records[record_id]

    # ==================== 자동 평가 ====================

    def evaluate_auto(
        self,
        query: str,
        response: str,
        contexts: List[str],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        자동 평가 실행

        Args:
            query: 질문
            response: 응답
            contexts: 검색된 컨텍스트 목록
            metrics: 평가할 메트릭 (None이면 기본 메트릭)

        Returns:
            메트릭별 점수 딕셔너리
        """
        metrics = metrics or self.auto_metrics
        scores: Dict[str, float] = {}

        for metric in metrics:
            if metric not in self.SUPPORTED_METRICS:
                logger.warning(f"Unknown metric: {metric}")
                continue

            score = self._compute_metric(metric, query, response, contexts)
            scores[metric] = score

        # 레코드 업데이트
        record = self._get_or_create_record(query, response, contexts)
        record.auto_scores = scores
        record.auto_avg_score = sum(scores.values()) / len(scores) if scores else 0.0
        record.response = response
        record.contexts = contexts
        self._update_unified_score(record)

        logger.info(
            f"Auto evaluation: query='{query[:30]}...', avg_score={record.auto_avg_score:.4f}"
        )

        return scores

    # ==================== Human 피드백 ====================

    def collect_human_feedback(
        self,
        query: str,
        rating: float,
        feedback_type: str = "overall",
        comment: Optional[str] = None,
        response: Optional[str] = None,
    ):
        """
        Human 피드백 수집

        Args:
            query: 질문
            rating: 평점 (0.0-1.0)
            feedback_type: 피드백 유형 ("overall", "relevance", "faithfulness", ...)
            comment: 추가 코멘트
            response: 응답 (레코드 업데이트용)
        """
        record = self._get_or_create_record(query, response or "")
        self._human_feedback_manager.collect_feedback(record, rating, feedback_type, comment)

        # 통합 점수 업데이트
        self._update_unified_score(record)

        # 저장
        self._save_history()

    def collect_comparison_feedback(
        self,
        query: str,
        response_a: str,
        response_b: str,
        winner: str,  # "A", "B", "TIE"
    ):
        """
        비교 피드백 수집 (A/B)

        Args:
            query: 질문
            response_a: 응답 A
            response_b: 응답 B
            winner: 선택된 응답 ("A", "B", "TIE")
        """
        record = self._get_or_create_record(query)
        self._human_feedback_manager.collect_comparison_feedback(
            record, response_a, response_b, winner
        )

        # 통합 점수 업데이트
        self._update_unified_score(record)

        # 저장
        self._save_history()

    # ==================== 통합 점수 ====================

    def _update_unified_score(self, record: EvalRecord) -> None:
        """통합 점수 업데이트"""
        auto_score = record.auto_avg_score
        human_score = record.human_avg_rating

        # Human 피드백이 있으면 가중 평균
        if record.human_feedback_count > 0 and auto_score > 0:
            record.unified_score = self.auto_weight * auto_score + self.human_weight * human_score
        elif record.human_feedback_count > 0:
            record.unified_score = human_score
        elif auto_score > 0:
            record.unified_score = auto_score
        else:
            record.unified_score = 0.0

    def get_unified_score(self, query: str) -> Optional[float]:
        """통합 점수 조회"""
        record_id = self._generate_record_id(query)
        if record_id in self._records:
            return self._records[record_id].unified_score
        return None

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """전체 평가 요약"""
        if not self._records:
            return {"total": 0, "message": "No evaluations yet"}

        records = list(self._records.values())
        unified_scores = [r.unified_score for r in records if r.unified_score > 0]
        auto_scores = [r.auto_avg_score for r in records if r.auto_avg_score > 0]
        human_scores = [r.human_avg_rating for r in records if r.human_feedback_count > 0]

        return {
            "total_records": len(records),
            "total_human_feedbacks": sum(r.human_feedback_count for r in records),
            "unified_score": {
                "avg": sum(unified_scores) / len(unified_scores) if unified_scores else 0,
                "min": min(unified_scores) if unified_scores else 0,
                "max": max(unified_scores) if unified_scores else 0,
            },
            "auto_score": {
                "avg": sum(auto_scores) / len(auto_scores) if auto_scores else 0,
            },
            "human_score": {
                "avg": sum(human_scores) / len(human_scores) if human_scores else 0,
            },
            "low_scoring_queries": [r.query for r in records if r.unified_score < 0.5][:10],
        }

    # ==================== 개선 제안 ====================

    def get_improvement_suggestions(self) -> List[ImprovementSuggestion]:
        """피드백 기반 개선 제안 생성"""
        records = list(self._records.values())
        return self._improvement_analyzer.get_improvement_suggestions(records)

    # ==================== Drift 감지 ====================

    def detect_drift(self) -> Optional[Dict[str, Any]]:
        """성능 저하 감지"""
        records = list(self._records.values())
        return self._drift_detector.detect_drift(records)

    # ==================== 유틸리티 ====================

    def get_record(self, query: str) -> Optional[EvalRecord]:
        """쿼리에 대한 평가 레코드 조회"""
        record_id = self._generate_record_id(query)
        return self._records.get(record_id)

    def get_all_records(self) -> List[EvalRecord]:
        """모든 평가 레코드 조회"""
        return list(self._records.values())

    def export_report(self, format: str = "markdown") -> str:
        """평가 리포트 내보내기"""
        summary = self.get_evaluation_summary()
        suggestions = self.get_improvement_suggestions()
        drift = self.detect_drift()

        if format == "markdown":
            lines = [
                "# RAG Evaluation Report",
                "",
                f"**Total Records:** {summary['total_records']}",
                f"**Total Human Feedbacks:** {summary['total_human_feedbacks']}",
                "",
                "## Scores",
                f"- Unified Avg: {summary['unified_score']['avg']:.4f}",
                f"- Auto Avg: {summary['auto_score']['avg']:.4f}",
                f"- Human Avg: {summary['human_score']['avg']:.4f}",
                "",
            ]

            if drift and drift.get("detected"):
                lines.extend(
                    [
                        "## ⚠️ Drift Detected",
                        f"- {drift['message']}",
                        "",
                    ]
                )

            if suggestions:
                lines.extend(
                    [
                        "## Improvement Suggestions",
                        "",
                    ]
                )
                for s in suggestions:
                    lines.append(f"### [{s.priority.upper()}] {s.category}")
                    lines.append(f"- **Issue:** {s.issue}")
                    lines.append(f"- **Suggestion:** {s.suggestion}")
                    lines.append("")

            return "\n".join(lines)

        return str(summary)

    def __repr__(self) -> str:
        return (
            f"UnifiedEvaluator("
            f"records={len(self._records)}, "
            f"metrics={self.auto_metrics}, "
            f"human_weight={self.human_weight:.2f})"
        )
