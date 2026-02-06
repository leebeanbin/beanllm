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

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from beanllm.utils.logging import get_logger
except ImportError:

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


@dataclass
class EvalRecord:
    """평가 기록"""

    record_id: str
    query: str
    response: str
    contexts: List[str]

    # 자동 평가 결과
    auto_scores: Dict[str, float] = field(default_factory=dict)
    auto_avg_score: float = 0.0

    # Human 피드백
    human_ratings: List[float] = field(default_factory=list)
    human_avg_rating: float = 0.0
    human_feedback_count: int = 0
    human_comments: List[str] = field(default_factory=list)

    # 통합 점수
    unified_score: float = 0.0

    # 메타데이터
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImprovementSuggestion:
    """개선 제안"""

    category: str  # "chunking", "retrieval", "generation", "prompt"
    priority: str  # "high", "medium", "low"
    issue: str
    suggestion: str
    affected_queries: List[str] = field(default_factory=list)
    expected_improvement: float = 0.0


class UnifiedEvaluator:
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
            f"Auto evaluation: query='{query[:30]}...', " f"avg_score={record.auto_avg_score:.4f}"
        )

        return scores

    def _compute_metric(self, metric: str, query: str, response: str, contexts: List[str]) -> float:
        """개별 메트릭 계산"""
        context_text = " ".join(contexts)

        if metric == "faithfulness":
            return self._compute_faithfulness(response, context_text)

        elif metric == "relevance":
            return self._compute_relevance(query, response)

        elif metric == "context_precision":
            return self._compute_context_precision(query, contexts)

        elif metric == "context_recall":
            return self._compute_context_recall(query, response, contexts)

        elif metric == "coherence":
            return self._compute_coherence(response)

        elif metric == "completeness":
            return self._compute_completeness(query, response)

        return 0.5  # 기본값

    def _compute_faithfulness(self, response: str, context: str) -> float:
        """Faithfulness 계산 (환각 감지)"""
        if self.llm_judge:
            prompt = f"""
            Context: {context[:2000]}
            Response: {response}

            Is the response faithful to the context? (0.0-1.0)
            Only return a number.
            """
            try:
                return float(self.llm_judge(prompt, "", []))
            except (ValueError, TypeError):
                pass

        # 간단한 단어 겹침 기반
        response_words = set(response.lower().split())
        context_words = set(context.lower().split())
        overlap = len(response_words & context_words)
        return min(1.0, overlap / len(response_words)) if response_words else 0.0

    def _compute_relevance(self, query: str, response: str) -> float:
        """Relevance 계산"""
        if self.embedding_function:
            import math

            query_emb = self.embedding_function(query)
            response_emb = self.embedding_function(response)

            dot = sum(a * b for a, b in zip(query_emb, response_emb))
            mag1 = math.sqrt(sum(a * a for a in query_emb))
            mag2 = math.sqrt(sum(b * b for b in response_emb))

            if mag1 > 0 and mag2 > 0:
                return (dot / (mag1 * mag2) + 1) / 2  # 0-1 범위로 정규화

        # 단어 겹침 기반
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words & response_words)
        return overlap / len(query_words) if query_words else 0.0

    def _compute_context_precision(self, query: str, contexts: List[str]) -> float:
        """Context Precision 계산"""
        if not contexts:
            return 0.0

        query_words = set(query.lower().split())
        relevant_count = 0

        for ctx in contexts:
            ctx_words = set(ctx.lower().split())
            overlap = len(query_words & ctx_words)
            if overlap / len(query_words) > 0.2:  # 20% 이상 겹치면 관련
                relevant_count += 1

        return relevant_count / len(contexts)

    def _compute_context_recall(self, query: str, response: str, contexts: List[str]) -> float:
        """Context Recall 계산"""
        if not contexts:
            return 0.0

        # 응답에서 사용된 컨텍스트 비율
        response_words = set(response.lower().split())
        used_contexts = 0

        for ctx in contexts:
            ctx_words = set(ctx.lower().split())
            overlap = len(response_words & ctx_words)
            if overlap > 5:  # 5개 이상 단어 겹치면 사용된 것으로 간주
                used_contexts += 1

        return used_contexts / len(contexts)

    def _compute_coherence(self, response: str) -> float:
        """Coherence 계산 (문장 연결성)"""
        sentences = response.split(".")
        if len(sentences) < 2:
            return 1.0  # 단일 문장은 일관성 있음

        # 간단한 휴리스틱: 문장 길이 일관성
        lengths = [len(s.split()) for s in sentences if s.strip()]
        if not lengths:
            return 0.5

        avg_len = sum(lengths) / len(lengths)
        variance = sum((length - avg_len) ** 2 for length in lengths) / len(lengths)

        # 분산이 작을수록 일관성 높음
        return max(0.0, 1.0 - variance / 100)

    def _compute_completeness(self, query: str, response: str) -> float:
        """Completeness 계산"""
        # 질문 유형 감지 및 응답 완전성 평가
        query_lower = query.lower()

        # WH-질문 감지
        wh_words = ["what", "why", "how", "when", "where", "who", "which"]
        is_wh_question = any(w in query_lower for w in wh_words)

        # 응답 길이 기반 완전성
        response_words = len(response.split())

        if is_wh_question:
            # WH-질문은 최소 10단어 이상 답변 기대
            return min(1.0, response_words / 20)
        else:
            # 일반 질문은 최소 5단어
            return min(1.0, response_words / 10)

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

        # 피드백 추가
        record.human_ratings.append(rating)
        record.human_feedback_count += 1
        record.human_avg_rating = sum(record.human_ratings) / len(record.human_ratings)

        if comment:
            record.human_comments.append(comment)

        # 통합 점수 업데이트
        self._update_unified_score(record)

        logger.info(
            f"Human feedback collected: query='{query[:30]}...', "
            f"rating={rating:.2f}, type={feedback_type}"
        )

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
        # Winner를 rating으로 변환
        if winner == "A":
            self.collect_human_feedback(query, 1.0, "comparison", "Winner: A")
        elif winner == "B":
            self.collect_human_feedback(query, 0.0, "comparison", "Winner: B")
        else:
            self.collect_human_feedback(query, 0.5, "comparison", "TIE")

    # ==================== 통합 점수 ====================

    def _update_unified_score(self, record: EvalRecord):
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
        suggestions: List[ImprovementSuggestion] = []

        if not self._records:
            return suggestions

        records = list(self._records.values())

        # 1. 낮은 Faithfulness 감지
        low_faithfulness = [r for r in records if r.auto_scores.get("faithfulness", 1.0) < 0.5]
        if low_faithfulness:
            suggestions.append(
                ImprovementSuggestion(
                    category="retrieval",
                    priority="high",
                    issue=f"{len(low_faithfulness)}개 쿼리에서 환각 감지됨",
                    suggestion=(
                        "검색된 컨텍스트가 불충분합니다. "
                        "청크 크기를 늘리거나, top_k를 증가시키거나, "
                        "Reranker를 추가해보세요."
                    ),
                    affected_queries=[r.query for r in low_faithfulness[:5]],
                    expected_improvement=0.2,
                )
            )

        # 2. 낮은 Relevance 감지
        low_relevance = [r for r in records if r.auto_scores.get("relevance", 1.0) < 0.5]
        if low_relevance:
            suggestions.append(
                ImprovementSuggestion(
                    category="generation",
                    priority="high",
                    issue=f"{len(low_relevance)}개 쿼리에서 관련성 부족",
                    suggestion=(
                        "프롬프트를 개선하여 질문에 더 집중하도록 하세요. "
                        "또는 쿼리 확장(HyDE, MultiQuery)을 시도해보세요."
                    ),
                    affected_queries=[r.query for r in low_relevance[:5]],
                    expected_improvement=0.15,
                )
            )

        # 3. 낮은 Human 피드백
        low_human = [r for r in records if r.human_avg_rating < 0.4 and r.human_feedback_count > 0]
        if low_human:
            # 코멘트 분석
            all_comments = []
            for r in low_human:
                all_comments.extend(r.human_comments)

            suggestions.append(
                ImprovementSuggestion(
                    category="overall",
                    priority="high",
                    issue=f"{len(low_human)}개 쿼리에서 사용자 불만족",
                    suggestion=(
                        "사용자 피드백이 낮습니다. "
                        f"주요 코멘트: {all_comments[:3] if all_comments else '없음'}. "
                        "청킹 전략 변경과 프롬프트 개선을 고려하세요."
                    ),
                    affected_queries=[r.query for r in low_human[:5]],
                    expected_improvement=0.25,
                )
            )

        # 4. Context Precision 낮음
        low_ctx_precision = [
            r for r in records if r.auto_scores.get("context_precision", 1.0) < 0.5
        ]
        if low_ctx_precision:
            suggestions.append(
                ImprovementSuggestion(
                    category="chunking",
                    priority="medium",
                    issue=f"{len(low_ctx_precision)}개 쿼리에서 컨텍스트 정확도 낮음",
                    suggestion=(
                        "검색된 청크가 질문과 관련이 적습니다. "
                        "시맨틱 청킹을 사용하거나, 청크 크기를 줄여보세요. "
                        "하이브리드 검색(BM25 + Dense)도 효과적입니다."
                    ),
                    affected_queries=[r.query for r in low_ctx_precision[:5]],
                    expected_improvement=0.2,
                )
            )

        # 우선순위순 정렬
        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda s: priority_order.get(s.priority, 2))

        return suggestions

    # ==================== Drift 감지 ====================

    def detect_drift(self) -> Optional[Dict[str, Any]]:
        """성능 저하 감지"""
        if len(self._records) < 10:
            return None

        records = sorted(self._records.values(), key=lambda r: r.timestamp)

        # 최근 vs 이전 비교
        midpoint = len(records) // 2
        old_records = records[:midpoint]
        new_records = records[midpoint:]

        old_avg = sum(r.unified_score for r in old_records) / len(old_records)
        new_avg = sum(r.unified_score for r in new_records) / len(new_records)

        drift = old_avg - new_avg

        if drift > self.drift_threshold:
            return {
                "detected": True,
                "old_score": old_avg,
                "new_score": new_avg,
                "drift_magnitude": drift,
                "severity": "high" if drift > 0.3 else "medium",
                "message": (
                    f"성능 저하 감지: {old_avg:.2f} → {new_avg:.2f} " f"(하락폭: {drift:.2f})"
                ),
            }

        return {"detected": False, "old_score": old_avg, "new_score": new_avg}

    # ==================== 저장/로드 ====================

    def _save_history(self):
        """평가 히스토리 저장"""
        if not self.persist_path:
            return

        history_file = self.persist_path / "eval_history.json"

        data = {
            "records": [
                {
                    "record_id": r.record_id,
                    "query": r.query,
                    "response": r.response[:500],  # 응답 일부만 저장
                    "auto_scores": r.auto_scores,
                    "auto_avg_score": r.auto_avg_score,
                    "human_ratings": r.human_ratings,
                    "human_avg_rating": r.human_avg_rating,
                    "human_feedback_count": r.human_feedback_count,
                    "human_comments": r.human_comments,
                    "unified_score": r.unified_score,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in self._records.values()
            ],
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.debug(f"Saved {len(self._records)} records to {history_file}")

    def _load_history(self):
        """평가 히스토리 로드"""
        if not self.persist_path:
            return

        history_file = self.persist_path / "eval_history.json"

        if not history_file.exists():
            return

        try:
            with open(history_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for r in data.get("records", []):
                record = EvalRecord(
                    record_id=r["record_id"],
                    query=r["query"],
                    response=r.get("response", ""),
                    contexts=[],
                    auto_scores=r.get("auto_scores", {}),
                    auto_avg_score=r.get("auto_avg_score", 0.0),
                    human_ratings=r.get("human_ratings", []),
                    human_avg_rating=r.get("human_avg_rating", 0.0),
                    human_feedback_count=r.get("human_feedback_count", 0),
                    human_comments=r.get("human_comments", []),
                    unified_score=r.get("unified_score", 0.0),
                    timestamp=datetime.fromisoformat(
                        r.get("timestamp", datetime.now(timezone.utc).isoformat())
                    ),
                )
                self._records[record.record_id] = record
                self._query_to_record[record.query] = record.record_id

            logger.info(f"Loaded {len(self._records)} records from {history_file}")

        except Exception as e:
            logger.error(f"Failed to load history: {e}")

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
