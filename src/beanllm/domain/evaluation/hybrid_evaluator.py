"""
Hybrid Evaluator - LLM + Human 하이브리드 평가기
"""

from typing import TYPE_CHECKING, Optional

from .human_feedback import HumanFeedback, HumanFeedbackCollector
from .results import EvaluationResult

if TYPE_CHECKING:
    from .metrics import LLMJudgeMetric


class HybridEvaluator:
    """
    하이브리드 평가기 (LLM + Human)

    LLM 평가와 인간 피드백을 결합하여 더 신뢰성 높은 평가 제공
    """

    def __init__(
        self,
        llm_grader: "LLMJudgeMetric",
        feedback_collector: Optional[HumanFeedbackCollector] = None,
        human_weight: float = 0.7,
        llm_weight: float = 0.3,
    ):
        """
        Args:
            llm_grader: LLM 평가 메트릭
            feedback_collector: 피드백 수집기 (선택적)
            human_weight: 인간 피드백 가중치 (기본값: 0.7)
            llm_weight: LLM 평가 가중치 (기본값: 0.3)

        Note:
            human_weight + llm_weight = 1.0이어야 함
        """
        if abs(human_weight + llm_weight - 1.0) > 0.01:
            raise ValueError(
                f"human_weight ({human_weight}) + llm_weight ({llm_weight}) must equal 1.0"
            )

        self.llm_grader = llm_grader
        self.feedback_collector = feedback_collector or HumanFeedbackCollector()
        self.human_weight = human_weight
        self.llm_weight = llm_weight

    async def evaluate_hybrid(
        self,
        output: str,
        reference: Optional[str] = None,
        human_feedback: Optional[HumanFeedback] = None,
        criteria: Optional[str] = None,
        **kwargs,
    ) -> EvaluationResult:
        """
        하이브리드 평가 실행

        LLM 평가 + 인간 피드백 결합

        Args:
            output: 평가 대상 출력
            reference: 참조 출력 (선택적)
            human_feedback: 인간 피드백 (선택적, 없으면 LLM 평가만 사용)
            criteria: 평가 기준 (선택적)
            **kwargs: 추가 메트릭 파라미터

        Returns:
            EvaluationResult: 하이브리드 평가 결과

        Process:
            1. LLM으로 1차 평가
            2. 인간 피드백이 있으면 가중 평균
            3. 인간 피드백이 없으면 LLM 평가만 사용
        """
        # 1. LLM 평가
        llm_result = self.llm_grader.compute(
            prediction=output,
            reference=reference or "",
            criteria=criteria,
            **kwargs,
        )

        # 2. 인간 피드백이 없으면 LLM 평가만 반환
        if human_feedback is None:
            return EvaluationResult(
                metric_name="hybrid_evaluation",
                score=llm_result.score,
                metadata={
                    "llm_score": llm_result.score,
                    "human_score": None,
                    "has_human_feedback": False,
                    "llm_explanation": llm_result.explanation,
                },
                explanation=f"LLM only: {llm_result.explanation or 'No explanation'}",
            )

        # 3. 인간 피드백에서 점수 추출
        human_score = self._extract_score_from_feedback(human_feedback)

        # 4. 가중 평균 계산
        hybrid_score = (self.human_weight * human_score) + (self.llm_weight * llm_result.score)

        # 5. 결과 생성
        return EvaluationResult(
            metric_name="hybrid_evaluation",
            score=hybrid_score,
            metadata={
                "llm_score": llm_result.score,
                "human_score": human_score,
                "human_weight": self.human_weight,
                "llm_weight": self.llm_weight,
                "has_human_feedback": True,
                "feedback_type": human_feedback.feedback_type.value,
                "feedback_id": human_feedback.feedback_id,
                "llm_explanation": llm_result.explanation,
                "human_comment": human_feedback.comment,
            },
            explanation=self._generate_explanation(llm_result, human_feedback, hybrid_score),
        )

    def _extract_score_from_feedback(self, feedback: HumanFeedback) -> float:
        """
        피드백에서 점수 추출

        Args:
            feedback: 인간 피드백

        Returns:
            float: 점수 (0.0 ~ 1.0)
        """
        # 평점 피드백
        if feedback.rating is not None:
            return feedback.rating

        # 비교 평가 피드백
        if hasattr(feedback, "winner"):
            from .human_feedback import ComparisonWinner

            if feedback.winner == ComparisonWinner.A:
                return 1.0
            elif feedback.winner == ComparisonWinner.B:
                return 0.0
            else:  # TIE
                return 0.5

        # 수정 제안이나 코멘트만 있는 경우
        # 기본값으로 중간 점수 반환 (사용자가 명시적으로 평가하지 않음)
        return 0.5

    def _generate_explanation(
        self,
        llm_result: EvaluationResult,
        human_feedback: HumanFeedback,
        hybrid_score: float,
    ) -> str:
        """설명 생성"""
        parts = [
            f"Hybrid Score: {hybrid_score:.4f}",
            f"  - LLM Score: {llm_result.score:.4f} (weight: {self.llm_weight})",
            f"  - Human Score: {self._extract_score_from_feedback(human_feedback):.4f} (weight: {self.human_weight})",
        ]

        if llm_result.explanation:
            parts.append(f"  - LLM Explanation: {llm_result.explanation}")

        if human_feedback.comment:
            parts.append(f"  - Human Comment: {human_feedback.comment}")

        return "\n".join(parts)

    def evaluate_with_collection(
        self,
        output: str,
        reference: Optional[str] = None,
        criteria: Optional[str] = None,
        **kwargs,
    ) -> tuple[EvaluationResult, HumanFeedback]:
        """
        평가 실행 및 피드백 수집 준비

        LLM 평가를 실행하고, 인간 피드백을 수집할 수 있는 인터페이스 제공

        Args:
            output: 평가 대상 출력
            reference: 참조 출력 (선택적)
            criteria: 평가 기준 (선택적)
            **kwargs: 추가 메트릭 파라미터

        Returns:
            tuple[EvaluationResult, HumanFeedback]:
                - LLM 평가 결과
                - 수집할 피드백 객체 (사용자가 채워야 함)
        """

        # LLM 평가 실행
        llm_result = self.llm_grader.compute(
            prediction=output,
            reference=reference or "",
            criteria=criteria,
            **kwargs,
        )

        # 피드백 수집 준비 (평점 형태로)
        feedback = self.feedback_collector.collect_rating(
            output=output,
            rating=0.5,  # 임시 값, 사용자가 수정해야 함
            criteria=criteria,
        )

        return llm_result, feedback
