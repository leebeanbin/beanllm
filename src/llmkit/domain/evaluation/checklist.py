"""
CheckEval - 체크리스트 기반 평가
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base_metric import BaseMetric
from .enums import MetricType
from .results import EvaluationResult


@dataclass
class ChecklistItem:
    """체크리스트 항목"""

    question: str
    description: Optional[str] = None
    weight: float = 1.0  # 가중치
    required: bool = False  # 필수 항목인지 여부


@dataclass
class Checklist:
    """체크리스트"""

    name: str
    description: str
    items: List[ChecklistItem]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """검증"""
        if not self.items:
            raise ValueError("Checklist must have at least one item")
        # 가중치 정규화
        total_weight = sum(item.weight for item in self.items)
        if total_weight > 0:
            for item in self.items:
                item.weight = item.weight / total_weight


class ChecklistGrader(BaseMetric):
    """
    체크리스트 기반 평가기

    Boolean 질문 기반 평가로 명확하고 신뢰성 높은 평가 제공
    """

    def __init__(
        self,
        checklist: Checklist,
        client=None,
        use_llm: bool = True,
    ):
        """
        Args:
            checklist: 평가 체크리스트
            client: LLM 클라이언트 (use_llm=True일 때 필요)
            use_llm: LLM을 사용하여 평가할지 여부 (False면 수동 평가만)
        """
        super().__init__(f"checklist_{checklist.name}", MetricType.QUALITY)
        self.checklist = checklist
        self.client = client
        self.use_llm = use_llm

    def _get_client(self):
        """클라이언트 lazy loading"""
        if self.client is None:
            try:
                from ...facade.client_facade import create_client

                self.client = create_client()
            except Exception:
                raise RuntimeError("LLM client not available. Please provide a client.")
        return self.client

    def _create_checklist_prompt(self, prediction: str, reference: Optional[str] = None) -> str:
        """체크리스트 평가 프롬프트 생성"""
        prompt_parts = [
            "Evaluate the following response using this checklist:",
            f"\nChecklist: {self.checklist.name}",
            f"Description: {self.checklist.description}",
            "\nItems:",
        ]

        for i, item in enumerate(self.checklist.items, 1):
            prompt_parts.append(f"\n{i}. {item.question}")
            if item.description:
                prompt_parts.append(f"   {item.description}")
            if item.required:
                prompt_parts.append("   [REQUIRED]")
            prompt_parts.append(f"   Weight: {item.weight:.2f}")

        if reference:
            prompt_parts.append(f"\nReference: {reference}")

        prompt_parts.append(f"\nResponse to evaluate: {prediction}")
        prompt_parts.append(
            "\nFor each item, answer YES or NO, and provide a brief justification."
            "\nFormat: ITEM_NUMBER. YES/NO - JUSTIFICATION"
        )

        return "\n".join(prompt_parts)

    def compute(
        self,
        prediction: str,
        reference: str = "",
        manual_answers: Optional[Dict[int, bool]] = None,
        **kwargs,
    ) -> EvaluationResult:
        """
        체크리스트 기반 평가 실행

        Args:
            prediction: 평가 대상 출력
            reference: 참조 출력 (선택적)
            manual_answers: 수동 평가 답변 {item_index: True/False} (선택적)
            **kwargs: 추가 파라미터

        Returns:
            EvaluationResult: 평가 결과
        """
        if manual_answers:
            # 수동 평가 사용
            return self._compute_manual(prediction, manual_answers)
        elif self.use_llm:
            # LLM 평가 사용
            return self._compute_llm(prediction, reference)
        else:
            raise ValueError("Either manual_answers must be provided or use_llm must be True")

    def _compute_manual(
        self,
        prediction: str,
        manual_answers: Dict[int, bool],
    ) -> EvaluationResult:
        """수동 평가 실행"""
        item_results = {}
        total_weighted_score = 0.0
        total_weight = 0.0
        required_failed = []

        for i, item in enumerate(self.checklist.items):
            answer = manual_answers.get(i)
            if answer is None:
                # 답변이 없으면 False로 처리
                answer = False

            score = 1.0 if answer else 0.0
            item_results[i] = {
                "question": item.question,
                "answer": answer,
                "score": score,
            }

            total_weighted_score += score * item.weight
            total_weight += item.weight

            # 필수 항목 실패 체크
            if item.required and not answer:
                required_failed.append(item.question)

        # 최종 점수 계산
        final_score = total_weighted_score / total_weight if total_weight > 0 else 0.0

        # 필수 항목 실패 시 점수 조정
        if required_failed:
            final_score = min(final_score, 0.5)  # 최대 0.5로 제한

        return EvaluationResult(
            metric_name=self.name,
            score=final_score,
            metadata={
                "checklist_name": self.checklist.name,
                "item_results": item_results,
                "required_failed": required_failed,
                "manual_evaluation": True,
            },
            explanation=self._generate_explanation(item_results, final_score, required_failed),
        )

    def _compute_llm(self, prediction: str, reference: str = "") -> EvaluationResult:
        """LLM 평가 실행"""
        client = self._get_client()

        # 체크리스트 평가 프롬프트 생성
        prompt = self._create_checklist_prompt(prediction, reference if reference else None)

        # LLM 평가
        response = client.chat([{"role": "user", "content": prompt}])
        llm_output = response.content

        # 결과 파싱
        item_answers = self._parse_llm_response(llm_output)

        item_results = {}
        total_weighted_score = 0.0
        total_weight = 0.0
        required_failed = []

        for i, item in enumerate(self.checklist.items):
            answer = item_answers.get(i, False)
            score = 1.0 if answer else 0.0
            item_results[i] = {
                "question": item.question,
                "answer": answer,
                "score": score,
            }

            total_weighted_score += score * item.weight
            total_weight += item.weight

            # 필수 항목 실패 체크
            if item.required and not answer:
                required_failed.append(item.question)

        # 최종 점수 계산
        final_score = total_weighted_score / total_weight if total_weight > 0 else 0.0

        # 필수 항목 실패 시 점수 조정
        if required_failed:
            final_score = min(final_score, 0.5)  # 최대 0.5로 제한

        return EvaluationResult(
            metric_name=self.name,
            score=final_score,
            metadata={
                "checklist_name": self.checklist.name,
                "item_results": item_results,
                "required_failed": required_failed,
                "llm_evaluation": True,
                "llm_output": llm_output,
            },
            explanation=self._generate_explanation(item_results, final_score, required_failed),
        )

    def _parse_llm_response(self, llm_output: str) -> Dict[int, bool]:
        """LLM 응답 파싱"""
        import re

        item_answers = {}

        # 각 항목별로 파싱
        for i, item in enumerate(self.checklist.items):
            # 패턴: "NUMBER. YES/NO - JUSTIFICATION"
            pattern = rf"{i + 1}\.\s*(YES|NO)\s*-\s*.+"
            match = re.search(pattern, llm_output, re.IGNORECASE | re.MULTILINE)

            if match:
                answer_str = match.group(1).upper()
                item_answers[i] = answer_str == "YES"
            else:
                # 대체 패턴 시도
                pattern2 = rf"{i + 1}\.\s*(YES|NO)"
                match2 = re.search(pattern2, llm_output, re.IGNORECASE)
                if match2:
                    answer_str = match2.group(1).upper()
                    item_answers[i] = answer_str == "YES"
                else:
                    # 기본값: False
                    item_answers[i] = False

        return item_answers

    def _generate_explanation(
        self,
        item_results: Dict[int, Dict[str, Any]],
        final_score: float,
        required_failed: List[str],
    ) -> str:
        """설명 생성"""
        parts = [
            f"Checklist: {self.checklist.name}",
            f"Final Score: {final_score:.3f}",
            "\nItem Results:",
        ]

        for i, item in enumerate(self.checklist.items):
            result = item_results.get(i, {})
            answer = result.get("answer", False)
            status = "✓" if answer else "✗"
            parts.append(f"  {status} {item.question}")

        if required_failed:
            parts.append("\nRequired Items Failed:")
            for question in required_failed:
                parts.append(f"  - {question}")

        return "\n".join(parts)
