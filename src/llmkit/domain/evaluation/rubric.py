"""
Rubric-Driven Grading - 루브릭 기반 평가
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base_metric import BaseMetric
from .enums import MetricType
from .results import EvaluationResult


@dataclass
class RubricCriterion:
    """루브릭 기준"""

    name: str
    description: str
    weight: float = 1.0  # 가중치
    levels: Optional[Dict[str, float]] = None  # {"excellent": 1.0, "good": 0.8, ...}

    def __post_init__(self):
        """검증"""
        if self.weight < 0:
            raise ValueError(f"Weight must be non-negative, got {self.weight}")
        if self.levels is None:
            # 기본 레벨 설정
            self.levels = {
                "excellent": 1.0,
                "good": 0.8,
                "satisfactory": 0.6,
                "needs_improvement": 0.4,
                "poor": 0.2,
            }


@dataclass
class Rubric:
    """루브릭"""

    name: str
    description: str
    criteria: List[RubricCriterion]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """검증"""
        if not self.criteria:
            raise ValueError("Rubric must have at least one criterion")
        # 가중치 정규화
        total_weight = sum(c.weight for c in self.criteria)
        if total_weight > 0:
            for criterion in self.criteria:
                criterion.weight = criterion.weight / total_weight


class RubricGrader(BaseMetric):
    """
    루브릭 기반 평가기

    구조화된 루브릭을 사용하여 출력을 평가
    """

    def __init__(
        self,
        rubric: Rubric,
        client=None,
        use_llm: bool = True,
    ):
        """
        Args:
            rubric: 평가 루브릭
            client: LLM 클라이언트 (use_llm=True일 때 필요)
            use_llm: LLM을 사용하여 평가할지 여부 (False면 수동 평가만)
        """
        super().__init__(f"rubric_{rubric.name}", MetricType.QUALITY)
        self.rubric = rubric
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

    def _create_rubric_prompt(self, prediction: str, reference: Optional[str] = None) -> str:
        """루브릭 평가 프롬프트 생성"""
        prompt_parts = [
            "Evaluate the following response using this rubric:",
            f"\nRubric: {self.rubric.name}",
            f"Description: {self.rubric.description}",
            "\nCriteria:",
        ]

        for i, criterion in enumerate(self.rubric.criteria, 1):
            prompt_parts.append(f"\n{i}. {criterion.name} (weight: {criterion.weight:.2f})")
            prompt_parts.append(f"   {criterion.description}")
            if criterion.levels:
                prompt_parts.append("   Levels:")
                for level, score in criterion.levels.items():
                    prompt_parts.append(f"     - {level}: {score:.1f}")

        if reference:
            prompt_parts.append(f"\nReference: {reference}")

        prompt_parts.append(f"\nResponse to evaluate: {prediction}")
        prompt_parts.append(
            "\nFor each criterion, provide:"
            "\n1. The level (excellent, good, satisfactory, needs_improvement, poor)"
            "\n2. A brief justification"
            "\nFormat: CRITERION_NAME: LEVEL - JUSTIFICATION"
        )

        return "\n".join(prompt_parts)

    def compute(
        self,
        prediction: str,
        reference: str = "",
        manual_scores: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> EvaluationResult:
        """
        루브릭 기반 평가 실행

        Args:
            prediction: 평가 대상 출력
            reference: 참조 출력 (선택적)
            manual_scores: 수동 평가 점수 {criterion_name: level} (선택적)
            **kwargs: 추가 파라미터

        Returns:
            EvaluationResult: 평가 결과
        """
        if manual_scores:
            # 수동 평가 사용
            return self._compute_manual(prediction, manual_scores)
        elif self.use_llm:
            # LLM 평가 사용
            return self._compute_llm(prediction, reference)
        else:
            raise ValueError("Either manual_scores must be provided or use_llm must be True")

    def _compute_manual(
        self,
        prediction: str,
        manual_scores: Dict[str, str],
    ) -> EvaluationResult:
        """수동 평가 실행"""
        criterion_scores = {}
        total_weighted_score = 0.0
        total_weight = 0.0

        for criterion in self.rubric.criteria:
            level = manual_scores.get(criterion.name)
            if level is None:
                continue

            # 레벨에서 점수 추출
            if criterion.levels and level in criterion.levels:
                score = criterion.levels[level]
            else:
                # 기본 레벨 매핑
                level_lower = level.lower()
                if level_lower in ["excellent", "excellent"]:
                    score = 1.0
                elif level_lower in ["good", "good"]:
                    score = 0.8
                elif level_lower in ["satisfactory", "satisfactory"]:
                    score = 0.6
                elif level_lower in ["needs_improvement", "needs improvement"]:
                    score = 0.4
                elif level_lower in ["poor", "poor"]:
                    score = 0.2
                else:
                    score = 0.5  # 기본값

            criterion_scores[criterion.name] = score
            total_weighted_score += score * criterion.weight
            total_weight += criterion.weight

        # 최종 점수 계산
        final_score = total_weighted_score / total_weight if total_weight > 0 else 0.0

        return EvaluationResult(
            metric_name=self.name,
            score=final_score,
            metadata={
                "rubric_name": self.rubric.name,
                "criterion_scores": criterion_scores,
                "manual_evaluation": True,
            },
            explanation=self._generate_explanation(criterion_scores, final_score),
        )

    def _compute_llm(self, prediction: str, reference: str = "") -> EvaluationResult:
        """LLM 평가 실행"""
        client = self._get_client()

        # 루브릭 평가 프롬프트 생성
        prompt = self._create_rubric_prompt(prediction, reference if reference else None)

        # LLM 평가
        response = client.chat([{"role": "user", "content": prompt}])
        llm_output = response.content

        # 결과 파싱
        criterion_scores = self._parse_llm_response(llm_output)
        total_weighted_score = 0.0
        total_weight = 0.0

        for criterion in self.rubric.criteria:
            level = criterion_scores.get(criterion.name, {}).get("level")
            if level is None:
                continue

            # 레벨에서 점수 추출
            if criterion.levels and level in criterion.levels:
                score = criterion.levels[level]
            else:
                # 기본 레벨 매핑
                level_lower = level.lower()
                if "excellent" in level_lower:
                    score = 1.0
                elif "good" in level_lower:
                    score = 0.8
                elif "satisfactory" in level_lower:
                    score = 0.6
                elif "needs" in level_lower or "improvement" in level_lower:
                    score = 0.4
                elif "poor" in level_lower:
                    score = 0.2
                else:
                    score = 0.5  # 기본값

            total_weighted_score += score * criterion.weight
            total_weight += criterion.weight

        # 최종 점수 계산
        final_score = total_weighted_score / total_weight if total_weight > 0 else 0.0

        # 점수만 추출 (메타데이터용)
        score_dict = {name: data.get("level", "unknown") for name, data in criterion_scores.items()}

        return EvaluationResult(
            metric_name=self.name,
            score=final_score,
            metadata={
                "rubric_name": self.rubric.name,
                "criterion_scores": score_dict,
                "llm_evaluation": True,
                "llm_output": llm_output,
            },
            explanation=self._generate_explanation(score_dict, final_score),
        )

    def _parse_llm_response(self, llm_output: str) -> Dict[str, Dict[str, str]]:
        """LLM 응답 파싱"""
        import re

        criterion_scores = {}

        # 각 기준별로 파싱
        for criterion in self.rubric.criteria:
            # 패턴: "CRITERION_NAME: LEVEL - JUSTIFICATION"
            pattern = rf"{re.escape(criterion.name)}:\s*(\w+)\s*-\s*(.+?)(?=\n|$)"
            match = re.search(pattern, llm_output, re.IGNORECASE | re.MULTILINE)

            if match:
                level = match.group(1).strip()
                justification = match.group(2).strip()
                criterion_scores[criterion.name] = {
                    "level": level,
                    "justification": justification,
                }
            else:
                # 대체 패턴 시도
                pattern2 = rf"{re.escape(criterion.name)}[:\s]+(\w+)"
                match2 = re.search(pattern2, llm_output, re.IGNORECASE)
                if match2:
                    level = match2.group(1).strip()
                    criterion_scores[criterion.name] = {
                        "level": level,
                        "justification": "",
                    }

        return criterion_scores

    def _generate_explanation(
        self,
        criterion_scores: Dict[str, Any],
        final_score: float,
    ) -> str:
        """설명 생성"""
        parts = [
            f"Rubric: {self.rubric.name}",
            f"Final Score: {final_score:.3f}",
            "\nCriterion Scores:",
        ]

        for criterion in self.rubric.criteria:
            score_info = criterion_scores.get(criterion.name)
            if score_info:
                if isinstance(score_info, dict):
                    level = score_info.get("level", "unknown")
                    justification = score_info.get("justification", "")
                    parts.append(f"  - {criterion.name}: {level}")
                    if justification:
                        parts.append(f"    {justification}")
                else:
                    parts.append(f"  - {criterion.name}: {score_info}")
            else:
                parts.append(f"  - {criterion.name}: not evaluated")

        return "\n".join(parts)
