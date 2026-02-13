"""
LLM-as-Judge metric.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

from beanllm.domain.evaluation.base_metric import BaseMetric
from beanllm.domain.evaluation.enums import MetricType
from beanllm.domain.evaluation.results import EvaluationResult

if TYPE_CHECKING:
    from beanllm.domain.evaluation.protocols import LLMClientProtocol


class LLMJudgeMetric(BaseMetric):
    """
    LLM-as-a-Judge

    LLM을 사용하여 출력 품질 평가
    """

    def __init__(
        self,
        client: Optional["LLMClientProtocol"] = None,
        criterion: str = "quality",
        use_reference: bool = True,
    ) -> None:
        super().__init__(f"llm_judge_{criterion}", MetricType.QUALITY)
        self.client = client
        self.criterion = criterion
        self.use_reference = use_reference

    def _get_client(self):
        """클라이언트 반환 (생성자에서 주입 필수)"""
        if self.client is None:
            raise RuntimeError(
                "LLM client not available. "
                "Please provide a client via constructor: "
                "LLMJudge(client=your_client)"
            )
        return self.client

    def _create_judge_prompt(
        self,
        prediction: str,
        reference: Optional[str],
        criterion: str,
    ) -> str:
        """Judge 프롬프트 생성"""
        if criterion == "quality":
            instruction = (
                "Evaluate the quality of the response. "
                "Consider accuracy, completeness, and clarity."
            )
        elif criterion == "relevance":
            instruction = (
                "Evaluate how relevant the response is to the reference. "
                "Consider whether it addresses the same topic and intent."
            )
        elif criterion == "factuality":
            instruction = (
                "Evaluate the factual accuracy of the response. "
                "Check if the information is correct and verifiable."
            )
        elif criterion == "coherence":
            instruction = (
                "Evaluate the coherence of the response. "
                "Check if it's well-structured and logically consistent."
            )
        elif criterion == "helpfulness":
            instruction = (
                "Evaluate how helpful the response is. "
                "Consider usefulness, actionability, and clarity."
            )
        else:
            instruction = f"Evaluate the {criterion} of the response."

        prompt_parts = [instruction]

        if self.use_reference and reference:
            prompt_parts.append(f"\nReference: {reference}")

        prompt_parts.append(f"\nResponse to evaluate: {prediction}")
        prompt_parts.append(
            "\nProvide a score from 0 to 1 (where 1 is best) and a brief explanation."
            "\nFormat your response as: SCORE: <number> EXPLANATION: <text>"
        )

        return "\n".join(prompt_parts)

    def compute(self, prediction: str, reference: str, **kwargs) -> EvaluationResult:
        client = self._get_client()

        # Judge 프롬프트 생성
        prompt = self._create_judge_prompt(
            prediction,
            reference if self.use_reference else None,
            self.criterion,
        )

        # LLM 평가
        response = client.chat([{"role": "user", "content": prompt}])
        judge_output = response.content

        # 점수 추출
        score_match = re.search(r"SCORE:\s*([\d.]+)", judge_output)
        if score_match:
            score = float(score_match.group(1))
        else:
            # 폴백: 0-10 스케일 찾기
            score_match = re.search(
                r"(\d+(?:\.\d+)?)\s*(?:out of|/)\s*(?:10|1)",
                judge_output,
            )
            if score_match:
                score = float(score_match.group(1))
                if score > 1:
                    score = score / 10
            else:
                score = 0.5  # 기본값

        # 설명 추출
        explanation_match = re.search(r"EXPLANATION:\s*(.+)", judge_output, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else judge_output

        return EvaluationResult(
            metric_name=self.name,
            score=score,
            metadata={"criterion": self.criterion},
            explanation=explanation,
        )
