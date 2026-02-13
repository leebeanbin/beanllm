"""
Custom metric (user-defined compute function).
"""

from __future__ import annotations

from typing import Callable

from beanllm.domain.evaluation.base_metric import BaseMetric
from beanllm.domain.evaluation.enums import MetricType
from beanllm.domain.evaluation.results import EvaluationResult


class CustomMetric(BaseMetric):
    """
    사용자 정의 메트릭

    커스텀 평가 함수를 사용하여 메트릭 생성
    """

    def __init__(
        self,
        name: str,
        compute_fn: Callable[[str, str], float],
        metric_type: MetricType = MetricType.CUSTOM,
    ) -> None:
        super().__init__(name, metric_type)
        self.compute_fn = compute_fn

    def compute(self, prediction: str, reference: str, **kwargs) -> EvaluationResult:
        score = self.compute_fn(prediction, reference)

        return EvaluationResult(
            metric_name=self.name,
            score=score,
            metadata={"type": "custom"},
        )
