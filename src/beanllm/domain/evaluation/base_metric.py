"""
BaseMetric - 메트릭 베이스 클래스
"""

from abc import ABC, abstractmethod
from typing import List

from .enums import MetricType
from .results import BatchEvaluationResult, EvaluationResult


class BaseMetric(ABC):
    """메트릭 베이스 클래스"""

    def __init__(self, name: str, metric_type: MetricType):
        self.name = name
        self.metric_type = metric_type

    @abstractmethod
    def compute(self, prediction: str, reference: str, **kwargs) -> EvaluationResult:
        """메트릭 계산"""
        pass

    def batch_compute(
        self, predictions: List[str], references: List[str], **kwargs
    ) -> BatchEvaluationResult:
        """배치 평가"""
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        results = []
        for pred, ref in zip(predictions, references):
            result = self.compute(pred, ref, **kwargs)
            results.append(result)

        average_score = sum(r.score for r in results) / len(results)

        return BatchEvaluationResult(
            results=results, average_score=average_score, metadata={"count": len(results)}
        )
