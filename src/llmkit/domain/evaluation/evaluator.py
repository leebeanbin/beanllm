"""
Evaluator - 통합 평가기
"""

from typing import List, Optional

from .base_metric import BaseMetric
from .results import BatchEvaluationResult, EvaluationResult


class Evaluator:
    """
    통합 평가기

    여러 메트릭을 한 번에 실행
    """

    def __init__(self, metrics: Optional[List[BaseMetric]] = None):
        self.metrics = metrics or []

    def add_metric(self, metric: BaseMetric) -> "Evaluator":
        """메트릭 추가"""
        self.metrics.append(metric)
        return self

    def evaluate(self, prediction: str, reference: str, **kwargs) -> BatchEvaluationResult:
        """모든 메트릭으로 평가"""
        results = []

        for metric in self.metrics:
            try:
                result = metric.compute(prediction, reference, **kwargs)
                results.append(result)
            except Exception as e:
                # 에러가 나도 다른 메트릭은 계속 실행
                results.append(
                    EvaluationResult(metric_name=metric.name, score=0.0, metadata={"error": str(e)})
                )

        if not results:
            average_score = 0.0
        else:
            average_score = sum(r.score for r in results) / len(results)

        return BatchEvaluationResult(
            results=results, average_score=average_score, metadata={"metrics_count": len(results)}
        )

    def batch_evaluate(
        self, predictions: List[str], references: List[str], **kwargs
    ) -> List[BatchEvaluationResult]:
        """배치 평가"""
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        batch_results = []
        for pred, ref in zip(predictions, references):
            result = self.evaluate(pred, ref, **kwargs)
            batch_results.append(result)

        return batch_results
