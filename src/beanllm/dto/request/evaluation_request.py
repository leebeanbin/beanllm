"""
Evaluation Request DTOs
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from ...domain.evaluation.base_metric import BaseMetric


class EvaluationRequest:
    """평가 요청 DTO"""

    def __init__(
        self,
        prediction: str,
        reference: str,
        metrics: Optional[List[BaseMetric]] = None,
        **kwargs: Any,
    ):
        self.prediction = prediction
        self.reference = reference
        self.metrics = metrics or []
        self.kwargs = kwargs


class BatchEvaluationRequest:
    """배치 평가 요청 DTO"""

    def __init__(
        self,
        predictions: List[str],
        references: List[str],
        metrics: Optional[List[BaseMetric]] = None,
        **kwargs: Any,
    ):
        self.predictions = predictions
        self.references = references
        self.metrics = metrics or []
        self.kwargs = kwargs


class TextEvaluationRequest:
    """텍스트 평가 요청 DTO (편의 함수용)"""

    def __init__(
        self,
        prediction: str,
        reference: str,
        metrics: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        self.prediction = prediction
        self.reference = reference
        self.metrics = metrics or ["bleu", "rouge-1", "f1"]
        self.kwargs = kwargs


class RAGEvaluationRequest:
    """RAG 평가 요청 DTO"""

    def __init__(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        **kwargs: Any,
    ):
        self.question = question
        self.answer = answer
        self.contexts = contexts
        self.ground_truth = ground_truth
        self.kwargs = kwargs


class CreateEvaluatorRequest:
    """Evaluator 생성 요청 DTO"""

    def __init__(self, metric_names: List[str]):
        self.metric_names = metric_names
