"""
Evaluation Request DTOs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from beanllm.domain.evaluation.base_metric import BaseMetric


@dataclass(slots=True, kw_only=True)
class EvaluationRequest:
    """평가 요청 DTO"""

    prediction: str
    reference: str
    metrics: list["BaseMetric"] = field(default_factory=list)
    extra_params: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, kw_only=True)
class BatchEvaluationRequest:
    """배치 평가 요청 DTO"""

    predictions: list[str]
    references: list[str]
    metrics: list["BaseMetric"] = field(default_factory=list)
    extra_params: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, kw_only=True)
class TextEvaluationRequest:
    """텍스트 평가 요청 DTO (편의 함수용)"""

    prediction: str
    reference: str
    metrics: list[str] = field(default_factory=lambda: ["bleu", "rouge-1", "f1"])
    extra_params: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, kw_only=True)
class RAGEvaluationRequest:
    """RAG 평가 요청 DTO"""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: Optional[str] = None
    extra_params: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, kw_only=True)
class CreateEvaluatorRequest:
    """Evaluator 생성 요청 DTO"""

    metric_names: list[str]
