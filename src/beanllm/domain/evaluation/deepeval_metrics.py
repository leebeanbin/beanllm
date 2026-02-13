"""
DeepEval metric creation and factory logic.

Provides functions and configuration to create DeepEval metric instances
(Answer Relevancy, Faithfulness, Contextual Precision/Recall, Hallucination,
Toxicity, Bias, Summarization, G-Eval). Used by DeepEvalWrapper for
evaluation execution.

Requirements:
    pip install deepeval

References:
    - https://github.com/confident-ai/deepeval
    - https://docs.confident-ai.com/
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Type

try:
    from beanllm.utils.logging import get_logger
except ImportError:

    def get_logger(name: str) -> logging.Logger:  # type: ignore[misc]
        return logging.getLogger(name)


logger = get_logger(__name__)

# Metric names supported by this module (no deepeval import required)
AVAILABLE_METRICS: List[str] = [
    "answer_relevancy",
    "faithfulness",
    "contextual_precision",
    "contextual_recall",
    "hallucination",
    "toxicity",
    "bias",
    "summarization",
    "geval",
]


def _get_metric_map() -> Dict[str, Type[Any]]:
    """
    Return mapping of metric name -> DeepEval metric class.

    Lazy-imports deepeval.metrics to avoid requiring deepeval at import time.

    Returns:
        Dict mapping metric name to metric class.
    """
    from deepeval.metrics import (  # type: ignore[import-untyped]
        AnswerRelevancyMetric,
        BiasMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
        GEval,
        HallucinationMetric,
        SummarizationMetric,
        ToxicityMetric,
    )

    return {
        "answer_relevancy": AnswerRelevancyMetric,
        "faithfulness": FaithfulnessMetric,
        "contextual_precision": ContextualPrecisionMetric,
        "contextual_recall": ContextualRecallMetric,
        "hallucination": HallucinationMetric,
        "toxicity": ToxicityMetric,
        "bias": BiasMetric,
        "summarization": SummarizationMetric,
        "geval": GEval,
    }


def get_metric_class(metric_name: str) -> Type[Any]:
    """
    Get the DeepEval metric class for a given metric name.

    Args:
        metric_name: One of answer_relevancy, faithfulness, contextual_precision,
            contextual_recall, hallucination, toxicity, bias, summarization, geval.

    Returns:
        The DeepEval metric class (not an instance).

    Raises:
        ValueError: If metric_name is not supported.
    """
    metric_map = _get_metric_map()
    if metric_name not in metric_map:
        raise ValueError(f"Unknown metric: {metric_name}. Available: {list(metric_map.keys())}")
    return metric_map[metric_name]


def create_metric(
    metric_name: str,
    model: str,
    threshold: float,
    include_reason: bool,
    async_mode: bool,
    **kwargs: Any,
) -> Any:
    """
    Create a DeepEval metric instance.

    Args:
        metric_name: Metric identifier (e.g. "answer_relevancy", "faithfulness").
        model: LLM model for the metric (e.g. "gpt-4o-mini").
        threshold: Pass/fail threshold.
        include_reason: Whether to include reason in results.
        async_mode: Whether to use async mode.
        **kwargs: Additional arguments passed to the metric constructor.

    Returns:
        An instantiated DeepEval metric object.
    """
    metric_class = get_metric_class(metric_name)
    metric = metric_class(
        model=model,
        threshold=threshold,
        include_reason=include_reason,
        async_mode=async_mode,
        **kwargs,
    )
    logger.info(f"DeepEval metric created: {metric_name}")
    return metric


def list_metric_names() -> List[str]:
    """
    Return the list of supported metric names.

    Returns:
        List of metric name strings.
    """
    return list(AVAILABLE_METRICS)
