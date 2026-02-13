"""Model Router - Routing strategy functions."""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

from beanllm.infrastructure.routing.router_types import (
    ModelInfo,
    RequestCharacteristics,
    RoutingStrategy,
)


def filter_by_capabilities(
    models: List[ModelInfo],
    request: RequestCharacteristics,
    exclude_models: Optional[List[str]],
) -> List[ModelInfo]:
    """Filter models by capability and constraints."""
    eligible = []
    for model in models:
        if exclude_models and model.model_id in exclude_models:
            continue
        if model.context_window < request.context_window_needed:
            continue
        if request.requires_vision and not model.supports_vision:
            continue
        if request.requires_function_calling and not model.supports_function_calling:
            continue
        if request.requires_json_mode and not model.supports_json_mode:
            continue
        if request.max_cost_per_1k and model.cost_per_1k_input > request.max_cost_per_1k:
            continue
        if request.min_quality_score and model.quality_score < request.min_quality_score:
            continue
        eligible.append(model)
    return eligible


def score_cost_optimized(model: ModelInfo, models: List[ModelInfo]) -> float:
    """Cost-optimized score."""
    max_cost = max(m.cost_per_1k_input for m in models)
    if max_cost == 0:
        return 1.0
    cost_score = 1.0 - (model.cost_per_1k_input / max_cost)
    return cost_score * 0.9 + model.quality_score * 0.1


def score_quality_optimized(model: ModelInfo, models: List[ModelInfo]) -> float:
    """Quality-optimized score."""
    max_cost = max(m.cost_per_1k_input for m in models)
    cost_score = 1.0 - (model.cost_per_1k_input / max_cost) if max_cost > 0 else 1.0
    return model.quality_score * 0.9 + cost_score * 0.1


def score_balanced(model: ModelInfo, models: List[ModelInfo]) -> float:
    """Balanced score."""
    max_cost = max(m.cost_per_1k_input for m in models)
    cost_score = 1.0 - (model.cost_per_1k_input / max_cost) if max_cost > 0 else 1.0
    latency_score = 1.0 - model.latency_score
    return model.quality_score * 0.4 + cost_score * 0.4 + latency_score * 0.2


def score_complexity_based(
    model: ModelInfo,
    models: List[ModelInfo],
    request: RequestCharacteristics,
) -> float:
    """Complexity-based score."""
    complexity = request.complexity_score or 0.5
    quality_match = 1.0 - abs(model.quality_score - complexity)
    max_cost = max(m.cost_per_1k_input for m in models)
    cost_score = 1.0 - (model.cost_per_1k_input / max_cost) if max_cost > 0 else 1.0
    return quality_match * 0.6 + cost_score * 0.4


def score_capability_match(
    model: ModelInfo,
    models: List[ModelInfo],
    request: RequestCharacteristics,
) -> float:
    """Capability-match score."""
    required_caps = sum(
        [request.requires_vision, request.requires_function_calling, request.requires_json_mode]
    )
    model_caps = sum(
        [model.supports_vision, model.supports_function_calling, model.supports_json_mode]
    )
    over_qualified_penalty = (model_caps - required_caps) * 0.2 if model_caps > required_caps else 0.0
    max_cost = max(m.cost_per_1k_input for m in models)
    cost_score = 1.0 - (model.cost_per_1k_input / max_cost) if max_cost > 0 else 1.0
    return cost_score - over_qualified_penalty


def score_models(
    models: List[ModelInfo],
    request: RequestCharacteristics,
    strategy: RoutingStrategy,
    get_success_rate: Callable[[ModelInfo], float],
) -> List[Tuple[ModelInfo, float]]:
    """Score models by strategy."""
    scored = []
    for model in models:
        if strategy == RoutingStrategy.COST_OPTIMIZED:
            score = score_cost_optimized(model, models)
        elif strategy == RoutingStrategy.QUALITY_OPTIMIZED:
            score = score_quality_optimized(model, models)
        elif strategy == RoutingStrategy.BALANCED:
            score = score_balanced(model, models)
        elif strategy == RoutingStrategy.COMPLEXITY_BASED:
            score = score_complexity_based(model, models, request)
        elif strategy == RoutingStrategy.CAPABILITY_MATCH:
            score = score_capability_match(model, models, request)
        else:
            score = 0.5
        score *= model.reliability_score
        score *= get_success_rate(model)
        scored.append((model, score))
    return scored


def estimate_cost(model: ModelInfo, request: RequestCharacteristics) -> float:
    """Estimate cost for model and request."""
    input_tokens = request.prompt_length / 4
    output_tokens = 500
    input_cost = (input_tokens / 1000) * model.cost_per_1k_input
    output_cost = (output_tokens / 1000) * model.cost_per_1k_output
    return input_cost + output_cost


def generate_reason(
    model: ModelInfo,
    request: RequestCharacteristics,
    strategy: RoutingStrategy,
) -> str:
    """Generate selection reason string."""
    reasons = []
    if strategy == RoutingStrategy.COST_OPTIMIZED:
        reasons.append(f"Cost-optimized ({model.cost_per_1k_input:.4f}$/1k)")
    elif strategy == RoutingStrategy.QUALITY_OPTIMIZED:
        reasons.append(f"Quality-optimized (score: {model.quality_score:.2f})")
    elif strategy == RoutingStrategy.BALANCED:
        reasons.append("Balanced cost and quality")
    elif strategy == RoutingStrategy.COMPLEXITY_BASED:
        complexity = request.complexity_score or 0.5
        reasons.append(f"Complexity match ({complexity:.2f})")
    elif strategy == RoutingStrategy.CAPABILITY_MATCH:
        reasons.append("Capability match (minimal over-qualification)")
    if request.requires_vision and model.supports_vision:
        reasons.append("vision support")
    if request.requires_function_calling and model.supports_function_calling:
        reasons.append("function calling")
    return ", ".join(reasons)
