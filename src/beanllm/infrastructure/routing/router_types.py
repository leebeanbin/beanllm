"""Model Router - Type definitions and config."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class RoutingStrategy(str, Enum):
    """라우팅 전략"""

    COST_OPTIMIZED = "cost_optimized"
    QUALITY_OPTIMIZED = "quality_optimized"
    BALANCED = "balanced"
    COMPLEXITY_BASED = "complexity_based"
    CAPABILITY_MATCH = "capability_match"


@dataclass
class RequestCharacteristics:
    """요청 특성"""

    prompt_length: int
    requires_vision: bool = False
    requires_function_calling: bool = False
    requires_json_mode: bool = False
    complexity_score: Optional[float] = None
    max_cost_per_1k: Optional[float] = None
    min_quality_score: Optional[float] = None
    context_window_needed: int = 8000


@dataclass
class ModelInfo:
    """모델 정보"""

    provider: str
    model_id: str
    context_window: int
    cost_per_1k_input: float
    cost_per_1k_output: float
    quality_score: float
    supports_vision: bool = False
    supports_function_calling: bool = False
    supports_json_mode: bool = False
    latency_score: float = 0.5
    reliability_score: float = 1.0


@dataclass
class RoutingDecision:
    """라우팅 결정"""

    selected_model: ModelInfo
    reason: str
    fallback_models: List[ModelInfo]
    estimated_cost: float
    confidence_score: float
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


DEFAULT_MODELS: List[ModelInfo] = [
    ModelInfo(
        provider="openai",
        model_id="gpt-4-turbo",
        context_window=128000,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
        quality_score=0.95,
        supports_vision=True,
        supports_function_calling=True,
        supports_json_mode=True,
        latency_score=0.4,
    ),
    ModelInfo(
        provider="openai",
        model_id="gpt-4",
        context_window=8000,
        cost_per_1k_input=0.03,
        cost_per_1k_output=0.06,
        quality_score=0.95,
        supports_vision=False,
        supports_function_calling=True,
        supports_json_mode=True,
        latency_score=0.5,
    ),
    ModelInfo(
        provider="openai",
        model_id="gpt-3.5-turbo",
        context_window=16000,
        cost_per_1k_input=0.0005,
        cost_per_1k_output=0.0015,
        quality_score=0.7,
        supports_function_calling=True,
        supports_json_mode=True,
        latency_score=0.2,
    ),
    ModelInfo(
        provider="anthropic",
        model_id="claude-3-opus",
        context_window=200000,
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
        quality_score=0.98,
        supports_vision=True,
        supports_function_calling=True,
        latency_score=0.5,
    ),
    ModelInfo(
        provider="anthropic",
        model_id="claude-3-sonnet",
        context_window=200000,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        quality_score=0.85,
        supports_vision=True,
        supports_function_calling=True,
        latency_score=0.3,
    ),
    ModelInfo(
        provider="google",
        model_id="gemini-1.5-pro",
        context_window=1000000,
        cost_per_1k_input=0.0035,
        cost_per_1k_output=0.0105,
        quality_score=0.9,
        supports_vision=True,
        supports_function_calling=True,
        latency_score=0.4,
    ),
]
