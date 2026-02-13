"""
Model Router - Intelligent Model Selection (core router).

Types and strategies are in router_types.py and router_strategies.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from beanllm.utils.logging import get_logger
except ImportError:
    import logging

    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


from beanllm.infrastructure.routing.router_strategies import (
    estimate_cost as _estimate_cost,
)
from beanllm.infrastructure.routing.router_strategies import (
    filter_by_capabilities as _filter_by_capabilities,
)
from beanllm.infrastructure.routing.router_strategies import (
    generate_reason as _generate_reason,
)
from beanllm.infrastructure.routing.router_strategies import (
    score_models as _score_models,
)
from beanllm.infrastructure.routing.router_types import (
    DEFAULT_MODELS,
    ModelInfo,
    RequestCharacteristics,
    RoutingDecision,
    RoutingStrategy,
)

logger = get_logger(__name__)

__all__ = [
    "ModelRouter",
    "RoutingStrategy",
    "RequestCharacteristics",
    "ModelInfo",
    "RoutingDecision",
    "DEFAULT_MODELS",
    "create_default_router",
]


class ModelRouter:
    """
    모델 라우터

    요청 특성에 따라 최적의 LLM 모델을 선택

    Example:
        ```python
        # Initialize router
        router = ModelRouter(strategy=RoutingStrategy.BALANCED)

        # Register models
        router.register_model(ModelInfo(
            provider="openai",
            model_id="gpt-4",
            context_window=8000,
            cost_per_1k_input=0.03,
            cost_per_1k_output=0.06,
            quality_score=0.95,
            supports_vision=True,
            supports_function_calling=True,
        ))
        router.register_model(ModelInfo(
            provider="openai",
            model_id="gpt-3.5-turbo",
            context_window=4000,
            cost_per_1k_input=0.0015,
            cost_per_1k_output=0.002,
            quality_score=0.7,
            supports_function_calling=True,
        ))

        # Route request
        request = RequestCharacteristics(
            prompt_length=1500,
            requires_function_calling=True,
            complexity_score=0.6,
        )
        decision = router.route(request)
        print(f"Selected: {decision.selected_model.model_id}")
        print(f"Reason: {decision.reason}")
        ```
    """

    def __init__(
        self,
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        enable_fallback: bool = True,
        max_fallback_attempts: int = 3,
    ):
        """
        초기화

        Args:
            strategy: 라우팅 전략
            enable_fallback: Fallback 활성화
            max_fallback_attempts: 최대 Fallback 시도 횟수
        """
        self.strategy = strategy
        self.enable_fallback = enable_fallback
        self.max_fallback_attempts = max_fallback_attempts

        # Registered models
        self.models: List[ModelInfo] = []

        # Model statistics (for adaptive routing)
        self.model_stats: Dict[str, Dict[str, Any]] = {}

        logger.info(f"ModelRouter initialized with strategy: {strategy}")

    def register_model(self, model: ModelInfo):
        """
        모델 등록

        Args:
            model: 모델 정보
        """
        self.models.append(model)
        model_key = f"{model.provider}:{model.model_id}"
        self.model_stats[model_key] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_latency": 0.0,
        }
        logger.info(f"Registered model: {model_key}")

    def register_models(self, models: List[ModelInfo]):
        """여러 모델 일괄 등록"""
        for model in models:
            self.register_model(model)

    def route(
        self,
        request: RequestCharacteristics,
        exclude_models: Optional[List[str]] = None,
    ) -> RoutingDecision:
        """
        요청 라우팅 - 최적 모델 선택

        Args:
            request: 요청 특성
            exclude_models: 제외할 모델 리스트 (model_id)

        Returns:
            라우팅 결정

        Raises:
            ValueError: 적합한 모델이 없는 경우
        """
        if not self.models:
            raise ValueError("No models registered")

        eligible_models = _filter_by_capabilities(self.models, request, exclude_models)
        if not eligible_models:
            raise ValueError("No eligible models found matching the required capabilities")

        scored_models = _score_models(
            eligible_models,
            request,
            self.strategy,
            self._get_success_rate,
        )
        scored_models.sort(key=lambda x: x[1], reverse=True)
        selected_model, score = scored_models[0]
        fallback_models = [m for m, _ in scored_models[1 : self.max_fallback_attempts + 1]]
        estimated_cost = _estimate_cost(selected_model, request)
        reason = _generate_reason(selected_model, request, self.strategy)

        decision = RoutingDecision(
            selected_model=selected_model,
            reason=reason,
            fallback_models=fallback_models,
            estimated_cost=estimated_cost,
            confidence_score=score,
        )

        logger.info(
            f"Routed to {selected_model.provider}:{selected_model.model_id} "
            f"(score: {score:.3f}, reason: {reason})"
        )

        return decision

    def _get_success_rate(self, model: ModelInfo) -> float:
        """
        모델의 성공률 반환

        Args:
            model: 모델

        Returns:
            성공률 (0-1)
        """
        model_key = f"{model.provider}:{model.model_id}"
        stats = self.model_stats.get(model_key, {})

        total = stats.get("total_requests", 0)
        successful = stats.get("successful_requests", 0)

        if total == 0:
            return 1.0  # No history, assume 100%

        return float(successful / total)

    def record_result(
        self,
        model: ModelInfo,
        success: bool,
        latency: Optional[float] = None,
    ):
        """
        모델 실행 결과 기록 (통계 업데이트)

        Args:
            model: 모델
            success: 성공 여부
            latency: 지연 시간 (초)
        """
        model_key = f"{model.provider}:{model.model_id}"
        stats = self.model_stats[model_key]

        stats["total_requests"] += 1
        if success:
            stats["successful_requests"] += 1
        else:
            stats["failed_requests"] += 1

        if latency is not None:
            # Update average latency (exponential moving average)
            alpha = 0.3
            stats["avg_latency"] = alpha * latency + (1 - alpha) * stats["avg_latency"]

    def get_stats(self) -> Dict[str, Any]:
        """라우터 통계 반환"""
        return {
            "strategy": self.strategy.value,
            "registered_models": len(self.models),
            "enable_fallback": self.enable_fallback,
            "model_stats": self.model_stats,
        }


def create_default_router(strategy: RoutingStrategy = RoutingStrategy.BALANCED) -> ModelRouter:
    """Create router with default model pool."""
    router = ModelRouter(strategy=strategy)
    router.register_models(DEFAULT_MODELS)
    return router
