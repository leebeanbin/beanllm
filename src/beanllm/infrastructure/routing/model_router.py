"""
Model Router - Intelligent Model Selection

책임:
- 요청 특성 분석
- 최적 모델 선택
- Failover/Fallback 처리

SOLID:
- SRP: 모델 라우팅만 담당
- OCP: 새로운 라우팅 전략 쉽게 추가
- DIP: 라우팅 규칙 인터페이스에 의존
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

try:
    from beanllm.utils.logging import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


class RoutingStrategy(str, Enum):
    """라우팅 전략"""

    COST_OPTIMIZED = "cost_optimized"  # 비용 최적화 (가장 저렴한 모델)
    QUALITY_OPTIMIZED = "quality_optimized"  # 품질 최적화 (가장 강력한 모델)
    BALANCED = "balanced"  # 균형 (비용과 품질의 균형)
    COMPLEXITY_BASED = "complexity_based"  # 복잡도 기반 (쿼리 복잡도에 따라)
    CAPABILITY_MATCH = "capability_match"  # 기능 매칭 (필요한 기능만 지원하는 모델)


@dataclass
class RequestCharacteristics:
    """요청 특성"""

    prompt_length: int  # 프롬프트 길이
    requires_vision: bool = False  # 비전 기능 필요
    requires_function_calling: bool = False  # 함수 호출 필요
    requires_json_mode: bool = False  # JSON 모드 필요
    complexity_score: Optional[float] = None  # 복잡도 점수 (0-1)
    max_cost_per_1k: Optional[float] = None  # 최대 비용 (1k tokens당)
    min_quality_score: Optional[float] = None  # 최소 품질 점수 (0-1)
    context_window_needed: int = 8000  # 필요한 컨텍스트 윈도우


@dataclass
class ModelInfo:
    """모델 정보"""

    provider: str  # 제공자 (openai, anthropic, google, etc.)
    model_id: str  # 모델 ID
    context_window: int  # 컨텍스트 윈도우 크기
    cost_per_1k_input: float  # 입력 1k tokens당 비용 ($)
    cost_per_1k_output: float  # 출력 1k tokens당 비용 ($)
    quality_score: float  # 품질 점수 (0-1)
    supports_vision: bool = False  # 비전 지원
    supports_function_calling: bool = False  # 함수 호출 지원
    supports_json_mode: bool = False  # JSON 모드 지원
    latency_score: float = 0.5  # 지연 점수 (0-1, 낮을수록 빠름)
    reliability_score: float = 1.0  # 신뢰성 점수 (0-1)


@dataclass
class RoutingDecision:
    """라우팅 결정"""

    selected_model: ModelInfo  # 선택된 모델
    reason: str  # 선택 이유
    fallback_models: List[ModelInfo]  # Fallback 모델 리스트
    estimated_cost: float  # 예상 비용
    confidence_score: float  # 신뢰도 점수 (0-1)
    metadata: Dict[str, Any] = None  # 추가 메타데이터

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


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

        # Filter by capabilities
        eligible_models = self._filter_by_capabilities(request, exclude_models)

        if not eligible_models:
            raise ValueError("No eligible models found matching the required capabilities")

        # Score models based on strategy
        scored_models = self._score_models(eligible_models, request)

        # Sort by score (descending)
        scored_models.sort(key=lambda x: x[1], reverse=True)

        # Select best model
        selected_model, score = scored_models[0]

        # Prepare fallback models
        fallback_models = [model for model, _ in scored_models[1 : self.max_fallback_attempts + 1]]

        # Estimate cost
        estimated_cost = self._estimate_cost(selected_model, request)

        # Generate reason
        reason = self._generate_reason(selected_model, request, score)

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

    def _filter_by_capabilities(
        self,
        request: RequestCharacteristics,
        exclude_models: Optional[List[str]],
    ) -> List[ModelInfo]:
        """
        기능 요구사항으로 모델 필터링

        Args:
            request: 요청 특성
            exclude_models: 제외할 모델 리스트

        Returns:
            적합한 모델 리스트
        """
        eligible = []

        for model in self.models:
            # Exclude if in exclusion list
            if exclude_models and model.model_id in exclude_models:
                continue

            # Check context window
            if model.context_window < request.context_window_needed:
                continue

            # Check vision support
            if request.requires_vision and not model.supports_vision:
                continue

            # Check function calling support
            if request.requires_function_calling and not model.supports_function_calling:
                continue

            # Check JSON mode support
            if request.requires_json_mode and not model.supports_json_mode:
                continue

            # Check cost constraint
            if request.max_cost_per_1k and model.cost_per_1k_input > request.max_cost_per_1k:
                continue

            # Check quality constraint
            if request.min_quality_score and model.quality_score < request.min_quality_score:
                continue

            eligible.append(model)

        return eligible

    def _score_models(
        self,
        models: List[ModelInfo],
        request: RequestCharacteristics,
    ) -> List[Tuple[ModelInfo, float]]:
        """
        전략에 따라 모델 점수 계산

        Args:
            models: 모델 리스트
            request: 요청 특성

        Returns:
            [(모델, 점수), ...] 리스트
        """
        scored = []

        for model in models:
            if self.strategy == RoutingStrategy.COST_OPTIMIZED:
                score = self._score_cost_optimized(model, request)
            elif self.strategy == RoutingStrategy.QUALITY_OPTIMIZED:
                score = self._score_quality_optimized(model, request)
            elif self.strategy == RoutingStrategy.BALANCED:
                score = self._score_balanced(model, request)
            elif self.strategy == RoutingStrategy.COMPLEXITY_BASED:
                score = self._score_complexity_based(model, request)
            elif self.strategy == RoutingStrategy.CAPABILITY_MATCH:
                score = self._score_capability_match(model, request)
            else:
                score = 0.5  # Default

            # Adjust by reliability
            score *= model.reliability_score

            # Adjust by historical performance
            score *= self._get_success_rate(model)

            scored.append((model, score))

        return scored

    def _score_cost_optimized(self, model: ModelInfo, request: RequestCharacteristics) -> float:
        """비용 최적화 점수 (낮은 비용 = 높은 점수)"""
        # Normalize cost (inverse, 0-1)
        max_cost = max(m.cost_per_1k_input for m in self.models)
        if max_cost == 0:
            return 1.0
        cost_score = 1.0 - (model.cost_per_1k_input / max_cost)

        # Weight: 90% cost, 10% quality
        return cost_score * 0.9 + model.quality_score * 0.1

    def _score_quality_optimized(self, model: ModelInfo, request: RequestCharacteristics) -> float:
        """품질 최적화 점수 (높은 품질 = 높은 점수)"""
        # Weight: 90% quality, 10% cost efficiency
        max_cost = max(m.cost_per_1k_input for m in self.models)
        cost_score = 1.0 - (model.cost_per_1k_input / max_cost) if max_cost > 0 else 1.0

        return model.quality_score * 0.9 + cost_score * 0.1

    def _score_balanced(self, model: ModelInfo, request: RequestCharacteristics) -> float:
        """균형 점수 (비용과 품질의 균형)"""
        # Normalize cost
        max_cost = max(m.cost_per_1k_input for m in self.models)
        cost_score = 1.0 - (model.cost_per_1k_input / max_cost) if max_cost > 0 else 1.0

        # Normalize latency
        latency_score = 1.0 - model.latency_score

        # Weight: 40% quality, 40% cost, 20% latency
        return model.quality_score * 0.4 + cost_score * 0.4 + latency_score * 0.2

    def _score_complexity_based(self, model: ModelInfo, request: RequestCharacteristics) -> float:
        """복잡도 기반 점수"""
        complexity = request.complexity_score or 0.5

        # Match model quality to task complexity
        # Low complexity -> cheap model, High complexity -> powerful model
        quality_match = 1.0 - abs(model.quality_score - complexity)

        # Normalize cost
        max_cost = max(m.cost_per_1k_input for m in self.models)
        cost_score = 1.0 - (model.cost_per_1k_input / max_cost) if max_cost > 0 else 1.0

        # Weight: 60% quality match, 40% cost
        return quality_match * 0.6 + cost_score * 0.4

    def _score_capability_match(self, model: ModelInfo, request: RequestCharacteristics) -> float:
        """기능 매칭 점수 (필요한 기능만 지원)"""
        # Count required capabilities
        required_caps = sum(
            [
                request.requires_vision,
                request.requires_function_calling,
                request.requires_json_mode,
            ]
        )

        # Count model capabilities
        model_caps = sum(
            [
                model.supports_vision,
                model.supports_function_calling,
                model.supports_json_mode,
            ]
        )

        # Penalize over-qualified models (more expensive than needed)
        if model_caps > required_caps:
            over_qualified_penalty = (model_caps - required_caps) * 0.2
        else:
            over_qualified_penalty = 0.0

        # Normalize cost
        max_cost = max(m.cost_per_1k_input for m in self.models)
        cost_score = 1.0 - (model.cost_per_1k_input / max_cost) if max_cost > 0 else 1.0

        return cost_score - over_qualified_penalty

    def _estimate_cost(self, model: ModelInfo, request: RequestCharacteristics) -> float:
        """
        예상 비용 계산

        Args:
            model: 모델
            request: 요청 특성

        Returns:
            예상 비용 ($)
        """
        # Estimate tokens (simple approximation: 1 token ≈ 4 chars)
        input_tokens = request.prompt_length / 4
        output_tokens = 500  # Assume average 500 token output

        input_cost = (input_tokens / 1000) * model.cost_per_1k_input
        output_cost = (output_tokens / 1000) * model.cost_per_1k_output

        return input_cost + output_cost

    def _generate_reason(
        self,
        model: ModelInfo,
        request: RequestCharacteristics,
        score: float,
    ) -> str:
        """선택 이유 생성"""
        reasons = []

        if self.strategy == RoutingStrategy.COST_OPTIMIZED:
            reasons.append(f"Cost-optimized ({model.cost_per_1k_input:.4f}$/1k)")
        elif self.strategy == RoutingStrategy.QUALITY_OPTIMIZED:
            reasons.append(f"Quality-optimized (score: {model.quality_score:.2f})")
        elif self.strategy == RoutingStrategy.BALANCED:
            reasons.append("Balanced cost and quality")
        elif self.strategy == RoutingStrategy.COMPLEXITY_BASED:
            complexity = request.complexity_score or 0.5
            reasons.append(f"Complexity match ({complexity:.2f})")
        elif self.strategy == RoutingStrategy.CAPABILITY_MATCH:
            reasons.append("Capability match (minimal over-qualification)")

        # Add capability reasons
        if request.requires_vision and model.supports_vision:
            reasons.append("vision support")
        if request.requires_function_calling and model.supports_function_calling:
            reasons.append("function calling")

        return ", ".join(reasons)

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

        return successful / total

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


# Pre-configured model pools
DEFAULT_MODELS = [
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


def create_default_router(strategy: RoutingStrategy = RoutingStrategy.BALANCED) -> ModelRouter:
    """
    기본 모델 풀로 라우터 생성

    Args:
        strategy: 라우팅 전략

    Returns:
        ModelRouter 인스턴스
    """
    router = ModelRouter(strategy=strategy)
    router.register_models(DEFAULT_MODELS)
    return router
