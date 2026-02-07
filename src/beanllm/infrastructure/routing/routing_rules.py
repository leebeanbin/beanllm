"""
Routing Rules - Extensible Rules for Model Selection

책임:
- 커스텀 라우팅 규칙 정의
- 규칙 기반 필터링 및 점수 계산

SOLID:
- SRP: 라우팅 규칙만 담당
- OCP: 새로운 규칙 쉽게 추가
"""

from abc import ABC, abstractmethod
from typing import List

from .model_router import ModelInfo, RequestCharacteristics


class RoutingRule(ABC):
    """라우팅 규칙 인터페이스"""

    @abstractmethod
    def evaluate(self, model: ModelInfo, request: RequestCharacteristics) -> float:
        """
        모델 평가

        Args:
            model: 모델 정보
            request: 요청 특성

        Returns:
            점수 (0-1)
        """
        pass


class ComplexityRule(RoutingRule):
    """
    복잡도 기반 규칙

    작업 복잡도와 모델 품질을 매칭
    """

    def evaluate(self, model: ModelInfo, request: RequestCharacteristics) -> float:
        complexity = request.complexity_score or 0.5

        # Perfect match gets 1.0, large mismatch gets 0.0
        quality_match = 1.0 - abs(model.quality_score - complexity)

        return quality_match


class CostRule(RoutingRule):
    """
    비용 기반 규칙

    비용 대비 품질 최적화
    """

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def evaluate(self, model: ModelInfo, request: RequestCharacteristics) -> float:
        # Avoid division by zero
        if model.cost_per_1k_input == 0:
            return 1.0

        # Quality per dollar spent
        value_score = model.quality_score / (model.cost_per_1k_input * 1000)

        # Normalize (assuming max value is ~100)
        normalized = min(value_score / 100, 1.0)

        return normalized * self.weight


class CapabilityRule(RoutingRule):
    """
    기능 요구사항 규칙

    필요한 기능을 정확히 갖춘 모델 선호
    """

    def evaluate(self, model: ModelInfo, request: RequestCharacteristics) -> float:
        score = 1.0

        # Check vision
        if request.requires_vision:
            if not model.supports_vision:
                return 0.0  # Hard requirement
        elif model.supports_vision:
            score -= 0.1  # Slight penalty for over-qualification

        # Check function calling
        if request.requires_function_calling:
            if not model.supports_function_calling:
                return 0.0
        elif model.supports_function_calling:
            score -= 0.1

        # Check JSON mode
        if request.requires_json_mode:
            if not model.supports_json_mode:
                return 0.0
        elif model.supports_json_mode:
            score -= 0.1

        return max(score, 0.0)


class LatencyRule(RoutingRule):
    """
    지연 시간 규칙

    빠른 응답 선호
    """

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def evaluate(self, model: ModelInfo, request: RequestCharacteristics) -> float:
        # Lower latency score = faster model = higher routing score
        return (1.0 - model.latency_score) * self.weight


class ReliabilityRule(RoutingRule):
    """
    신뢰성 규칙

    안정적인 모델 선호
    """

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def evaluate(self, model: ModelInfo, request: RequestCharacteristics) -> float:
        return model.reliability_score * self.weight


class CompositeRule(RoutingRule):
    """
    복합 규칙

    여러 규칙을 조합하여 사용
    """

    def __init__(self, rules: List[tuple]):  # [(rule, weight), ...]
        self.rules = rules

    def evaluate(self, model: ModelInfo, request: RequestCharacteristics) -> float:
        total_weight = sum(weight for _, weight in self.rules)
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(rule.evaluate(model, request) * weight for rule, weight in self.rules)

        return float(weighted_sum / total_weight)
