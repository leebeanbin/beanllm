"""
OptimizerHandler - Auto-Optimizer Handler
SOLID 원칙:
- SRP: 검증 및 에러 처리만 담당
- DIP: 인터페이스에 의존
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

from beanllm.decorators.error_handler import handle_errors
from beanllm.dto.request.advanced.optimizer_request import (
    ABTestRequest,
    BenchmarkRequest,
    OptimizeRequest,
    ProfileRequest,
)
from beanllm.dto.response.advanced.optimizer_response import (
    ABTestResponse,
    BenchmarkResponse,
    OptimizeResponse,
    ProfileResponse,
    RecommendationResponse,
)
from beanllm.handler.base_handler import BaseHandler
from beanllm.utils.logging import get_logger

if TYPE_CHECKING:
    from beanllm.service.optimizer_service import IOptimizerService

logger = get_logger(__name__)

# 유효한 상수 (Magic Number 방지)
_VALID_QUERY_TYPES = frozenset({"simple", "complex", "edge_case", "multi_hop", "aggregation"})
_VALID_OPTIMIZATION_METHODS = frozenset({"bayesian", "grid", "random", "genetic"})
_VALID_PARAM_TYPES = frozenset({"integer", "float", "categorical", "boolean"})
_NUMERIC_PARAM_TYPES = frozenset({"integer", "float"})
_VALID_PROFILE_COMPONENTS = frozenset(
    {
        "embedding",
        "retrieval",
        "reranking",
        "generation",
        "preprocessing",
        "postprocessing",
        "total",
    }
)
_MIN_OBJECTIVES_FOR_MULTI = 2
_MIN_CONFIGS_FOR_COMPARISON = 2


class OptimizerHandler(BaseHandler["IOptimizerService"]):
    """
    Auto-Optimizer Handler

    책임:
    - 요청 검증
    - 에러 처리 (@handle_errors 데코레이터)
    - 응답 포매팅
    """

    def __init__(self, service: "IOptimizerService") -> None:
        """
        Args:
            service: Optimizer 서비스
        """
        super().__init__(service)

    @handle_errors(error_message="Failed to run benchmark")
    async def handle_benchmark(self, request: BenchmarkRequest) -> BenchmarkResponse:
        """
        벤치마크 실행

        Args:
            request: BenchmarkRequest

        Returns:
            BenchmarkResponse

        Raises:
            ValueError: 검증 실패
        """
        if not request.queries and not request.num_queries:
            raise ValueError("Either queries or num_queries must be provided")

        if request.num_queries and request.num_queries <= 0:
            raise ValueError("num_queries must be positive")

        if request.query_types:
            for qt in request.query_types:
                if qt.lower() not in _VALID_QUERY_TYPES:
                    raise ValueError(f"Invalid query type: {qt}")

        return await self._service.benchmark(request)

    @handle_errors(error_message="Failed to optimize parameters")
    async def handle_optimize(self, request: OptimizeRequest) -> OptimizeResponse:
        """
        파라미터 최적화

        Args:
            request: OptimizeRequest

        Returns:
            OptimizeResponse

        Raises:
            ValueError: 검증 실패
        """
        if not request.parameters:
            raise ValueError("parameters are required")

        if request.n_trials and request.n_trials <= 0:
            raise ValueError("n_trials must be positive")

        method = request.method or request.optimization_method
        if method.lower() not in _VALID_OPTIMIZATION_METHODS:
            raise ValueError(
                f"Invalid optimization method: {method}. "
                f"Must be one of {sorted(_VALID_OPTIMIZATION_METHODS)}"
            )

        self._validate_parameters(request.parameters)
        self._validate_multi_objective(request)

        return await self._service.optimize(request)

    @handle_errors(error_message="Failed to profile system")
    async def handle_profile(self, request: ProfileRequest) -> ProfileResponse:
        """
        시스템 프로파일링

        Args:
            request: ProfileRequest

        Returns:
            ProfileResponse

        Raises:
            ValueError: 검증 실패
        """
        if request.components:
            for component in request.components:
                if component.lower() not in _VALID_PROFILE_COMPONENTS:
                    raise ValueError(f"Invalid component: {component}")

        return await self._service.profile(request)

    @handle_errors(error_message="Failed to run A/B test")
    async def handle_ab_test(self, request: ABTestRequest) -> ABTestResponse:
        """
        A/B 테스트 실행

        Args:
            request: ABTestRequest

        Returns:
            ABTestResponse

        Raises:
            ValueError: 검증 실패
        """
        if not request.variant_a_name:
            raise ValueError("variant_a_name is required")

        if not request.variant_b_name:
            raise ValueError("variant_b_name is required")

        if request.num_queries and request.num_queries <= 0:
            raise ValueError("num_queries must be positive")

        if request.confidence_level:
            if not (0 < request.confidence_level < 1):
                raise ValueError("confidence_level must be between 0 and 1")

        return await self._service.ab_test(request)

    @handle_errors(error_message="Failed to get recommendations")
    async def handle_get_recommendations(self, profile_id: str) -> RecommendationResponse:
        """
        권장사항 조회

        Args:
            profile_id: Profile ID

        Returns:
            RecommendationResponse

        Raises:
            ValueError: 검증 실패
        """
        if not profile_id:
            raise ValueError("profile_id is required")

        return await self._service.get_recommendations(profile_id)

    @handle_errors(error_message="Failed to compare configs")
    async def handle_compare_configs(self, config_ids: List[str]) -> Dict[str, Any]:
        """
        설정 비교

        Args:
            config_ids: List of config IDs

        Returns:
            Dict with comparison results

        Raises:
            ValueError: 검증 실패
        """
        if not config_ids:
            raise ValueError("config_ids is required")

        if len(config_ids) < _MIN_CONFIGS_FOR_COMPARISON:
            raise ValueError(
                f"At least {_MIN_CONFIGS_FOR_COMPARISON} config IDs required for comparison"
            )

        return await self._service.compare_configs(config_ids)

    # ===== Private validation helpers =====

    @staticmethod
    def _validate_parameters(parameters: List[Dict[str, Any]]) -> None:
        """파라미터 목록 검증"""
        for param in parameters:
            if "name" not in param:
                raise ValueError("Parameter must have 'name' field")

            if "type" not in param:
                raise ValueError(f"Parameter {param['name']} must have 'type' field")

            param_type = str(param["type"]).lower()
            if param_type not in _VALID_PARAM_TYPES:
                raise ValueError(f"Invalid parameter type: {param_type} for {param['name']}")

            if param_type in _NUMERIC_PARAM_TYPES:
                if "low" not in param or "high" not in param:
                    raise ValueError(f"Parameter {param['name']} must have 'low' and 'high' fields")
                low = float(str(param["low"]))
                high = float(str(param["high"]))
                if low >= high:
                    raise ValueError(f"Parameter {param['name']}: low must be less than high")

            elif param_type == "categorical":
                if "categories" not in param or not param["categories"]:
                    raise ValueError(
                        f"Parameter {param['name']} must have non-empty 'categories' field"
                    )

    @staticmethod
    def _validate_multi_objective(request: OptimizeRequest) -> None:
        """멀티 오브젝티브 검증"""
        if request.multi_objective:
            if not request.objectives or len(request.objectives) < _MIN_OBJECTIVES_FOR_MULTI:
                raise ValueError(
                    f"multi_objective requires at least {_MIN_OBJECTIVES_FOR_MULTI} objectives"
                )

            for obj in request.objectives:
                if "name" not in obj:
                    raise ValueError("Objective must have 'name' field")
