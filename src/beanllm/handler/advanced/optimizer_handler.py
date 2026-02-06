"""
OptimizerHandler - Auto-Optimizer Handler
SOLID 원칙:
- SRP: 검증 및 에러 처리만 담당
- DIP: 인터페이스에 의존
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

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
from beanllm.utils.logging import get_logger

if TYPE_CHECKING:
    from beanllm.service.optimizer_service import IOptimizerService

logger = get_logger(__name__)


class OptimizerHandler:
    """
    Auto-Optimizer Handler

    책임:
    - 요청 검증
    - 에러 처리
    - 응답 포매팅

    SOLID:
    - SRP: 검증 및 에러 처리만
    - DIP: 인터페이스에 의존
    """

    def __init__(self, service: "IOptimizerService") -> None:
        """
        Args:
            service: Optimizer 서비스
        """
        self._service = service

    async def handle_benchmark(self, request: BenchmarkRequest) -> BenchmarkResponse:
        """
        벤치마크 실행

        Args:
            request: BenchmarkRequest

        Returns:
            BenchmarkResponse

        Raises:
            ValueError: 검증 실패
            RuntimeError: 실행 실패
        """
        # Validation
        if not request.queries and not request.num_queries:
            raise ValueError("Either queries or num_queries must be provided")

        if request.num_queries and request.num_queries <= 0:
            raise ValueError("num_queries must be positive")

        if request.query_types:
            valid_types = ["simple", "complex", "edge_case", "multi_hop", "aggregation"]
            for qt in request.query_types:
                if qt.lower() not in valid_types:
                    raise ValueError(f"Invalid query type: {qt}")

        # Service call with error handling
        try:
            response = await self._service.benchmark(request)
            return response

        except ValueError as e:
            logger.error(f"Validation error in benchmark: {e}")
            raise

        except Exception as e:
            logger.error(f"Error running benchmark: {e}")
            raise RuntimeError(f"Failed to run benchmark: {e}") from e

    async def handle_optimize(self, request: OptimizeRequest) -> OptimizeResponse:
        """
        파라미터 최적화

        Args:
            request: OptimizeRequest

        Returns:
            OptimizeResponse

        Raises:
            ValueError: 검증 실패
            RuntimeError: 실행 실패
        """
        # Validation
        if not request.parameters:
            raise ValueError("parameters are required")

        if request.n_trials and request.n_trials <= 0:
            raise ValueError("n_trials must be positive")

        # Validate method
        valid_methods = ["bayesian", "grid", "random", "genetic"]
        if request.method.lower() not in valid_methods:
            raise ValueError(
                f"Invalid optimization method: {request.method}. " f"Must be one of {valid_methods}"
            )

        # Validate parameters
        for param in request.parameters:
            if "name" not in param:
                raise ValueError("Parameter must have 'name' field")

            if "type" not in param:
                raise ValueError(f"Parameter {param['name']} must have 'type' field")

            param_type = param["type"].lower()
            if param_type not in ["integer", "float", "categorical", "boolean"]:
                raise ValueError(f"Invalid parameter type: {param_type} for {param['name']}")

            # Type-specific validation
            if param_type in ["integer", "float"]:
                if "low" not in param or "high" not in param:
                    raise ValueError(f"Parameter {param['name']} must have 'low' and 'high' fields")
                if param["low"] >= param["high"]:
                    raise ValueError(f"Parameter {param['name']}: low must be less than high")

            elif param_type == "categorical":
                if "categories" not in param or not param["categories"]:
                    raise ValueError(
                        f"Parameter {param['name']} must have non-empty 'categories' field"
                    )

        # Validate multi-objective
        if request.multi_objective:
            if not request.objectives or len(request.objectives) < 2:
                raise ValueError("multi_objective requires at least 2 objectives")

            for obj in request.objectives:
                if "name" not in obj:
                    raise ValueError("Objective must have 'name' field")

        # Service call with error handling
        try:
            response = await self._service.optimize(request)
            return response

        except ValueError as e:
            logger.error(f"Validation error in optimize: {e}")
            raise

        except Exception as e:
            logger.error(f"Error optimizing parameters: {e}")
            raise RuntimeError(f"Failed to optimize parameters: {e}") from e

    async def handle_profile(self, request: ProfileRequest) -> ProfileResponse:
        """
        시스템 프로파일링

        Args:
            request: ProfileRequest

        Returns:
            ProfileResponse

        Raises:
            ValueError: 검증 실패
            RuntimeError: 실행 실패
        """
        # Validation
        if request.components:
            valid_components = [
                "embedding",
                "retrieval",
                "reranking",
                "generation",
                "preprocessing",
                "postprocessing",
                "total",
            ]
            for component in request.components:
                if component.lower() not in valid_components:
                    raise ValueError(f"Invalid component: {component}")

        # Service call with error handling
        try:
            response = await self._service.profile(request)
            return response

        except ValueError as e:
            logger.error(f"Validation error in profile: {e}")
            raise

        except Exception as e:
            logger.error(f"Error profiling system: {e}")
            raise RuntimeError(f"Failed to profile system: {e}") from e

    async def handle_ab_test(self, request: ABTestRequest) -> ABTestResponse:
        """
        A/B 테스트 실행

        Args:
            request: ABTestRequest

        Returns:
            ABTestResponse

        Raises:
            ValueError: 검증 실패
            RuntimeError: 실행 실패
        """
        # Validation
        if not request.variant_a_name:
            raise ValueError("variant_a_name is required")

        if not request.variant_b_name:
            raise ValueError("variant_b_name is required")

        if request.num_queries and request.num_queries <= 0:
            raise ValueError("num_queries must be positive")

        if request.confidence_level:
            if not (0 < request.confidence_level < 1):
                raise ValueError("confidence_level must be between 0 and 1")

        # Service call with error handling
        try:
            response = await self._service.ab_test(request)
            return response

        except ValueError as e:
            logger.error(f"Validation error in ab_test: {e}")
            raise

        except Exception as e:
            logger.error(f"Error running A/B test: {e}")
            raise RuntimeError(f"Failed to run A/B test: {e}") from e

    async def handle_get_recommendations(self, profile_id: str) -> RecommendationResponse:
        """
        권장사항 조회

        Args:
            profile_id: Profile ID

        Returns:
            RecommendationResponse

        Raises:
            ValueError: 검증 실패
            RuntimeError: 실행 실패
        """
        # Validation
        if not profile_id:
            raise ValueError("profile_id is required")

        # Service call with error handling
        try:
            response = await self._service.get_recommendations(profile_id)
            return response

        except ValueError as e:
            logger.error(f"Validation error in get_recommendations: {e}")
            raise

        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            raise RuntimeError(f"Failed to get recommendations: {e}") from e

    async def handle_compare_configs(self, config_ids: List[str]) -> Dict[str, Any]:
        """
        설정 비교

        Args:
            config_ids: List of config IDs

        Returns:
            Dict with comparison results

        Raises:
            ValueError: 검증 실패
            RuntimeError: 실행 실패
        """
        # Validation
        if not config_ids:
            raise ValueError("config_ids is required")

        if len(config_ids) < 2:
            raise ValueError("At least 2 config IDs required for comparison")

        # Service call with error handling
        try:
            response = await self._service.compare_configs(config_ids)
            return response

        except ValueError as e:
            logger.error(f"Validation error in compare_configs: {e}")
            raise

        except Exception as e:
            logger.error(f"Error comparing configs: {e}")
            raise RuntimeError(f"Failed to compare configs: {e}") from e
