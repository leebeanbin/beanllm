"""
OptimizerHandler 테스트
"""

from unittest.mock import AsyncMock

import pytest

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
from beanllm.handler.advanced.optimizer_handler import OptimizerHandler
from beanllm.service.optimizer_service import IOptimizerService


def _make_benchmark_response() -> BenchmarkResponse:
    return BenchmarkResponse(
        benchmark_id="bench-1",
        num_queries=10,
        avg_latency=0.5,
        throughput=2.0,
    )


def _make_optimize_response() -> OptimizeResponse:
    return OptimizeResponse(
        optimization_id="opt-1",
        system_id="sys-1",
        optimal_parameters={"top_k": 5},
        improvement_metrics={"quality": 0.1},
        num_trials=10,
    )


def _make_profile_response() -> ProfileResponse:
    return ProfileResponse(
        profile_id="prof-1",
        system_id="sys-1",
        duration=1.5,
        component_breakdown={"embedding": {"latency": 0.3}},
        total_latency=1.5,
        total_cost=0.01,
        bottlenecks=[],
        cost_breakdown={"embedding": 0.005},
        bottleneck="embedding",
    )


def _make_ab_response() -> ABTestResponse:
    return ABTestResponse(
        test_id="ab-1",
        config_a_id="cfg-a",
        config_b_id="cfg-b",
        num_queries=10,
        results_a={"quality": 0.8},
        results_b={"quality": 0.85},
        statistical_significance={"p_value": 0.03, "significant": True},
        winner="variant_b",
    )


def _make_recommendation_response() -> RecommendationResponse:
    return RecommendationResponse(
        profile_id="prof-1",
        recommendations=[{"title": "Increase top_k", "priority": "high"}],
        estimated_improvements={"quality": 0.1},
        implementation_difficulty={"Increase top_k": "easy"},
        priority_order=["Increase top_k"],
    )


class TestOptimizerHandler:
    @pytest.fixture
    def mock_service(self) -> AsyncMock:
        service = AsyncMock(spec=IOptimizerService)
        service.benchmark.return_value = _make_benchmark_response()
        service.optimize.return_value = _make_optimize_response()
        service.profile.return_value = _make_profile_response()
        service.ab_test.return_value = _make_ab_response()
        service.get_recommendations.return_value = _make_recommendation_response()
        service.compare_configs.return_value = {"config_a": 0.8, "config_b": 0.85}
        return service

    @pytest.fixture
    def handler(self, mock_service: AsyncMock) -> OptimizerHandler:
        return OptimizerHandler(service=mock_service)

    async def test_handle_benchmark(self, handler: OptimizerHandler) -> None:
        request = BenchmarkRequest(system_id="sys-1", num_queries=10)
        result = await handler.handle_benchmark(request)
        assert isinstance(result, BenchmarkResponse)
        assert result.num_queries == 10

    async def test_handle_benchmark_zero_queries_raises(self, handler: OptimizerHandler) -> None:
        request = BenchmarkRequest(system_id="sys-1", num_queries=0, queries=[])
        with pytest.raises(Exception):
            await handler.handle_benchmark(request)

    async def test_handle_benchmark_invalid_query_type_raises(
        self, handler: OptimizerHandler
    ) -> None:
        request = BenchmarkRequest(
            system_id="sys-1",
            num_queries=10,
            query_types=["invalid_query_type_xyz"],
        )
        with pytest.raises(Exception):
            await handler.handle_benchmark(request)

    async def test_handle_optimize(self, handler: OptimizerHandler) -> None:
        request = OptimizeRequest(
            system_id="sys-1",
            parameter_space={"top_k": {"type": "integer", "low": 1, "high": 20}},
            parameters=[{"name": "top_k", "type": "integer", "low": 1, "high": 20}],
            optimization_method="bayesian",
        )
        result = await handler.handle_optimize(request)
        assert isinstance(result, OptimizeResponse)
        assert "top_k" in result.optimal_parameters

    async def test_handle_optimize_no_parameters_raises(self, handler: OptimizerHandler) -> None:
        request = OptimizeRequest(
            system_id="sys-1",
            parameter_space={},
            parameters=[],
        )
        with pytest.raises(Exception):
            await handler.handle_optimize(request)

    async def test_handle_optimize_invalid_method_raises(self, handler: OptimizerHandler) -> None:
        request = OptimizeRequest(
            system_id="sys-1",
            parameter_space={},
            parameters=[{"name": "top_k", "type": "integer", "low": 1, "high": 20}],
            optimization_method="invalid_method_xyz",
        )
        with pytest.raises(Exception):
            await handler.handle_optimize(request)

    async def test_handle_profile(self, handler: OptimizerHandler) -> None:
        request = ProfileRequest(
            system_id="sys-1",
            components=["embedding", "retrieval"],
        )
        result = await handler.handle_profile(request)
        assert isinstance(result, ProfileResponse)
        assert result.bottleneck == "embedding"

    async def test_handle_profile_invalid_component_raises(self, handler: OptimizerHandler) -> None:
        request = ProfileRequest(
            system_id="sys-1",
            components=["invalid_component_xyz"],
        )
        with pytest.raises(Exception):
            await handler.handle_profile(request)

    async def test_handle_ab_test(self, handler: OptimizerHandler) -> None:
        request = ABTestRequest(
            config_a_id="cfg-a",
            config_b_id="cfg-b",
            test_queries=["What is AI?"],
            variant_a_name="baseline",
            variant_b_name="optimized",
        )
        result = await handler.handle_ab_test(request)
        assert isinstance(result, ABTestResponse)
        assert result.winner == "variant_b"

    async def test_handle_ab_test_no_variant_name_raises(self, handler: OptimizerHandler) -> None:
        request = ABTestRequest(
            config_a_id="cfg-a",
            config_b_id="cfg-b",
            test_queries=["query"],
            variant_a_name="",
        )
        with pytest.raises(Exception):
            await handler.handle_ab_test(request)

    async def test_handle_get_recommendations(self, handler: OptimizerHandler) -> None:
        result = await handler.handle_get_recommendations("prof-1")
        assert isinstance(result, RecommendationResponse)
        assert len(result.recommendations) > 0

    async def test_handle_get_recommendations_no_id_raises(self, handler: OptimizerHandler) -> None:
        with pytest.raises(Exception):
            await handler.handle_get_recommendations("")

    async def test_handle_compare_configs(self, handler: OptimizerHandler) -> None:
        result = await handler.handle_compare_configs(["cfg-a", "cfg-b"])
        assert isinstance(result, dict)

    async def test_handle_compare_configs_too_few_raises(self, handler: OptimizerHandler) -> None:
        with pytest.raises(Exception):
            await handler.handle_compare_configs(["only-one"])
