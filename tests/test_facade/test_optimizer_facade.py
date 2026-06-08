"""
Optimizer Facade 테스트
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from beanllm.dto.response.advanced.optimizer_response import (
    ABTestResponse,
    BenchmarkResponse,
    OptimizeResponse,
    ProfileResponse,
    RecommendationResponse,
)
from beanllm.facade.advanced.optimizer_facade import Optimizer
from beanllm.handler.advanced.optimizer_handler import OptimizerHandler


def _make_handler() -> MagicMock:
    handler = AsyncMock(spec=OptimizerHandler)
    handler.handle_benchmark.return_value = BenchmarkResponse(
        benchmark_id="bench-1",
        num_queries=50,
        avg_latency=0.4,
        throughput=2.5,
    )
    handler.handle_optimize.return_value = OptimizeResponse(
        optimization_id="opt-1",
        system_id="default",
        optimal_parameters={"top_k": 7},
        improvement_metrics={"quality": 0.15},
        num_trials=30,
    )
    handler.handle_profile.return_value = ProfileResponse(
        profile_id="prof-1",
        system_id="default",
        duration=2.0,
        component_breakdown={"embedding": {"latency": 0.5}},
        total_latency=2.0,
        total_cost=0.02,
        bottlenecks=[],
        cost_breakdown={"embedding": 0.01},
        bottleneck="embedding",
    )
    handler.handle_ab_test.return_value = ABTestResponse(
        test_id="ab-1",
        config_a_id="cfg-a",
        config_b_id="cfg-b",
        num_queries=20,
        results_a={"quality": 0.8},
        results_b={"quality": 0.88},
        statistical_significance={"p_value": 0.02, "significant": True},
        winner="variant_b",
    )
    handler.handle_get_recommendations.return_value = RecommendationResponse(
        profile_id="prof-1",
        recommendations=[{"title": "Use reranking", "priority": "high"}],
        estimated_improvements={"quality": 0.15},
        implementation_difficulty={"Use reranking": "medium"},
        priority_order=["Use reranking"],
    )
    handler.handle_compare_configs.return_value = {"cfg-a": 0.8, "cfg-b": 0.88}
    return handler


class TestOptimizerFacade:
    @pytest.fixture
    def optimizer(self) -> Optimizer:
        opt = object.__new__(Optimizer)
        opt._optimizer = _make_handler()
        return opt

    async def test_benchmark_with_num_queries(self, optimizer: Optimizer) -> None:
        result = await optimizer.benchmark(num_queries=50)
        assert isinstance(result, BenchmarkResponse)
        assert result.num_queries == 50
        assert result.avg_latency == 0.4

    async def test_benchmark_with_custom_queries(self, optimizer: Optimizer) -> None:
        result = await optimizer.benchmark(queries=["What is AI?", "How does RAG work?"])
        assert isinstance(result, BenchmarkResponse)

    async def test_benchmark_with_domain(self, optimizer: Optimizer) -> None:
        result = await optimizer.benchmark(num_queries=20, domain="healthcare")
        assert isinstance(result, BenchmarkResponse)

    async def test_optimize(self, optimizer: Optimizer) -> None:
        parameters = [
            {"name": "top_k", "type": "integer", "low": 1, "high": 20},
            {"name": "threshold", "type": "float", "low": 0.0, "high": 1.0},
        ]
        result = await optimizer.optimize(parameters=parameters, method="bayesian", n_trials=30)
        assert isinstance(result, OptimizeResponse)
        assert "top_k" in result.optimal_parameters

    async def test_optimize_multi_objective(self, optimizer: Optimizer) -> None:
        parameters = [{"name": "top_k", "type": "integer", "low": 1, "high": 20}]
        objectives = [
            {"name": "quality", "maximize": True, "weight": 0.6},
            {"name": "latency", "maximize": False, "weight": 0.4},
        ]
        result = await optimizer.optimize(
            parameters=parameters,
            multi_objective=True,
            objectives=objectives,
        )
        assert isinstance(result, OptimizeResponse)

    async def test_profile(self, optimizer: Optimizer) -> None:
        result = await optimizer.profile(components=["embedding", "retrieval"])
        assert isinstance(result, ProfileResponse)
        assert result.bottleneck == "embedding"

    async def test_ab_test(self, optimizer: Optimizer) -> None:
        result = await optimizer.ab_test(
            variant_a_name="baseline",
            variant_b_name="optimized",
            num_queries=20,
        )
        assert isinstance(result, ABTestResponse)
        assert result.winner == "variant_b"

    async def test_get_recommendations(self, optimizer: Optimizer) -> None:
        result = await optimizer.get_recommendations("prof-1")
        assert isinstance(result, RecommendationResponse)
        assert len(result.recommendations) > 0

    async def test_compare_configs(self, optimizer: Optimizer) -> None:
        result = await optimizer.compare_configs(["cfg-a", "cfg-b"])
        assert isinstance(result, dict)
        assert "cfg-a" in result

    async def test_optimizer_has_handler(self, optimizer: Optimizer) -> None:
        assert optimizer._optimizer is not None
