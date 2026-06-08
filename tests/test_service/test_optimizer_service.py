"""
OptimizerService 테스트 - 벤치마킹, 최적화, 프로파일링, A/B 테스트
"""

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
)
from beanllm.service.impl.advanced.optimizer_service_impl import OptimizerServiceImpl


@pytest.fixture
def service() -> OptimizerServiceImpl:
    return OptimizerServiceImpl()


class TestBenchmark:
    @pytest.mark.asyncio
    async def test_benchmark_basic(self, service: OptimizerServiceImpl) -> None:
        request = BenchmarkRequest(
            system_id="sys-1",
            num_queries=5,
            synthetic=True,
        )
        response = await service.benchmark(request)
        assert isinstance(response, BenchmarkResponse)
        assert response.benchmark_id is not None
        assert response.system_id == "sys-1"

    @pytest.mark.asyncio
    async def test_benchmark_with_test_queries(self, service: OptimizerServiceImpl) -> None:
        request = BenchmarkRequest(
            system_id="sys-2",
            num_queries=3,
            synthetic=False,
            test_queries=["query1", "query2", "query3"],
        )
        response = await service.benchmark(request)
        assert isinstance(response, BenchmarkResponse)
        assert response.num_queries == 3

    @pytest.mark.asyncio
    async def test_benchmark_stores_result(self, service: OptimizerServiceImpl) -> None:
        request = BenchmarkRequest(
            system_id="sys-3",
            num_queries=2,
            synthetic=True,
        )
        response = await service.benchmark(request)
        assert response.benchmark_id in service._benchmarks

    @pytest.mark.asyncio
    async def test_benchmark_returns_metrics(self, service: OptimizerServiceImpl) -> None:
        request = BenchmarkRequest(
            system_id="sys-4",
            num_queries=5,
            synthetic=True,
        )
        response = await service.benchmark(request)
        assert response.avg_latency >= 0
        assert response.throughput >= 0
        assert "avg_latency" in response.baseline_metrics


class TestOptimize:
    @pytest.mark.asyncio
    async def test_optimize_basic(self, service: OptimizerServiceImpl) -> None:
        request = OptimizeRequest(
            system_id="sys-opt-1",
            parameter_space={"chunk_size": {"type": "integer", "low": 100, "high": 1000}},
            parameters=[{"name": "chunk_size", "type": "integer", "low": 100, "high": 1000}],
            optimization_method="random",
            max_trials=5,
        )
        response = await service.optimize(request)
        assert isinstance(response, OptimizeResponse)
        assert response.optimization_id is not None
        assert response.system_id == "sys-opt-1"

    @pytest.mark.asyncio
    async def test_optimize_float_parameter(self, service: OptimizerServiceImpl) -> None:
        request = OptimizeRequest(
            system_id="sys-opt-2",
            parameter_space={"temperature": {"type": "float", "low": 0.0, "high": 1.0}},
            parameters=[{"name": "temperature", "type": "float", "low": 0.0, "high": 1.0}],
            optimization_method="bayesian",
            max_trials=3,
        )
        response = await service.optimize(request)
        assert response.optimization_id is not None

    @pytest.mark.asyncio
    async def test_optimize_stores_result(self, service: OptimizerServiceImpl) -> None:
        request = OptimizeRequest(
            system_id="sys-opt-3",
            parameter_space={},
            parameters=[{"name": "k", "type": "integer", "low": 1, "high": 10}],
            max_trials=2,
        )
        response = await service.optimize(request)
        assert response.optimization_id in service._optimizations

    @pytest.mark.asyncio
    async def test_optimize_empty_parameters(self, service: OptimizerServiceImpl) -> None:
        request = OptimizeRequest(
            system_id="sys-opt-4",
            parameter_space={},
            parameters=[],
            max_trials=2,
        )
        response = await service.optimize(request)
        assert isinstance(response, OptimizeResponse)


class TestProfile:
    @pytest.mark.asyncio
    async def test_profile_basic(self, service: OptimizerServiceImpl) -> None:
        request = ProfileRequest(
            system_id="sys-prof-1",
            components=["retrieval", "generation"],
        )
        response = await service.profile(request)
        assert isinstance(response, ProfileResponse)
        assert response.profile_id is not None
        assert response.system_id == "sys-prof-1"

    @pytest.mark.asyncio
    async def test_profile_stores_result(self, service: OptimizerServiceImpl) -> None:
        request = ProfileRequest(
            system_id="sys-prof-2",
            components=["retrieval"],
        )
        response = await service.profile(request)
        assert response.profile_id in service._profiles

    @pytest.mark.asyncio
    async def test_profile_returns_component_data(self, service: OptimizerServiceImpl) -> None:
        request = ProfileRequest(
            system_id="sys-prof-3",
            components=["retrieval", "reranking", "generation"],
        )
        response = await service.profile(request)
        assert response.total_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_get_recommendations_after_profile(self, service: OptimizerServiceImpl) -> None:
        profile_req = ProfileRequest(
            system_id="sys-rec-1",
            components=["retrieval", "generation"],
        )
        profile_resp = await service.profile(profile_req)

        from beanllm.dto.response.advanced.optimizer_response import RecommendationResponse

        rec_resp = await service.get_recommendations(profile_resp.profile_id)
        assert isinstance(rec_resp, RecommendationResponse)


class TestABTest:
    @pytest.mark.asyncio
    async def test_ab_test_basic(self, service: OptimizerServiceImpl) -> None:
        request = ABTestRequest(
            config_a_id="config-a-1",
            config_b_id="config-b-1",
            test_queries=["q1", "q2", "q3"],
            num_queries=3,
        )
        response = await service.ab_test(request)
        assert isinstance(response, ABTestResponse)
        assert response.test_id is not None

    @pytest.mark.asyncio
    async def test_ab_test_has_winner(self, service: OptimizerServiceImpl) -> None:
        request = ABTestRequest(
            config_a_id="config-a-2",
            config_b_id="config-b-2",
            test_queries=["test query 1", "test query 2"],
            num_queries=2,
        )
        response = await service.ab_test(request)
        assert isinstance(response.winner, str) or response.winner is None

    @pytest.mark.asyncio
    async def test_ab_test_stores_result(self, service: OptimizerServiceImpl) -> None:
        request = ABTestRequest(
            config_a_id="config-a-3",
            config_b_id="config-b-3",
            test_queries=["q1", "q2"],
            num_queries=2,
        )
        response = await service.ab_test(request)
        assert response.test_id in service._ab_tests


class TestCompareConfigs:
    @pytest.mark.asyncio
    async def test_compare_nonexistent_configs(self, service: OptimizerServiceImpl) -> None:
        result = await service.compare_configs(["nonexistent-1", "nonexistent-2"])
        assert "configs" in result
        assert "summary" in result
        assert result["summary"]["total_configs"] == 2
        assert result["summary"]["found"] == 0

    @pytest.mark.asyncio
    async def test_compare_optimization_configs(self, service: OptimizerServiceImpl) -> None:
        opt_req = OptimizeRequest(
            system_id="sys-cmp",
            parameter_space={},
            parameters=[{"name": "k", "type": "integer", "low": 1, "high": 5}],
            max_trials=2,
        )
        opt_resp = await service.optimize(opt_req)
        result = await service.compare_configs([opt_resp.optimization_id])
        assert opt_resp.optimization_id in result["configs"]
        assert result["configs"][opt_resp.optimization_id]["type"] == "optimization"

    @pytest.mark.asyncio
    async def test_compare_mixed_configs(self, service: OptimizerServiceImpl) -> None:
        opt_req = OptimizeRequest(
            system_id="sys-mix",
            parameter_space={},
            parameters=[{"name": "k", "type": "integer", "low": 1, "high": 3}],
            max_trials=1,
        )
        opt_resp = await service.optimize(opt_req)

        result = await service.compare_configs([opt_resp.optimization_id, "unknown-id"])
        assert result["summary"]["total_configs"] == 2
        assert result["summary"]["found"] == 1
