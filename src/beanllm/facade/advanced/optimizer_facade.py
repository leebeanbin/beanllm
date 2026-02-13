"""
Optimizer Facade - User-friendly Auto-Optimizer API

Core API lives here; convenience methods in optimizer_convenience.py,
standalone functions in optimizer_standalone.py (re-exported below).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

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
from beanllm.facade.advanced.optimizer_convenience import OptimizerConvenienceMixin
from beanllm.handler.advanced.optimizer_handler import OptimizerHandler
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class Optimizer(OptimizerConvenienceMixin):
    """
    User-friendly Auto-Optimizer Facade

    Provides simple, intuitive methods for:
    - Benchmarking RAG systems
    - Optimizing parameters
    - Profiling performance
    - A/B testing
    - Getting recommendations

    Example:
        ```python
        from beanllm import Optimizer

        # Initialize
        optimizer = Optimizer()

        # Benchmark
        result = await optimizer.benchmark(
            num_queries=50,
            query_types=["simple", "complex"],
            domain="machine learning"
        )
        print(f"Avg latency: {result.avg_latency:.3f}s")
        print(f"Throughput: {result.throughput:.1f} q/s")

        # Optimize parameters
        result = await optimizer.optimize(
            parameters=[
                {"name": "top_k", "type": "integer", "low": 1, "high": 20},
                {"name": "threshold", "type": "float", "low": 0.0, "high": 1.0},
            ],
            method="bayesian",
            n_trials=30
        )
        print(f"Best params: {result.best_params}")

        # Profile system
        result = await optimizer.profile(
            components=["embedding", "retrieval", "generation"]
        )
        print(f"Bottleneck: {result.bottleneck}")
        print(f"Total cost: ${result.total_cost:.4f}")

        # A/B test
        result = await optimizer.ab_test(
            variant_a_name="Baseline",
            variant_b_name="Optimized",
            num_queries=100
        )
        print(f"Winner: {result.winner}")
        print(f"Lift: {result.lift:.1f}%")
        print(f"P-value: {result.p_value:.4f}")

        # Get recommendations
        recs = await optimizer.get_recommendations(profile_id="...")
        for rec in recs.recommendations[:5]:
            print(f"[{rec['priority']}] {rec['title']}")
        ```
    """

    def __init__(self, handler: Optional[OptimizerHandler] = None) -> None:
        """
        Initialize Optimizer facade

        Args:
            handler: Optional OptimizerHandler (for DI)
        """
        if handler is None:
            from beanllm.utils.core.di_container import get_container

            container = get_container()
            service_factory = container.get_service_factory()
            service = service_factory.create_optimizer_service()
            handler = OptimizerHandler(service)

        self._optimizer = handler
        logger.info("Optimizer facade initialized")

    # ===== Core Methods =====

    async def benchmark(
        self,
        num_queries: Optional[int] = None,
        queries: Optional[List[str]] = None,
        query_types: Optional[List[str]] = None,
        domain: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResponse:
        """
        Run benchmark with synthetic or provided queries

        Args:
            num_queries: Number of synthetic queries to generate (default: 50)
            queries: Optional list of custom queries
            query_types: Query types to generate ["simple", "complex", "edge_case",
                        "multi_hop", "aggregation"] (default: all types)
            domain: Domain for synthetic queries (e.g., "machine learning")
            metadata: Optional metadata

        Returns:
            BenchmarkResponse: Benchmark results with latency and quality metrics

        Raises:
            ValueError: If validation fails

        Example:
            ```python
            # Generate synthetic queries
            result = await optimizer.benchmark(
                num_queries=50,
                query_types=["simple", "complex"],
                domain="healthcare"
            )

            # Use custom queries
            result = await optimizer.benchmark(
                queries=["What is RAG?", "How does it work?"]
            )
            ```
        """
        request = BenchmarkRequest(
            system_id="default",
            num_queries=num_queries or 50,
            queries=queries,
            query_types=query_types,
            domain=domain,
        )

        response = await self._optimizer.handle_benchmark(request)
        logger.info(
            f"Benchmark completed: {response.num_queries} queries, "
            f"avg_latency={response.avg_latency:.3f}s"
        )
        return cast(BenchmarkResponse, response)

    async def optimize(
        self,
        parameters: List[Dict[str, Any]],
        method: str = "bayesian",
        n_trials: int = 30,
        multi_objective: bool = False,
        objectives: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OptimizeResponse:
        """
        Optimize parameters using selected algorithm

        Args:
            parameters: List of parameter definitions
                [{"name": "top_k", "type": "integer", "low": 1, "high": 20}, ...]
            method: Optimization method: "bayesian", "grid", "random", "genetic"
                (default: "bayesian")
            n_trials: Number of optimization trials (default: 30)
            multi_objective: Enable multi-objective optimization (default: False)
            objectives: List of objectives for multi-objective optimization
                [{"name": "quality", "maximize": True, "weight": 0.6}, ...]
            metadata: Optional metadata

        Returns:
            OptimizeResponse: Optimization results with best parameters

        Raises:
            ValueError: If validation fails

        Example:
            ```python
            # Single-objective optimization
            result = await optimizer.optimize(
                parameters=[
                    {"name": "top_k", "type": "integer", "low": 1, "high": 20},
                    {"name": "threshold", "type": "float", "low": 0.0, "high": 1.0},
                ],
                method="bayesian",
                n_trials=30
            )
            print(f"Best top_k: {result.best_params['top_k']}")

            # Multi-objective optimization
            result = await optimizer.optimize(
                parameters=[...],
                multi_objective=True,
                objectives=[
                    {"name": "quality", "maximize": True, "weight": 0.6},
                    {"name": "latency", "maximize": False, "weight": 0.3},
                    {"name": "cost", "maximize": False, "weight": 0.1},
                ],
                n_trials=50
            )
            ```
        """
        request = OptimizeRequest(
            system_id="default",
            parameter_space={},
            parameters=parameters,
            method=method,
            n_trials=n_trials,
            multi_objective=multi_objective,
            objectives=objectives,  # type: ignore[arg-type]
        )

        response = await self._optimizer.handle_optimize(request)
        logger.info(
            f"Optimization completed: best_score={response.best_score:.4f}, "
            f"n_trials={response.n_trials}"
        )
        return cast(OptimizeResponse, response)

    async def profile(
        self,
        components: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProfileResponse:
        """
        Profile system components

        Args:
            components: Components to profile ["embedding", "retrieval", "reranking",
                       "generation", "preprocessing", "postprocessing", "total"]
                       (default: all components)
            metadata: Optional metadata

        Returns:
            ProfileResponse: Profiling results with bottleneck analysis and recommendations

        Raises:
            ValueError: If validation fails

        Example:
            ```python
            result = await optimizer.profile(
                components=["embedding", "retrieval", "generation"]
            )

            print(f"Total duration: {result.total_duration_ms}ms")
            print(f"Bottleneck: {result.bottleneck}")
            print(f"Total cost: ${result.total_cost:.4f}")

            # Component breakdown
            for name, pct in result.breakdown.items():
                print(f"{name}: {pct:.1f}%")

            # Recommendations
            for rec in result.recommendations[:3]:
                print(f"[{rec['priority']}] {rec['title']}")
                print(f"  Action: {rec['action']}")
            ```
        """
        request = ProfileRequest(
            system_id="default",
            components=components or [],
        )

        response = await self._optimizer.handle_profile(request)
        logger.info(
            f"Profile completed: total_duration={response.total_duration_ms}ms, "
            f"bottleneck={response.bottleneck}"
        )
        return cast(ProfileResponse, response)

    async def ab_test(
        self,
        variant_a_name: str,
        variant_b_name: str,
        num_queries: int = 50,
        confidence_level: float = 0.95,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ABTestResponse:
        """
        Run A/B test

        Args:
            variant_a_name: Name of variant A (baseline)
            variant_b_name: Name of variant B (new version)
            num_queries: Number of test queries (default: 50)
            confidence_level: Statistical confidence level (default: 0.95)
            metadata: Optional metadata

        Returns:
            ABTestResponse: A/B test results with statistical significance

        Raises:
            ValueError: If validation fails

        Example:
            ```python
            result = await optimizer.ab_test(
                variant_a_name="Baseline",
                variant_b_name="Optimized",
                num_queries=100,
                confidence_level=0.95
            )

            print(f"Variant A mean: {result.variant_a_mean:.3f}")
            print(f"Variant B mean: {result.variant_b_mean:.3f}")
            print(f"Winner: {result.winner}")
            print(f"Lift: {result.lift:.1f}%")
            print(f"P-value: {result.p_value:.4f}")
            print(f"Significant: {result.is_significant}")
            ```
        """
        request = ABTestRequest(
            config_a_id=variant_a_name,
            config_b_id=variant_b_name,
            test_queries=[],
            variant_a_name=variant_a_name,
            variant_b_name=variant_b_name,
            num_queries=num_queries,
            confidence_level=confidence_level,
        )

        response = await self._optimizer.handle_ab_test(request)
        logger.info(
            f"A/B test completed: winner={response.winner}, "
            f"lift={response.lift:.1f}%, significant={response.is_significant}"
        )
        return cast(ABTestResponse, response)

    async def get_recommendations(
        self,
        profile_id: str,
    ) -> RecommendationResponse:
        """
        Get optimization recommendations for a profile

        Args:
            profile_id: Profile ID from profiling

        Returns:
            RecommendationResponse: Prioritized recommendations

        Raises:
            ValueError: If profile not found

        Example:
            ```python
            # First profile the system
            profile = await optimizer.profile()

            # Get recommendations
            recs = await optimizer.get_recommendations(profile.profile_id)

            # Filter by priority
            critical = [r for r in recs.recommendations if r['priority'] == 'critical']
            high = [r for r in recs.recommendations if r['priority'] == 'high']

            for rec in critical + high:
                print(f"[{rec['priority'].upper()}] {rec['title']}")
                print(f"  {rec['description']}")
                print(f"  Action: {rec['action']}")
                print(f"  Impact: {rec['expected_impact']}")
            ```
        """
        response = await self._optimizer.handle_get_recommendations(profile_id)
        logger.info(
            f"Retrieved {len(response.recommendations)} recommendations for profile {profile_id}"
        )
        return cast(RecommendationResponse, response)

    async def compare_configs(
        self,
        config_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Compare multiple configurations

        Args:
            config_ids: List of config IDs (optimization_id, profile_id, test_id)

        Returns:
            Dict with comparison results

        Raises:
            ValueError: If configs not found or < 2 configs

        Example:
            ```python
            # Run multiple optimizations
            opt1 = await optimizer.optimize([...], method="bayesian")
            opt2 = await optimizer.optimize([...], method="grid")

            # Compare
            comparison = await optimizer.compare_configs([
                opt1.optimization_id,
                opt2.optimization_id
            ])

            for config_id, config_data in comparison['configs'].items():
                print(f"{config_id}: {config_data['type']}")
                if config_data['type'] == 'optimization':
                    print(f"  Best score: {config_data['best_score']:.4f}")
                    print(f"  Best params: {config_data['best_params']}")
            ```
        """
        response = await self._optimizer.handle_compare_configs(config_ids)
        logger.info(f"Compared {len(config_ids)} configs")
        return cast(Dict[str, Any], response)


# Re-export standalone functions for backward compatibility
from beanllm.facade.advanced.optimizer_standalone import (  # noqa: E402
    quick_optimizer,
    quick_profile,
)

__all__ = [
    "Optimizer",
    "quick_optimizer",
    "quick_profile",
]
