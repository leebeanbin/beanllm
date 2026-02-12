"""
Optimizer Facade - User-friendly Auto-Optimizer API
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

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
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class Optimizer:
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
            # Default initialization (ServiceFactory 경유)
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
        return response

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
        return response

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
        return response

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
        return response

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
        return response

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
        return response

    # ===== Convenience Methods =====

    async def quick_optimize(
        self,
        top_k_range: tuple = (1, 20),
        threshold_range: tuple = (0.0, 1.0),
        method: str = "bayesian",
        n_trials: int = 30,
    ) -> OptimizeResponse:
        """
        Quick parameter optimization for common RAG parameters

        Args:
            top_k_range: Range for top_k (default: 1-20)
            threshold_range: Range for score_threshold (default: 0.0-1.0)
            method: Optimization method (default: "bayesian")
            n_trials: Number of trials (default: 30)

        Returns:
            OptimizeResponse: Optimization results

        Example:
            ```python
            result = await optimizer.quick_optimize(
                top_k_range=(5, 15),
                threshold_range=(0.5, 0.9),
                n_trials=20
            )
            print(f"Optimal top_k: {result.best_params['top_k']}")
            print(f"Optimal threshold: {result.best_params['threshold']:.2f}")
            ```
        """
        parameters = [
            {
                "name": "top_k",
                "type": "integer",
                "low": top_k_range[0],
                "high": top_k_range[1],
            },
            {
                "name": "score_threshold",
                "type": "float",
                "low": threshold_range[0],
                "high": threshold_range[1],
            },
        ]

        return await self.optimize(
            parameters=parameters,
            method=method,
            n_trials=n_trials,
        )

    async def quick_benchmark(
        self,
        domain: str = "general",
        num_queries: int = 30,
    ) -> BenchmarkResponse:
        """
        Quick benchmark with defaults

        Args:
            domain: Domain for queries (default: "general")
            num_queries: Number of queries (default: 30)

        Returns:
            BenchmarkResponse: Benchmark results

        Example:
            ```python
            result = await optimizer.quick_benchmark(
                domain="machine learning",
                num_queries=50
            )
            print(f"Avg latency: {result.avg_latency:.3f}s")
            print(f"P95 latency: {result.p95_latency:.3f}s")
            ```
        """
        return await self.benchmark(
            num_queries=num_queries,
            query_types=["simple", "complex"],
            domain=domain,
        )

    async def quick_profile_and_recommend(
        self,
        components: Optional[List[str]] = None,
    ) -> tuple[ProfileResponse, RecommendationResponse]:
        """
        Profile system and get recommendations in one call

        Args:
            components: Components to profile (default: all)

        Returns:
            Tuple of (ProfileResponse, RecommendationResponse)

        Example:
            ```python
            profile, recs = await optimizer.quick_profile_and_recommend()

            print(f"Bottleneck: {profile.bottleneck}")
            print(f"Total cost: ${profile.total_cost:.4f}")

            print(f"\\nTop recommendations:")
            for rec in recs.recommendations[:3]:
                print(f"- [{rec['priority']}] {rec['title']}")
            ```
        """
        profile = await self.profile(components=components)
        recommendations = await self.get_recommendations(profile.profile_id)
        return profile, recommendations

    async def multi_objective_optimize(
        self,
        parameters: List[Dict[str, Any]],
        quality_weight: float = 0.6,
        latency_weight: float = 0.3,
        cost_weight: float = 0.1,
        n_trials: int = 50,
    ) -> OptimizeResponse:
        """
        Multi-objective optimization for quality, latency, and cost

        Args:
            parameters: Parameter definitions
            quality_weight: Weight for quality objective (default: 0.6)
            latency_weight: Weight for latency objective (default: 0.3)
            cost_weight: Weight for cost objective (default: 0.1)
            n_trials: Number of trials (default: 50)

        Returns:
            OptimizeResponse: Pareto optimal solution

        Example:
            ```python
            result = await optimizer.multi_objective_optimize(
                parameters=[
                    {"name": "top_k", "type": "integer", "low": 1, "high": 20},
                    {"name": "model", "type": "categorical",
                     "categories": ["gpt-3.5-turbo", "gpt-4"]},
                ],
                quality_weight=0.7,
                latency_weight=0.2,
                cost_weight=0.1,
                n_trials=50
            )
            ```
        """
        objectives = [
            {"name": "quality", "maximize": True, "weight": quality_weight},
            {"name": "latency", "maximize": False, "weight": latency_weight},
            {"name": "cost", "maximize": False, "weight": cost_weight},
        ]

        return await self.optimize(
            parameters=parameters,
            method="random",  # Multi-objective uses random sampling
            n_trials=n_trials,
            multi_objective=True,
            objectives=objectives,
        )

    async def benchmark_and_optimize(
        self,
        parameters: List[Dict[str, Any]],
        benchmark_num_queries: int = 30,
        optimize_n_trials: int = 30,
    ) -> Dict[str, Any]:
        """
        Run benchmark, then optimize parameters

        Args:
            parameters: Parameter definitions
            benchmark_num_queries: Number of benchmark queries (default: 30)
            optimize_n_trials: Number of optimization trials (default: 30)

        Returns:
            Dict with both benchmark and optimization results

        Example:
            ```python
            result = await optimizer.benchmark_and_optimize(
                parameters=[
                    {"name": "top_k", "type": "integer", "low": 1, "high": 20},
                ],
                benchmark_num_queries=50,
                optimize_n_trials=30
            )

            print(f"Baseline avg latency: {result['benchmark'].avg_latency:.3f}s")
            print(f"Optimized params: {result['optimization'].best_params}")
            ```
        """
        # Run benchmark
        benchmark_result = await self.benchmark(num_queries=benchmark_num_queries)

        # Run optimization
        optimization_result = await self.optimize(
            parameters=parameters,
            n_trials=optimize_n_trials,
        )

        return {
            "benchmark": benchmark_result,
            "optimization": optimization_result,
        }

    async def auto_tune(
        self,
        profile: bool = True,
        optimize: bool = True,
        recommend: bool = True,
    ) -> Dict[str, Any]:
        """
        Automatic tuning pipeline: profile → optimize → recommend

        Args:
            profile: Run profiling (default: True)
            optimize: Run optimization (default: True)
            recommend: Generate recommendations (default: True)

        Returns:
            Dict with all results

        Example:
            ```python
            results = await optimizer.auto_tune(
                profile=True,
                optimize=True,
                recommend=True
            )

            if 'profile' in results:
                print(f"Bottleneck: {results['profile'].bottleneck}")

            if 'optimization' in results:
                print(f"Best params: {results['optimization'].best_params}")

            if 'recommendations' in results:
                print(f"Top recommendation: {results['recommendations'].recommendations[0]['title']}")
            ```
        """
        results: Dict[str, Any] = {}

        # Profile
        if profile:
            profile_result = await self.profile()
            results["profile"] = profile_result

            # Get recommendations from profile
            if recommend:
                recommendations = await self.get_recommendations(profile_result.profile_id)
                results["recommendations"] = recommendations

        # Optimize common parameters
        if optimize:
            optimization = await self.quick_optimize(n_trials=30)
            results["optimization"] = optimization

        logger.info(
            f"Auto-tune completed: profile={profile}, optimize={optimize}, recommend={recommend}"
        )

        return results


# ===== Standalone Functions =====


async def quick_optimizer(
    parameters: List[Dict[str, Any]],
    method: str = "bayesian",
    n_trials: int = 30,
) -> OptimizeResponse:
    """
    One-liner for quick optimization

    Args:
        parameters: Parameter definitions
        method: Optimization method (default: "bayesian")
        n_trials: Number of trials (default: 30)

    Returns:
        OptimizeResponse: Optimization results

    Example:
        ```python
        from beanllm.facade.optimizer_facade import quick_optimizer

        result = await quick_optimizer(
            parameters=[
                {"name": "top_k", "type": "integer", "low": 1, "high": 20},
            ],
            method="bayesian",
            n_trials=20
        )
        print(f"Best top_k: {result.best_params['top_k']}")
        ```
    """
    optimizer = Optimizer()
    return await optimizer.optimize(
        parameters=parameters,
        method=method,
        n_trials=n_trials,
    )


async def quick_profile() -> ProfileResponse:
    """
    One-liner for quick profiling

    Returns:
        ProfileResponse: Profile results

    Example:
        ```python
        from beanllm.facade.optimizer_facade import quick_profile

        result = await quick_profile()
        print(f"Bottleneck: {result.bottleneck}")
        print(f"Total duration: {result.total_duration_ms}ms")
        ```
    """
    optimizer = Optimizer()
    return await optimizer.profile()
