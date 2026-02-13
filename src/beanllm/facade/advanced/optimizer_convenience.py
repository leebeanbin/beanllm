"""
Optimizer convenience methods - quick_optimize, quick_benchmark, etc.

Extracted from optimizer_facade for smaller, focused modules.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

from beanllm.dto.response.advanced.optimizer_response import (
    BenchmarkResponse,
    OptimizeResponse,
    ProfileResponse,
    RecommendationResponse,
)


class OptimizerConvenienceMixin:
    """
    Mixin providing convenience methods for Optimizer facade.

    Expects the mixed-in class to have: optimize(), benchmark(), profile(),
    get_recommendations() and _optimizer (handler).
    """

    async def quick_optimize(
        self,
        top_k_range: tuple[int, int] = (1, 20),
        threshold_range: tuple[float, float] = (0.0, 1.0),
        method: str = "bayesian",
        n_trials: int = 30,
    ) -> OptimizeResponse:
        """
        Quick parameter optimization for common RAG parameters.

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
        return cast(
            OptimizeResponse,
            await self.optimize(  # type: ignore[attr-defined]
                parameters=parameters,
                method=method,
                n_trials=n_trials,
            ),
        )

    async def quick_benchmark(
        self,
        domain: str = "general",
        num_queries: int = 30,
    ) -> BenchmarkResponse:
        """
        Quick benchmark with defaults.

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
        return cast(
            BenchmarkResponse,
            await self.benchmark(  # type: ignore[attr-defined]
                num_queries=num_queries,
                query_types=["simple", "complex"],
                domain=domain,
            ),
        )

    async def quick_profile_and_recommend(
        self,
        components: Optional[List[str]] = None,
    ) -> tuple[ProfileResponse, RecommendationResponse]:
        """
        Profile system and get recommendations in one call.

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
        profile = await self.profile(components=components)  # type: ignore[attr-defined]
        recommendations = await self.get_recommendations(  # type: ignore[attr-defined]
            profile.profile_id
        )
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
        Multi-objective optimization for quality, latency, and cost.

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
        return cast(
            OptimizeResponse,
            await self.optimize(  # type: ignore[attr-defined]
                parameters=parameters,
                method="random",
                n_trials=n_trials,
                multi_objective=True,
                objectives=objectives,
            ),
        )

    async def benchmark_and_optimize(
        self,
        parameters: List[Dict[str, Any]],
        benchmark_num_queries: int = 30,
        optimize_n_trials: int = 30,
    ) -> Dict[str, Any]:
        """
        Run benchmark, then optimize parameters.

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
        benchmark_result = await self.benchmark(  # type: ignore[attr-defined]
            num_queries=benchmark_num_queries
        )
        optimization_result = await self.optimize(  # type: ignore[attr-defined]
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
        Automatic tuning pipeline: profile → optimize → recommend.

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
                rec = results['recommendations'].recommendations[0]
                print(f"Top recommendation: {rec['title']}")
            ```
        """
        results: Dict[str, Any] = {}

        if profile:
            profile_result = await self.profile()  # type: ignore[attr-defined]
            results["profile"] = profile_result

            if recommend:
                recommendations = await self.get_recommendations(  # type: ignore[attr-defined]
                    profile_result.profile_id
                )
                results["recommendations"] = recommendations

        if optimize:
            optimization = await self.quick_optimize(n_trials=30)
            results["optimization"] = optimization

        return results
