"""
OptimizerServiceImpl - Auto-Optimizer 서비스 구현체
SOLID 원칙:
- SRP: 최적화 비즈니스 로직만 담당
- DIP: 인터페이스에 의존
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from beanllm.domain.optimizer import (
    ABTestResult,
    ABTester,
    BenchmarkQuery,
    BenchmarkResult,
    Benchmarker,
    MultiObjectiveResult,
    Objective,
    OptimizationMethod,
    OptimizationResult,
    OptimizerEngine,
    ParameterSearch,
    ParameterSpace,
    ParameterType,
    Priority,
    ProfileResult,
    Profiler,
    QueryType,
    Recommendation,
    RecommendationCategory,
    Recommender,
)
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

from ...optimizer_service import IOptimizerService

logger = get_logger(__name__)


class OptimizerServiceImpl(IOptimizerService):
    """
    Auto-Optimizer 서비스 구현체

    책임:
    - 벤치마킹 실행
    - 파라미터 최적화
    - 시스템 프로파일링
    - A/B 테스팅
    - 최적화 권장사항 생성
    """

    def __init__(self) -> None:
        """Initialize optimizer service with domain objects"""
        # Domain objects
        self._benchmarker = Benchmarker()
        self._optimizer_engine = OptimizerEngine()
        self._profiler = Profiler()
        self._ab_tester = ABTester()
        self._recommender = Recommender()
        self._param_search = ParameterSearch()

        # State storage
        self._benchmarks: Dict[str, BenchmarkResult] = {}
        self._optimizations: Dict[str, OptimizationResult] = {}
        self._profiles: Dict[str, ProfileResult] = {}
        self._ab_tests: Dict[str, ABTestResult] = {}

        logger.info("OptimizerServiceImpl initialized")

    async def benchmark(self, request: BenchmarkRequest) -> BenchmarkResponse:
        """
        Run benchmark with synthetic or provided queries

        Args:
            request: BenchmarkRequest

        Returns:
            BenchmarkResponse: Benchmark results

        Raises:
            ValueError: If system_fn is not provided
            RuntimeError: If benchmark execution fails
        """
        logger.info(
            f"Running benchmark: {request.num_queries} queries, "
            f"types={request.query_types}"
        )

        benchmark_id = str(uuid.uuid4())

        try:
            # Generate or use provided queries
            if request.queries:
                queries = [
                    BenchmarkQuery(
                        query=q,
                        type=QueryType.SIMPLE,
                        expected_answer=None,
                        metadata={},
                    )
                    for q in request.queries
                ]
            else:
                # Generate synthetic queries
                query_types = (
                    [QueryType[qt.upper()] for qt in request.query_types]
                    if request.query_types
                    else None
                )

                queries = self._benchmarker.generate_queries(
                    num_queries=request.num_queries or 50,
                    query_types=query_types,
                    domain=request.domain,
                )

            # Run benchmark
            # Note: system_fn should be provided by the caller
            # For now, we'll store the queries and return a placeholder result
            # In production, this would call an actual system_fn

            # Create result
            result = BenchmarkResult(
                queries=queries,
                latencies=[],
                scores=[],
            )

            # Store benchmark
            self._benchmarks[benchmark_id] = result

            logger.info(
                f"Benchmark completed: {benchmark_id}, "
                f"{len(queries)} queries generated"
            )

            return BenchmarkResponse(
                benchmark_id=benchmark_id,
                num_queries=len(queries),
                queries=[q.query for q in queries],
                avg_latency=result.avg_latency,
                p50_latency=result.p50_latency,
                p95_latency=result.p95_latency,
                p99_latency=result.p99_latency,
                avg_score=result.avg_score,
                min_score=result.min_score,
                max_score=result.max_score,
                throughput=result.throughput,
                total_duration=result.total_duration,
                metadata={
                    "query_types": request.query_types or [],
                    "domain": request.domain,
                },
            )

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise RuntimeError(f"Failed to run benchmark: {e}") from e

    async def optimize(self, request: OptimizeRequest) -> OptimizeResponse:
        """
        Optimize parameters using selected algorithm

        Args:
            request: OptimizeRequest

        Returns:
            OptimizeResponse: Optimization results

        Raises:
            ValueError: If parameter spaces or objective_fn not provided
            RuntimeError: If optimization fails
        """
        logger.info(
            f"Starting optimization: method={request.method}, "
            f"n_trials={request.n_trials}"
        )

        optimization_id = str(uuid.uuid4())

        try:
            # Build parameter spaces
            param_spaces = []
            for param in request.parameters:
                param_type = ParameterType[param["type"].upper()]

                if param_type == ParameterType.INTEGER:
                    space = ParameterSpace(
                        name=param["name"],
                        type=param_type,
                        low=param["low"],
                        high=param["high"],
                    )
                elif param_type == ParameterType.FLOAT:
                    space = ParameterSpace(
                        name=param["name"],
                        type=param_type,
                        low=param["low"],
                        high=param["high"],
                    )
                elif param_type == ParameterType.CATEGORICAL:
                    space = ParameterSpace(
                        name=param["name"],
                        type=param_type,
                        categories=param["categories"],
                    )
                elif param_type == ParameterType.BOOLEAN:
                    space = ParameterSpace(
                        name=param["name"],
                        type=param_type,
                    )

                param_spaces.append(space)

            # Determine optimization method
            if request.multi_objective and len(request.objectives or []) > 1:
                # Multi-objective optimization
                result = await self._optimize_multi_objective(
                    param_spaces=param_spaces,
                    objectives=request.objectives or [],
                    n_trials=request.n_trials or 50,
                )

                # Get best balanced solution
                best_params = result.pareto_frontier[0].params
                best_score = result.pareto_frontier[0].combined_score
                history = [
                    {
                        "trial": i,
                        "params": r.params,
                        "scores": r.scores,
                        "combined_score": r.combined_score,
                    }
                    for i, r in enumerate(result.results)
                ]

                optimization_result = OptimizationResult(
                    best_params=best_params,
                    best_score=best_score,
                    history=history,
                    convergence_data={"pareto_size": len(result.pareto_frontier)},
                )

            else:
                # Single-objective optimization
                method = OptimizationMethod[request.method.upper()]

                # Note: objective_fn should be provided by caller
                # For now, we'll create a placeholder
                optimization_result = OptimizationResult(
                    best_params={space.name: space.sample() for space in param_spaces},
                    best_score=0.0,
                    history=[],
                    convergence_data={},
                )

            # Store optimization
            self._optimizations[optimization_id] = optimization_result

            logger.info(
                f"Optimization completed: {optimization_id}, "
                f"best_score={optimization_result.best_score:.4f}"
            )

            return OptimizeResponse(
                optimization_id=optimization_id,
                best_params=optimization_result.best_params,
                best_score=optimization_result.best_score,
                n_trials=len(optimization_result.history),
                convergence_data=optimization_result.convergence_data,
                metadata={
                    "method": request.method,
                    "multi_objective": request.multi_objective,
                },
            )

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise RuntimeError(f"Failed to optimize parameters: {e}") from e

    async def profile(self, request: ProfileRequest) -> ProfileResponse:
        """
        Profile system components

        Args:
            request: ProfileRequest

        Returns:
            ProfileResponse: Profiling results

        Raises:
            RuntimeError: If profiling fails
        """
        logger.info(f"Starting profiling: {request.components}")

        profile_id = str(uuid.uuid4())

        try:
            # Create profiler
            profiler = Profiler()

            # Note: Actual profiling should be done by caller
            # For now, we'll create a placeholder result
            result = ProfileResult(
                components={},
            )

            # Store profile
            self._profiles[profile_id] = result

            # Generate recommendations
            recommendations = self._recommender.analyze_profile(result)

            logger.info(
                f"Profiling completed: {profile_id}, "
                f"{len(recommendations)} recommendations"
            )

            return ProfileResponse(
                profile_id=profile_id,
                total_duration_ms=result.total_duration_ms,
                total_tokens=result.total_tokens,
                total_cost=result.total_cost,
                components={
                    name: {
                        "duration_ms": metrics.duration_ms,
                        "tokens": metrics.tokens,
                        "cost": metrics.cost,
                    }
                    for name, metrics in result.components.items()
                },
                bottleneck=result.bottleneck,
                breakdown=result.get_breakdown(),
                recommendations=[
                    {
                        "category": rec.category.value,
                        "priority": rec.priority.value,
                        "title": rec.title,
                        "description": rec.description,
                        "action": rec.action,
                        "expected_impact": rec.expected_impact,
                    }
                    for rec in recommendations
                ],
                metadata={
                    "components_profiled": request.components or [],
                },
            )

        except Exception as e:
            logger.error(f"Profiling failed: {e}")
            raise RuntimeError(f"Failed to profile system: {e}") from e

    async def ab_test(self, request: ABTestRequest) -> ABTestResponse:
        """
        Run A/B test

        Args:
            request: ABTestRequest

        Returns:
            ABTestResponse: A/B test results

        Raises:
            ValueError: If variants not provided
            RuntimeError: If A/B test fails
        """
        logger.info(
            f"Running A/B test: {request.variant_a_name} vs {request.variant_b_name}, "
            f"{request.num_queries} queries"
        )

        test_id = str(uuid.uuid4())

        try:
            # Note: Variants should be provided by caller
            # For now, we'll create a placeholder result
            result = ABTestResult(
                variant_a_name=request.variant_a_name,
                variant_b_name=request.variant_b_name,
                variant_a_mean=0.0,
                variant_b_mean=0.0,
                variant_a_std=0.0,
                variant_b_std=0.0,
                p_value=1.0,
                is_significant=False,
                confidence_level=request.confidence_level or 0.95,
                sample_size_a=request.num_queries or 50,
                sample_size_b=request.num_queries or 50,
            )

            # Store test
            self._ab_tests[test_id] = result

            logger.info(
                f"A/B test completed: {test_id}, "
                f"winner={result.winner}, lift={result.lift:.1f}%"
            )

            return ABTestResponse(
                test_id=test_id,
                variant_a_name=result.variant_a_name,
                variant_b_name=result.variant_b_name,
                variant_a_mean=result.variant_a_mean,
                variant_b_mean=result.variant_b_mean,
                p_value=result.p_value,
                is_significant=result.is_significant,
                winner=result.winner,
                lift=result.lift,
                confidence_level=result.confidence_level,
                metadata={
                    "num_queries": request.num_queries,
                },
            )

        except Exception as e:
            logger.error(f"A/B test failed: {e}")
            raise RuntimeError(f"Failed to run A/B test: {e}") from e

    async def get_recommendations(self, profile_id: str) -> RecommendationResponse:
        """
        Get optimization recommendations for a profile

        Args:
            profile_id: Profile ID

        Returns:
            RecommendationResponse: Recommendations

        Raises:
            ValueError: If profile not found
        """
        logger.info(f"Getting recommendations for profile: {profile_id}")

        if profile_id not in self._profiles:
            raise ValueError(f"Profile not found: {profile_id}")

        profile_result = self._profiles[profile_id]

        # Generate recommendations
        recommendations = self._recommender.analyze_profile(profile_result)

        # Sort by priority
        priority_order = {
            Priority.CRITICAL: 0,
            Priority.HIGH: 1,
            Priority.MEDIUM: 2,
            Priority.LOW: 3,
        }
        recommendations = sorted(
            recommendations, key=lambda r: priority_order[r.priority]
        )

        logger.info(f"Generated {len(recommendations)} recommendations")

        return RecommendationResponse(
            profile_id=profile_id,
            recommendations=[
                {
                    "category": rec.category.value,
                    "priority": rec.priority.value,
                    "title": rec.title,
                    "description": rec.description,
                    "rationale": rec.rationale,
                    "action": rec.action,
                    "expected_impact": rec.expected_impact,
                }
                for rec in recommendations
            ],
            summary={
                "critical": len(
                    [r for r in recommendations if r.priority == Priority.CRITICAL]
                ),
                "high": len([r for r in recommendations if r.priority == Priority.HIGH]),
                "medium": len(
                    [r for r in recommendations if r.priority == Priority.MEDIUM]
                ),
                "low": len([r for r in recommendations if r.priority == Priority.LOW]),
            },
        )

    async def compare_configs(self, config_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple configurations

        Args:
            config_ids: List of config IDs (optimization_id, profile_id, etc.)

        Returns:
            Dict with comparison results

        Raises:
            ValueError: If configs not found
        """
        logger.info(f"Comparing {len(config_ids)} configs")

        results = {}

        for config_id in config_ids:
            # Try to find in different stores
            if config_id in self._optimizations:
                opt_result = self._optimizations[config_id]
                results[config_id] = {
                    "type": "optimization",
                    "best_params": opt_result.best_params,
                    "best_score": opt_result.best_score,
                    "n_trials": len(opt_result.history),
                }

            elif config_id in self._profiles:
                profile_result = self._profiles[config_id]
                results[config_id] = {
                    "type": "profile",
                    "total_duration_ms": profile_result.total_duration_ms,
                    "total_cost": profile_result.total_cost,
                    "bottleneck": profile_result.bottleneck,
                }

            elif config_id in self._ab_tests:
                test_result = self._ab_tests[config_id]
                results[config_id] = {
                    "type": "ab_test",
                    "winner": test_result.winner,
                    "lift": test_result.lift,
                    "is_significant": test_result.is_significant,
                }

            else:
                logger.warning(f"Config not found: {config_id}")
                results[config_id] = {
                    "type": "unknown",
                    "error": "Config not found",
                }

        logger.info(f"Comparison completed: {len(results)} configs")

        return {
            "configs": results,
            "summary": {
                "total_configs": len(config_ids),
                "found": len([r for r in results.values() if r.get("type") != "unknown"]),
            },
        }

    async def _optimize_multi_objective(
        self,
        param_spaces: List[ParameterSpace],
        objectives: List[Dict[str, Any]],
        n_trials: int,
    ) -> MultiObjectiveResult:
        """
        Run multi-objective optimization

        Args:
            param_spaces: Parameter spaces
            objectives: Objective definitions
            n_trials: Number of trials

        Returns:
            MultiObjectiveResult
        """
        logger.info(
            f"Running multi-objective optimization: {len(objectives)} objectives"
        )

        # Build objectives
        objective_list = []
        for obj_def in objectives:
            # Note: objective functions should be provided by caller
            # For now, we'll create placeholders
            objective = Objective(
                name=obj_def["name"],
                fn=lambda params: 0.0,  # Placeholder
                maximize=obj_def.get("maximize", True),
                weight=obj_def.get("weight", 1.0),
            )
            objective_list.append(objective)

        # Run optimization
        result = self._param_search.multi_objective_search(
            param_spaces=param_spaces,
            objectives=objective_list,
            n_trials=n_trials,
            method="random",
        )

        logger.info(
            f"Multi-objective optimization completed: "
            f"{len(result.pareto_frontier)} Pareto optimal solutions"
        )

        return result
