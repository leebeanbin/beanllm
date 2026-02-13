"""
OptimizerServiceImpl - Auto-Optimizer 서비스 구현체 (core + mixins).
"""

from __future__ import annotations

from typing import Any, Dict, List

from beanllm.domain.optimizer import (
    ABTester,
    ABTestResult,
    Benchmarker,
    BenchmarkResult,
    OptimizationResult,
    OptimizerEngine,
    Profiler,
    ProfileResult,
    Recommender,
)
from beanllm.service.impl.advanced.optimizer_ab_test import OptimizerABTestMixin
from beanllm.service.impl.advanced.optimizer_benchmark import OptimizerBenchmarkMixin
from beanllm.service.impl.advanced.optimizer_optimize import OptimizerOptimizeMixin
from beanllm.service.impl.advanced.optimizer_profile import OptimizerProfileMixin
from beanllm.service.optimizer_service import IOptimizerService
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class OptimizerServiceImpl(
    OptimizerBenchmarkMixin,
    OptimizerOptimizeMixin,
    OptimizerProfileMixin,
    OptimizerABTestMixin,
    IOptimizerService,
):
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

        # State storage
        self._benchmarks: Dict[str, BenchmarkResult] = {}
        self._optimizations: Dict[str, OptimizationResult] = {}
        self._profiles: Dict[str, ProfileResult] = {}
        self._ab_tests: Dict[str, ABTestResult] = {}

        logger.info("OptimizerServiceImpl initialized")

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
