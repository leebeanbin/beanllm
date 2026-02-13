"""
OptimizerEngine - 핵심 최적화 알고리즘

다양한 최적화 방법을 통합하고 파라미터 공간 탐색을 조율합니다.
SOLID: SRP(알고리즘 위임), OCP(새 전략 추가 용이)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from beanllm.domain.optimizer.optimization_evaluation import get_convergence_plot_data
from beanllm.domain.optimizer.optimization_strategies import (
    run_bayesian,
    run_genetic,
    run_grid,
    run_random,
)
from beanllm.domain.optimizer.parameter_management import (
    OptimizationResult,
    ParameterSpace,
    ParameterType,
)
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class OptimizationMethod(Enum):
    """최적화 방법"""

    BAYESIAN = "bayesian"
    GRID = "grid"
    RANDOM = "random"
    GENETIC = "genetic"


class OptimizerEngine:
    """
    최적화 엔진

    다양한 최적화 알고리즘을 제공하고 파라미터 공간 탐색을 조율합니다.

    Example:
        ```python
        param_spaces = [
            ParameterSpace("top_k", ParameterType.INTEGER, low=1, high=20),
            ParameterSpace("threshold", ParameterType.FLOAT, low=0.0, high=1.0),
        ]
        engine = OptimizerEngine()
        result = engine.optimize(
            param_spaces=param_spaces,
            objective_fn=objective,
            method=OptimizationMethod.BAYESIAN,
            n_trials=30
        )
        ```
    """

    def __init__(self) -> None:
        self.history: List[Dict[str, Any]] = []

    def optimize(
        self,
        param_spaces: List[ParameterSpace],
        objective_fn: Callable[[Dict[str, Any]], float],
        method: OptimizationMethod = OptimizationMethod.BAYESIAN,
        n_trials: int = 30,
        initial_params: Optional[Dict[str, Any]] = None,
        maximize: bool = True,
        **kwargs: Any,
    ) -> OptimizationResult:
        """
        최적화 실행

        Args:
            param_spaces: 파라미터 공간 정의 리스트
            objective_fn: 목적 함수 (params -> score)
            method: 최적화 방법
            n_trials: 시행 횟수
            initial_params: 초기 파라미터 (optional)
            maximize: True면 최대화, False면 최소화
            **kwargs: 전략별 추가 옵션

        Returns:
            OptimizationResult
        """
        logger.info(f"Starting optimization: method={method.value}, n_trials={n_trials}")

        self.history = []

        strategy_map = {
            OptimizationMethod.BAYESIAN: run_bayesian,
            OptimizationMethod.GRID: run_grid,
            OptimizationMethod.RANDOM: run_random,
            OptimizationMethod.GENETIC: run_genetic,
        }

        strategy_fn = strategy_map.get(method)
        if strategy_fn is None:
            raise ValueError(f"Unknown optimization method: {method}")

        result: OptimizationResult
        if method == OptimizationMethod.GRID:
            result = run_grid(param_spaces, objective_fn, maximize, self.history, **kwargs)
        elif method == OptimizationMethod.BAYESIAN:
            result = run_bayesian(
                param_spaces, objective_fn, n_trials, maximize, self.history, **kwargs
            )
        elif method == OptimizationMethod.RANDOM:
            result = run_random(
                param_spaces, objective_fn, n_trials, maximize, self.history, **kwargs
            )
        elif method == OptimizationMethod.GENETIC:
            result = run_genetic(
                param_spaces, objective_fn, n_trials, maximize, self.history, **kwargs
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        result.method = method.value
        result.history = self.history

        logger.info(
            f"Optimization completed: best_score={result.best_score:.4f}, "
            f"trials={result.total_trials}"
        )

        return result

    def get_convergence_plot_data(self) -> Tuple[List[int], List[float]]:
        """수렴 그래프 데이터 (trial_nums, best_scores_so_far) 반환"""
        return get_convergence_plot_data(self.history)


__all__ = [
    "OptimizationMethod",
    "OptimizerEngine",
    "OptimizationResult",
    "ParameterSpace",
    "ParameterType",
]
