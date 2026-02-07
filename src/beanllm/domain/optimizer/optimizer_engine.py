"""
OptimizerEngine - 핵심 최적화 알고리즘
SOLID 원칙:
- SRP: 최적화 알고리즘만 담당
- OCP: 새로운 최적화 방법 추가 가능
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class OptimizationMethod(Enum):
    """최적화 방법"""

    BAYESIAN = "bayesian"  # Bayesian Optimization
    GRID = "grid"  # Grid Search
    RANDOM = "random"  # Random Search
    GENETIC = "genetic"  # Genetic Algorithm


class ParameterType(Enum):
    """파라미터 타입"""

    INTEGER = "integer"  # 정수
    FLOAT = "float"  # 실수
    CATEGORICAL = "categorical"  # 범주형
    BOOLEAN = "boolean"  # 불리언


@dataclass
class ParameterSpace:
    """
    파라미터 공간 정의

    Example:
        ```python
        # Integer parameter
        top_k = ParameterSpace(
            name="top_k",
            type=ParameterType.INTEGER,
            low=1,
            high=20
        )

        # Float parameter
        threshold = ParameterSpace(
            name="score_threshold",
            type=ParameterType.FLOAT,
            low=0.0,
            high=1.0
        )

        # Categorical parameter
        strategy = ParameterSpace(
            name="strategy",
            type=ParameterType.CATEGORICAL,
            categories=["bm25", "semantic", "hybrid"]
        )
        ```
    """

    name: str
    type: ParameterType
    low: Optional[float] = None
    high: Optional[float] = None
    categories: Optional[List[Any]] = None
    default: Optional[Any] = None

    def __post_init__(self):
        """검증"""
        if self.type in [ParameterType.INTEGER, ParameterType.FLOAT]:
            if self.low is None or self.high is None:
                raise ValueError(f"{self.name}: low and high are required for {self.type}")
        elif self.type == ParameterType.CATEGORICAL:
            if not self.categories:
                raise ValueError(f"{self.name}: categories are required for CATEGORICAL")

    def sample(self) -> Any:
        """랜덤 샘플링"""
        if self.type == ParameterType.INTEGER:
            # __post_init__에서 검증되므로 low/high는 None이 아님
            assert self.low is not None and self.high is not None
            return random.randint(int(self.low), int(self.high))
        elif self.type == ParameterType.FLOAT:
            assert self.low is not None and self.high is not None
            return random.uniform(self.low, self.high)
        elif self.type == ParameterType.CATEGORICAL:
            assert self.categories is not None
            return random.choice(self.categories)
        elif self.type == ParameterType.BOOLEAN:
            return random.choice([True, False])
        return None


@dataclass
class OptimizationResult:
    """
    최적화 결과

    Attributes:
        best_params: 최적 파라미터
        best_score: 최고 점수
        total_trials: 총 시행 횟수
        history: 최적화 히스토리 [{params, score, trial_num}, ...]
        method: 사용된 최적화 방법
        metadata: 추가 메타데이터
    """

    best_params: Dict[str, Any]
    best_score: float
    total_trials: int
    history: List[Dict[str, Any]] = field(default_factory=list)
    method: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_top_n(self, n: int = 5) -> List[Dict[str, Any]]:
        """상위 N개 결과 반환"""
        sorted_history = sorted(self.history, key=lambda x: x["score"], reverse=True)
        return sorted_history[:n]


class OptimizerEngine:
    """
    최적화 엔진

    책임:
    - 다양한 최적화 알고리즘 제공
    - 파라미터 공간 탐색
    - 최적화 히스토리 관리

    Example:
        ```python
        # Define parameter space
        param_spaces = [
            ParameterSpace("top_k", ParameterType.INTEGER, low=1, high=20),
            ParameterSpace("threshold", ParameterType.FLOAT, low=0.0, high=1.0),
        ]

        # Define objective function
        def objective(params):
            # Evaluate RAG system with params
            result = rag.query(query, top_k=params["top_k"], threshold=params["threshold"])
            return evaluate_quality(result)  # returns score 0.0-1.0

        # Optimize
        engine = OptimizerEngine()
        result = engine.optimize(
            param_spaces=param_spaces,
            objective_fn=objective,
            method=OptimizationMethod.BAYESIAN,
            n_trials=30
        )

        print(f"Best params: {result.best_params}")
        print(f"Best score: {result.best_score}")
        ```
    """

    def __init__(self) -> None:
        """Initialize optimizer engine"""
        self.history: List[Dict[str, Any]] = []

    def optimize(
        self,
        param_spaces: List[ParameterSpace],
        objective_fn: Callable[[Dict[str, Any]], float],
        method: OptimizationMethod = OptimizationMethod.BAYESIAN,
        n_trials: int = 30,
        initial_params: Optional[Dict[str, Any]] = None,
        maximize: bool = True,
        **kwargs,
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
            **kwargs: 추가 옵션

        Returns:
            OptimizationResult: 최적화 결과

        Raises:
            ValueError: 잘못된 파라미터
        """
        logger.info(f"Starting optimization: method={method.value}, n_trials={n_trials}")

        # Reset history
        self.history = []

        # Optimize based on method
        if method == OptimizationMethod.BAYESIAN:
            result = self._optimize_bayesian(
                param_spaces, objective_fn, n_trials, maximize, **kwargs
            )
        elif method == OptimizationMethod.GRID:
            result = self._optimize_grid(param_spaces, objective_fn, maximize, **kwargs)
        elif method == OptimizationMethod.RANDOM:
            result = self._optimize_random(param_spaces, objective_fn, n_trials, maximize)
        elif method == OptimizationMethod.GENETIC:
            result = self._optimize_genetic(
                param_spaces, objective_fn, n_trials, maximize, **kwargs
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

    def _optimize_bayesian(
        self,
        param_spaces: List[ParameterSpace],
        objective_fn: Callable[[Dict[str, Any]], float],
        n_trials: int,
        maximize: bool = True,
        **kwargs,
    ) -> OptimizationResult:
        """
        Bayesian Optimization

        Uses Gaussian Process to model objective function and
        select next parameters to try.
        """
        try:
            from bayes_opt import BayesianOptimization
        except ImportError:
            logger.warning("bayesian-optimization not installed. Falling back to random search.")
            return self._optimize_random(param_spaces, objective_fn, n_trials, maximize)

        # Build parameter bounds for BayesianOptimization
        pbounds = {}
        categorical_params = {}

        for space in param_spaces:
            if space.type in [ParameterType.INTEGER, ParameterType.FLOAT]:
                assert space.low is not None and space.high is not None
                pbounds[space.name] = (space.low, space.high)
            elif space.type == ParameterType.CATEGORICAL:
                # Map categories to integers
                assert space.categories is not None
                categorical_params[space.name] = space.categories
                pbounds[space.name] = (0, len(space.categories) - 1)
            elif space.type == ParameterType.BOOLEAN:
                pbounds[space.name] = (0, 1)

        # Wrapper for objective function
        def wrapped_objective(**params_dict):
            # Convert categorical indices to actual categories
            actual_params = {}
            for name, value in params_dict.items():
                space = next(s for s in param_spaces if s.name == name)

                if space.type == ParameterType.INTEGER:
                    actual_params[name] = int(round(value))
                elif space.type == ParameterType.FLOAT:
                    actual_params[name] = float(value)
                elif space.type == ParameterType.CATEGORICAL:
                    idx = int(round(value))
                    idx = max(0, min(idx, len(categorical_params[name]) - 1))
                    actual_params[name] = categorical_params[name][idx]
                elif space.type == ParameterType.BOOLEAN:
                    actual_params[name] = value > 0.5

            # Evaluate
            score = objective_fn(actual_params)

            # Store in history
            self.history.append(
                {
                    "trial_num": len(self.history) + 1,
                    "params": actual_params.copy(),
                    "score": score,
                }
            )

            return score if maximize else -score

        # Run Bayesian Optimization
        optimizer = BayesianOptimization(
            f=wrapped_objective,
            pbounds=pbounds,
            random_state=42,
            verbose=0,
        )

        optimizer.maximize(
            init_points=kwargs.get("init_points", 5),
            n_iter=n_trials - kwargs.get("init_points", 5),
        )

        # Extract best result
        best_params_raw = optimizer.max["params"]
        best_params = {}

        for name, value in best_params_raw.items():
            space = next(s for s in param_spaces if s.name == name)

            if space.type == ParameterType.INTEGER:
                best_params[name] = int(round(value))
            elif space.type == ParameterType.FLOAT:
                best_params[name] = float(value)
            elif space.type == ParameterType.CATEGORICAL:
                idx = int(round(value))
                idx = max(0, min(idx, len(categorical_params[name]) - 1))
                best_params[name] = categorical_params[name][idx]
            elif space.type == ParameterType.BOOLEAN:
                best_params[name] = value > 0.5

        best_score = optimizer.max["target"]
        if not maximize:
            best_score = -best_score

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            total_trials=len(self.history),
        )

    def _optimize_grid(
        self,
        param_spaces: List[ParameterSpace],
        objective_fn: Callable[[Dict[str, Any]], float],
        maximize: bool = True,
        **kwargs,
    ) -> OptimizationResult:
        """
        Grid Search

        Exhaustively tries all combinations of parameters.
        """
        grid_size = kwargs.get("grid_size", 5)

        # Generate grid for each parameter
        param_grids = {}

        for space in param_spaces:
            if space.type == ParameterType.INTEGER:
                assert space.low is not None and space.high is not None
                step = max(1, int((space.high - space.low) // grid_size))
                param_grids[space.name] = list(range(int(space.low), int(space.high) + 1, step))
            elif space.type == ParameterType.FLOAT:
                assert space.low is not None and space.high is not None
                step = (space.high - space.low) / grid_size
                param_grids[space.name] = [space.low + i * step for i in range(grid_size + 1)]
            elif space.type == ParameterType.CATEGORICAL:
                assert space.categories is not None
                param_grids[space.name] = space.categories
            elif space.type == ParameterType.BOOLEAN:
                param_grids[space.name] = [True, False]

        # Generate all combinations
        import itertools

        param_names = list(param_grids.keys())
        param_values = list(param_grids.values())

        best_params: Dict[str, Any] = {}
        best_score = float("-inf") if maximize else float("inf")

        for combination in itertools.product(*param_values):
            params = dict(zip(param_names, combination))

            # Evaluate
            score = objective_fn(params)

            # Store in history
            self.history.append(
                {
                    "trial_num": len(self.history) + 1,
                    "params": params.copy(),
                    "score": score,
                }
            )

            # Update best
            if maximize:
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
            else:
                if score < best_score:
                    best_score = score
                    best_params = params.copy()

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            total_trials=len(self.history),
        )

    def _optimize_random(
        self,
        param_spaces: List[ParameterSpace],
        objective_fn: Callable[[Dict[str, Any]], float],
        n_trials: int,
        maximize: bool = True,
    ) -> OptimizationResult:
        """
        Random Search

        Randomly samples from parameter space.
        """
        best_params: Dict[str, Any] = {}
        best_score = float("-inf") if maximize else float("inf")

        for trial in range(n_trials):
            # Sample parameters
            params = {space.name: space.sample() for space in param_spaces}

            # Evaluate
            score = objective_fn(params)

            # Store in history
            self.history.append(
                {
                    "trial_num": trial + 1,
                    "params": params.copy(),
                    "score": score,
                }
            )

            # Update best
            if maximize:
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
            else:
                if score < best_score:
                    best_score = score
                    best_params = params.copy()

            logger.debug(f"Trial {trial + 1}/{n_trials}: score={score:.4f}, " f"params={params}")

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            total_trials=n_trials,
        )

    def _optimize_genetic(
        self,
        param_spaces: List[ParameterSpace],
        objective_fn: Callable[[Dict[str, Any]], float],
        n_trials: int,
        maximize: bool = True,
        **kwargs,
    ) -> OptimizationResult:
        """
        Genetic Algorithm

        Uses evolutionary approach with mutation and crossover.
        """
        population_size = kwargs.get("population_size", 20)
        mutation_rate = kwargs.get("mutation_rate", 0.1)
        crossover_rate = kwargs.get("crossover_rate", 0.7)

        # Initialize population
        population = [
            {space.name: space.sample() for space in param_spaces} for _ in range(population_size)
        ]

        best_params: Dict[str, Any] = {}
        best_score = float("-inf") if maximize else float("inf")

        generations = n_trials // population_size

        for gen in range(generations):
            # Evaluate population
            scores = []
            for individual in population:
                score = objective_fn(individual)

                # Store in history
                self.history.append(
                    {
                        "trial_num": len(self.history) + 1,
                        "params": individual.copy(),
                        "score": score,
                    }
                )

                scores.append(score)

                # Update best
                if maximize:
                    if score > best_score:
                        best_score = score
                        best_params = individual.copy()
                else:
                    if score < best_score:
                        best_score = score
                        best_params = individual.copy()

            # Selection (tournament)
            new_population = []

            for _ in range(population_size):
                # Tournament selection
                tournament = random.sample(list(zip(population, scores)), k=3)
                if maximize:
                    winner = max(tournament, key=lambda x: x[1])[0]
                else:
                    winner = min(tournament, key=lambda x: x[1])[0]

                new_population.append(winner.copy())

            # Crossover
            for i in range(0, len(new_population) - 1, 2):
                if random.random() < crossover_rate:
                    parent1 = new_population[i]
                    parent2 = new_population[i + 1]

                    # Uniform crossover
                    for param_name in parent1.keys():
                        if random.random() < 0.5:
                            parent1[param_name], parent2[param_name] = (
                                parent2[param_name],
                                parent1[param_name],
                            )

            # Mutation
            for individual in new_population:
                if random.random() < mutation_rate:
                    # Mutate one random parameter
                    param_to_mutate = random.choice(param_spaces)
                    individual[param_to_mutate.name] = param_to_mutate.sample()

            population = new_population

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            total_trials=len(self.history),
        )

    def get_convergence_plot_data(self) -> Tuple[List[int], List[float]]:
        """
        수렴 그래프 데이터 반환

        Returns:
            (trial_nums, best_scores_so_far)
        """
        if not self.history:
            return [], []

        trial_nums = []
        best_scores = []
        current_best = float("-inf")

        for entry in self.history:
            trial_nums.append(entry["trial_num"])

            if entry["score"] > current_best:
                current_best = entry["score"]

            best_scores.append(current_best)

        return trial_nums, best_scores
