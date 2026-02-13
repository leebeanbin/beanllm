"""
Optimization Strategies - 최적화 알고리즘 구현

Grid Search, Random Search, Bayesian Optimization, Genetic Algorithm.
"""

from __future__ import annotations

import itertools
import random
from typing import Any, Callable, Dict, List

from beanllm.domain.optimizer.parameter_management import (
    OptimizationResult,
    ParameterSpace,
    ParameterType,
)
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


def run_bayesian(
    param_spaces: List[ParameterSpace],
    objective_fn: Callable[[Dict[str, Any]], float],
    n_trials: int,
    maximize: bool,
    history: List[Dict[str, Any]],
    **kwargs: Any,
) -> OptimizationResult:
    """
    Bayesian Optimization (Gaussian Process 기반).

    bayes_opt 미설치 시 Random Search로 폴백합니다.
    """
    try:
        from bayes_opt import BayesianOptimization
    except ImportError:
        logger.warning("bayesian-optimization not installed. Falling back to random search.")
        return run_random(param_spaces, objective_fn, n_trials, maximize, history)

    pbounds: Dict[str, tuple] = {}
    categorical_params: Dict[str, List[Any]] = {}

    for space in param_spaces:
        if space.type in [ParameterType.INTEGER, ParameterType.FLOAT]:
            assert space.low is not None and space.high is not None
            pbounds[space.name] = (space.low, space.high)
        elif space.type == ParameterType.CATEGORICAL:
            assert space.categories is not None
            categorical_params[space.name] = space.categories
            pbounds[space.name] = (0, len(space.categories) - 1)
        elif space.type == ParameterType.BOOLEAN:
            pbounds[space.name] = (0, 1)

    def wrapped_objective(**params_dict: float) -> float:
        actual_params: Dict[str, Any] = {}
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

        score = objective_fn(actual_params)
        history.append(
            {"trial_num": len(history) + 1, "params": actual_params.copy(), "score": score}
        )
        return score if maximize else -score

    optimizer = BayesianOptimization(
        f=wrapped_objective, pbounds=pbounds, random_state=42, verbose=0
    )
    init_points = kwargs.get("init_points", 5)
    optimizer.maximize(
        init_points=init_points,
        n_iter=n_trials - init_points,
    )

    best_params_raw = optimizer.max["params"]
    best_params: Dict[str, Any] = {}
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

    best_score = float(optimizer.max["target"])
    if not maximize:
        best_score = -best_score

    return OptimizationResult(
        best_params=best_params,
        best_score=best_score,
        total_trials=len(history),
    )


def run_grid(
    param_spaces: List[ParameterSpace],
    objective_fn: Callable[[Dict[str, Any]], float],
    maximize: bool,
    history: List[Dict[str, Any]],
    **kwargs: Any,
) -> OptimizationResult:
    """Grid Search - 모든 파라미터 조합 시도."""
    grid_size = kwargs.get("grid_size", 5)
    param_grids: Dict[str, List[Any]] = {}

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

    param_names = list(param_grids.keys())
    param_values = list(param_grids.values())

    best_params: Dict[str, Any] = {}
    best_score = float("-inf") if maximize else float("inf")

    for combination in itertools.product(*param_values):
        params = dict(zip(param_names, combination))
        score = objective_fn(params)
        history.append({"trial_num": len(history) + 1, "params": params.copy(), "score": score})

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
        total_trials=len(history),
    )


def run_random(
    param_spaces: List[ParameterSpace],
    objective_fn: Callable[[Dict[str, Any]], float],
    n_trials: int,
    maximize: bool,
    history: List[Dict[str, Any]],
) -> OptimizationResult:
    """Random Search - 파라미터 공간에서 랜덤 샘플링."""
    best_params: Dict[str, Any] = {}
    best_score = float("-inf") if maximize else float("inf")

    for trial in range(n_trials):
        params = {space.name: space.sample() for space in param_spaces}
        score = objective_fn(params)
        history.append({"trial_num": trial + 1, "params": params.copy(), "score": score})

        if maximize:
            if score > best_score:
                best_score = score
                best_params = params.copy()
        else:
            if score < best_score:
                best_score = score
                best_params = params.copy()

        logger.debug(f"Trial {trial + 1}/{n_trials}: score={score:.4f}, params={params}")

    return OptimizationResult(
        best_params=best_params,
        best_score=best_score,
        total_trials=n_trials,
    )


def run_genetic(
    param_spaces: List[ParameterSpace],
    objective_fn: Callable[[Dict[str, Any]], float],
    n_trials: int,
    maximize: bool,
    history: List[Dict[str, Any]],
    **kwargs: Any,
) -> OptimizationResult:
    """Genetic Algorithm - 진화적 접근 (선택, 교차, mutation)."""
    population_size = kwargs.get("population_size", 20)
    mutation_rate = kwargs.get("mutation_rate", 0.1)
    crossover_rate = kwargs.get("crossover_rate", 0.7)

    population = [
        {space.name: space.sample() for space in param_spaces} for _ in range(population_size)
    ]

    best_params: Dict[str, Any] = {}
    best_score = float("-inf") if maximize else float("inf")
    generations = n_trials // population_size

    for _ in range(generations):
        scores: List[float] = []
        for individual in population:
            score = objective_fn(individual)
            history.append(
                {
                    "trial_num": len(history) + 1,
                    "params": individual.copy(),
                    "score": score,
                }
            )
            scores.append(score)

            if maximize:
                if score > best_score:
                    best_score = score
                    best_params = individual.copy()
            else:
                if score < best_score:
                    best_score = score
                    best_params = individual.copy()

        new_population = []
        for _ in range(population_size):
            tournament = random.sample(list(zip(population, scores)), k=3)
            winner = (
                max(tournament, key=lambda x: x[1])[0]
                if maximize
                else min(tournament, key=lambda x: x[1])[0]
            )
            new_population.append(winner.copy())

        for i in range(0, len(new_population) - 1, 2):
            if random.random() < crossover_rate:
                p1, p2 = new_population[i], new_population[i + 1]
                for param_name in p1:
                    if random.random() < 0.5:
                        p1[param_name], p2[param_name] = p2[param_name], p1[param_name]

        for individual in new_population:
            if random.random() < mutation_rate:
                param_to_mutate = random.choice(param_spaces)
                individual[param_to_mutate.name] = param_to_mutate.sample()

        population = new_population

    return OptimizationResult(
        best_params=best_params,
        best_score=best_score,
        total_trials=len(history),
    )
