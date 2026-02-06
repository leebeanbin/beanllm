"""
ParameterSearch - 다목적 파라미터 탐색
SOLID 원칙:
- SRP: 파라미터 탐색만 담당
- OCP: 새로운 목적 함수 추가 가능
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Objective:
    """
    목적 함수 정의

    Attributes:
        name: 목적 함수 이름
        fn: 목적 함수 (params -> score)
        maximize: True면 최대화, False면 최소화
        weight: 가중치 (multi-objective 시)
    """

    name: str
    fn: Callable[[Dict[str, Any]], float]
    maximize: bool = True
    weight: float = 1.0


@dataclass
class SearchResult:
    """
    파라미터 탐색 결과

    Attributes:
        params: 파라미터
        scores: 목적 함수별 점수 {objective_name: score}
        combined_score: 결합 점수 (weighted sum)
        is_pareto_optimal: Pareto optimal 여부
    """

    params: Dict[str, Any]
    scores: Dict[str, float]
    combined_score: float = 0.0
    is_pareto_optimal: bool = False


@dataclass
class MultiObjectiveResult:
    """
    다목적 최적화 결과

    Attributes:
        results: 모든 탐색 결과
        pareto_frontier: Pareto optimal 결과들
        best_by_objective: 목적 함수별 최고 결과
        trade_offs: Trade-off 분석
    """

    results: List[SearchResult]
    pareto_frontier: List[SearchResult] = field(default_factory=list)
    best_by_objective: Dict[str, SearchResult] = field(default_factory=dict)
    trade_offs: Dict[str, Any] = field(default_factory=dict)


class ParameterSearch:
    """
    다목적 파라미터 탐색

    책임:
    - 다목적 최적화 (latency, quality, cost 동시 고려)
    - Pareto frontier 계산
    - Trade-off 분석

    Example:
        ```python
        search = ParameterSearch()

        # Define objectives
        objectives = [
            Objective(
                name="quality",
                fn=lambda params: evaluate_quality(params),
                maximize=True,
                weight=0.6
            ),
            Objective(
                name="latency",
                fn=lambda params: measure_latency(params),
                maximize=False,  # minimize
                weight=0.3
            ),
            Objective(
                name="cost",
                fn=lambda params: estimate_cost(params),
                maximize=False,  # minimize
                weight=0.1
            ),
        ]

        # Search
        result = search.multi_objective_search(
            param_spaces=param_spaces,
            objectives=objectives,
            n_trials=50
        )

        # Get Pareto optimal solutions
        for solution in result.pareto_frontier:
            print(f"Params: {solution.params}")
            print(f"Scores: {solution.scores}")

        # Analyze trade-offs
        print(result.trade_offs)
        ```
    """

    def __init__(self) -> None:
        """Initialize parameter search"""
        pass

    def multi_objective_search(
        self,
        param_spaces: List[Any],  # ParameterSpace from optimizer_engine
        objectives: List[Objective],
        n_trials: int = 50,
        method: str = "random",  # "random", "grid", "bayesian"
    ) -> MultiObjectiveResult:
        """
        다목적 최적화 탐색

        Args:
            param_spaces: 파라미터 공간 리스트
            objectives: 목적 함수 리스트
            n_trials: 시행 횟수
            method: 탐색 방법

        Returns:
            MultiObjectiveResult: 다목적 최적화 결과
        """
        logger.info(
            f"Starting multi-objective search: {len(objectives)} objectives, " f"{n_trials} trials"
        )

        results: List[SearchResult] = []

        # Sample parameters
        if method == "random":
            param_combinations = self._sample_random(param_spaces, n_trials)
        elif method == "grid":
            param_combinations = self._sample_grid(param_spaces)
        else:
            param_combinations = self._sample_random(param_spaces, n_trials)

        # Evaluate each combination
        for i, params in enumerate(param_combinations):
            scores = {}

            for objective in objectives:
                try:
                    score = objective.fn(params)
                    scores[objective.name] = score
                except Exception as e:
                    logger.error(f"Error evaluating {objective.name}: {e}")
                    scores[objective.name] = 0.0

            # Calculate combined score (weighted sum)
            combined_score = 0.0
            for objective in objectives:
                score = scores[objective.name]

                # Normalize: maximize -> positive, minimize -> negative
                if objective.maximize:
                    normalized_score = score
                else:
                    normalized_score = -score

                combined_score += objective.weight * normalized_score

            results.append(
                SearchResult(
                    params=params,
                    scores=scores,
                    combined_score=combined_score,
                )
            )

            if (i + 1) % 10 == 0:
                logger.debug(f"Evaluated {i + 1}/{len(param_combinations)} combinations")

        # Calculate Pareto frontier
        pareto_frontier = self._calculate_pareto_frontier(results, objectives)

        # Find best for each objective
        best_by_objective = {}
        for objective in objectives:
            if objective.maximize:
                best = max(results, key=lambda r: r.scores[objective.name])
            else:
                best = min(results, key=lambda r: r.scores[objective.name])

            best_by_objective[objective.name] = best

        # Analyze trade-offs
        trade_offs = self._analyze_trade_offs(results, objectives)

        result = MultiObjectiveResult(
            results=results,
            pareto_frontier=pareto_frontier,
            best_by_objective=best_by_objective,
            trade_offs=trade_offs,
        )

        logger.info(
            f"Multi-objective search completed: "
            f"{len(pareto_frontier)} Pareto optimal solutions found"
        )

        return result

    def _sample_random(self, param_spaces: List[Any], n_trials: int) -> List[Dict[str, Any]]:
        """랜덤 샘플링"""
        combinations = []

        for _ in range(n_trials):
            params = {space.name: space.sample() for space in param_spaces}
            combinations.append(params)

        return combinations

    def _sample_grid(self, param_spaces: List[Any]) -> List[Dict[str, Any]]:
        """Grid 샘플링"""
        import itertools

        # Generate grid for each parameter
        param_grids = {}

        for space in param_spaces:
            if space.type.value == "integer":
                grid_size = min(5, space.high - space.low + 1)
                step = max(1, (space.high - space.low) // grid_size)
                param_grids[space.name] = list(range(int(space.low), int(space.high) + 1, step))
            elif space.type.value == "float":
                grid_size = 5
                step = (space.high - space.low) / grid_size
                param_grids[space.name] = [space.low + i * step for i in range(grid_size + 1)]
            elif space.type.value == "categorical":
                param_grids[space.name] = space.categories
            elif space.type.value == "boolean":
                param_grids[space.name] = [True, False]

        # Generate all combinations
        param_names = list(param_grids.keys())
        param_values = list(param_grids.values())

        combinations = []
        for combination in itertools.product(*param_values):
            params = dict(zip(param_names, combination))
            combinations.append(params)

        return combinations

    def _calculate_pareto_frontier(
        self, results: List[SearchResult], objectives: List[Objective]
    ) -> List[SearchResult]:
        """
        Pareto frontier 계산

        A solution is Pareto optimal if no other solution dominates it.
        Solution A dominates B if A is better than B in at least one objective
        and not worse in any objective.
        """
        pareto_frontier = []

        for candidate in results:
            is_dominated = False

            for other in results:
                if candidate == other:
                    continue

                # Check if 'other' dominates 'candidate'
                dominates = True
                strictly_better_in_at_least_one = False

                for objective in objectives:
                    candidate_score = candidate.scores[objective.name]
                    other_score = other.scores[objective.name]

                    if objective.maximize:
                        if other_score < candidate_score:
                            dominates = False
                            break
                        elif other_score > candidate_score:
                            strictly_better_in_at_least_one = True
                    else:  # minimize
                        if other_score > candidate_score:
                            dominates = False
                            break
                        elif other_score < candidate_score:
                            strictly_better_in_at_least_one = True

                if dominates and strictly_better_in_at_least_one:
                    is_dominated = True
                    break

            if not is_dominated:
                candidate.is_pareto_optimal = True
                pareto_frontier.append(candidate)

        return pareto_frontier

    def _analyze_trade_offs(
        self, results: List[SearchResult], objectives: List[Objective]
    ) -> Dict[str, Any]:
        """Trade-off 분석"""
        # Calculate correlation between objectives

        trade_offs = {}

        if len(objectives) < 2:
            return trade_offs

        # Pairwise correlations
        for i, obj1 in enumerate(objectives):
            for j, obj2 in enumerate(objectives):
                if i >= j:
                    continue

                scores1 = [r.scores[obj1.name] for r in results]
                scores2 = [r.scores[obj2.name] for r in results]

                # Calculate correlation
                correlation = self._calculate_correlation(scores1, scores2)

                trade_offs[f"{obj1.name}_vs_{obj2.name}"] = {
                    "correlation": correlation,
                    "interpretation": self._interpret_correlation(
                        correlation, obj1.maximize, obj2.maximize
                    ),
                }

        return trade_offs

    def _calculate_correlation(self, scores1: List[float], scores2: List[float]) -> float:
        """Pearson correlation 계산"""
        import statistics

        if len(scores1) < 2 or len(scores2) < 2:
            return 0.0

        mean1 = statistics.mean(scores1)
        mean2 = statistics.mean(scores2)

        numerator = sum((x - mean1) * (y - mean2) for x, y in zip(scores1, scores2))

        denom1 = sum((x - mean1) ** 2 for x in scores1) ** 0.5
        denom2 = sum((y - mean2) ** 2 for y in scores2) ** 0.5

        if denom1 == 0 or denom2 == 0:
            return 0.0

        return numerator / (denom1 * denom2)

    def _interpret_correlation(self, correlation: float, maximize1: bool, maximize2: bool) -> str:
        """Correlation 해석"""
        abs_corr = abs(correlation)

        if abs_corr < 0.3:
            strength = "weak"
        elif abs_corr < 0.7:
            strength = "moderate"
        else:
            strength = "strong"

        # Both maximize or both minimize: positive correlation is synergy
        # One maximize, one minimize: negative correlation is synergy
        if maximize1 == maximize2:
            if correlation > 0:
                relationship = "synergy (improving one improves the other)"
            else:
                relationship = "trade-off (improving one worsens the other)"
        else:
            if correlation < 0:
                relationship = "synergy (improving one improves the other)"
            else:
                relationship = "trade-off (improving one worsens the other)"

        return f"{strength} {relationship}"


def find_balanced_solution(
    result: MultiObjectiveResult,
    objectives: List[Objective],
) -> SearchResult:
    """
    균형잡힌 솔루션 찾기 (Pareto frontier 중에서)

    Args:
        result: 다목적 최적화 결과
        objectives: 목적 함수 리스트

    Returns:
        SearchResult: 균형잡힌 솔루션
    """
    if not result.pareto_frontier:
        return result.results[0] if result.results else None

    # Normalize scores and find closest to ideal point
    best_scores = {}

    for objective in objectives:
        if objective.maximize:
            best_scores[objective.name] = max(r.scores[objective.name] for r in result.results)
        else:
            best_scores[objective.name] = min(r.scores[objective.name] for r in result.results)

    # Calculate distance from ideal point for each Pareto optimal solution
    min_distance = float("inf")
    balanced_solution = None

    for solution in result.pareto_frontier:
        distance = 0.0

        for objective in objectives:
            score = solution.scores[objective.name]
            best = best_scores[objective.name]

            # Normalize
            if objective.maximize:
                normalized = score / best if best > 0 else 0
            else:
                normalized = best / score if score > 0 else 0

            # Distance from ideal (1.0)
            distance += (1.0 - normalized) ** 2

        distance = distance**0.5

        if distance < min_distance:
            min_distance = distance
            balanced_solution = solution

    return balanced_solution
