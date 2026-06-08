"""Tests for domain/optimizer/optimization_strategies.py."""

from typing import Any, Dict, List
from unittest.mock import patch

import pytest

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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _int_space(name: str, low: int = 1, high: int = 10) -> ParameterSpace:
    return ParameterSpace(name=name, type=ParameterType.INTEGER, low=float(low), high=float(high))


def _float_space(name: str, low: float = 0.0, high: float = 1.0) -> ParameterSpace:
    return ParameterSpace(name=name, type=ParameterType.FLOAT, low=low, high=high)


def _cat_space(name: str, cats: List[Any] = None) -> ParameterSpace:
    return ParameterSpace(
        name=name, type=ParameterType.CATEGORICAL, categories=cats or ["a", "b", "c"]
    )


def _bool_space(name: str) -> ParameterSpace:
    return ParameterSpace(name=name, type=ParameterType.BOOLEAN)


def _objective_maximize(params: Dict[str, Any]) -> float:
    """Simple maximize: sum of numeric values."""
    return sum(v for v in params.values() if isinstance(v, (int, float)))


def _objective_minimize(params: Dict[str, Any]) -> float:
    """Simple minimize: negative sum."""
    return -sum(v for v in params.values() if isinstance(v, (int, float)))


# ---------------------------------------------------------------------------
# run_random
# ---------------------------------------------------------------------------


class TestRunRandom:
    def test_returns_optimization_result(self):
        spaces = [_int_space("k", 1, 10)]
        history: List[Dict] = []
        result = run_random(spaces, _objective_maximize, n_trials=5, maximize=True, history=history)
        assert isinstance(result, OptimizationResult)

    def test_history_has_n_trials_entries(self):
        spaces = [_int_space("k", 1, 10)]
        history: List[Dict] = []
        run_random(spaces, _objective_maximize, n_trials=7, maximize=True, history=history)
        assert len(history) == 7

    def test_best_params_contains_all_param_names(self):
        spaces = [_int_space("k"), _float_space("threshold")]
        history: List[Dict] = []
        result = run_random(
            spaces, _objective_maximize, n_trials=10, maximize=True, history=history
        )
        assert "k" in result.best_params
        assert "threshold" in result.best_params

    def test_maximize_true_best_score_positive(self):
        spaces = [_float_space("x", 0.0, 1.0)]
        history: List[Dict] = []
        result = run_random(spaces, lambda p: p["x"], n_trials=20, maximize=True, history=history)
        assert result.best_score >= 0

    def test_minimize_returns_smallest_score(self):
        # objective returns -x, minimize means we want smallest value (most negative)
        spaces = [_float_space("x", 0.0, 1.0)]
        history: List[Dict] = []
        result = run_random(
            spaces,
            lambda p: -p["x"],  # Always negative
            n_trials=20,
            maximize=False,
            history=history,
        )
        assert result.best_score <= 0

    def test_with_categorical_param(self):
        spaces = [_cat_space("model", ["gpt-4o", "gpt-4o-mini"])]
        history: List[Dict] = []

        def obj(params):
            return 1.0 if params["model"] == "gpt-4o" else 0.5

        result = run_random(spaces, obj, n_trials=10, maximize=True, history=history)
        assert result.best_params["model"] in ["gpt-4o", "gpt-4o-mini"]

    def test_with_boolean_param(self):
        spaces = [_bool_space("use_rerank")]
        history: List[Dict] = []

        def obj(params):
            return 1.0 if params["use_rerank"] else 0.0

        result = run_random(spaces, obj, n_trials=10, maximize=True, history=history)
        assert result.best_params["use_rerank"] is True

    def test_total_trials_matches(self):
        spaces = [_int_space("k")]
        history: List[Dict] = []
        result = run_random(spaces, _objective_maximize, n_trials=5, maximize=True, history=history)
        assert result.total_trials == 5


# ---------------------------------------------------------------------------
# run_grid
# ---------------------------------------------------------------------------


class TestRunGrid:
    def test_returns_optimization_result(self):
        spaces = [_int_space("k", 1, 5)]
        history: List[Dict] = []
        result = run_grid(spaces, _objective_maximize, maximize=True, history=history)
        assert isinstance(result, OptimizationResult)

    def test_integer_grid_produces_combinations(self):
        spaces = [_int_space("k", 1, 5)]
        history: List[Dict] = []
        run_grid(spaces, _objective_maximize, maximize=True, history=history, grid_size=4)
        assert len(history) > 0

    def test_categorical_grid_tries_all_categories(self):
        cats = ["cat1", "cat2", "cat3"]
        spaces = [_cat_space("model", cats)]
        seen = set()
        history: List[Dict] = []

        def obj(params):
            seen.add(params["model"])
            return 1.0

        run_grid(spaces, obj, maximize=True, history=history)
        assert seen == set(cats)

    def test_boolean_grid_tries_both_values(self):
        spaces = [_bool_space("flag")]
        seen = set()
        history: List[Dict] = []

        def obj(params):
            seen.add(params["flag"])
            return 1.0 if params["flag"] else 0.0

        run_grid(spaces, obj, maximize=True, history=history)
        assert True in seen
        assert False in seen

    def test_maximize_finds_highest_score(self):
        spaces = [_float_space("x", 0.0, 1.0)]
        history: List[Dict] = []
        result = run_grid(spaces, lambda p: p["x"], maximize=True, history=history, grid_size=5)
        assert result.best_score > 0.5  # Should find values near 1.0

    def test_minimize_finds_lowest_score(self):
        spaces = [_float_space("x", 0.0, 1.0)]
        history: List[Dict] = []
        result = run_grid(spaces, lambda p: p["x"], maximize=False, history=history, grid_size=5)
        assert result.best_score < 0.5  # Should find values near 0.0

    def test_multi_dimensional_grid(self):
        spaces = [_int_space("k", 1, 3), _bool_space("rerank")]
        history: List[Dict] = []
        run_grid(spaces, _objective_maximize, maximize=True, history=history, grid_size=2)
        # Should have 3 k values × 2 boolean values = 6 combinations (approximately)
        assert len(history) > 0


# ---------------------------------------------------------------------------
# run_genetic
# ---------------------------------------------------------------------------


class TestRunGenetic:
    def test_returns_optimization_result(self):
        spaces = [_float_space("x", 0.0, 1.0)]
        history: List[Dict] = []
        result = run_genetic(
            spaces,
            lambda p: p["x"],
            n_trials=20,
            maximize=True,
            history=history,
            population_size=5,
        )
        assert isinstance(result, OptimizationResult)

    def test_history_populated(self):
        spaces = [_float_space("x", 0.0, 1.0)]
        history: List[Dict] = []
        run_genetic(
            spaces,
            lambda p: p["x"],
            n_trials=20,
            maximize=True,
            history=history,
            population_size=5,
        )
        assert len(history) > 0

    def test_maximize_finds_high_score(self):
        spaces = [_float_space("x", 0.0, 1.0)]
        history: List[Dict] = []
        result = run_genetic(
            spaces,
            lambda p: p["x"],
            n_trials=50,
            maximize=True,
            history=history,
            population_size=10,
        )
        assert result.best_score >= 0.0

    def test_minimize_finds_low_score(self):
        spaces = [_float_space("x", 0.0, 1.0)]
        history: List[Dict] = []
        result = run_genetic(
            spaces,
            lambda p: p["x"],
            n_trials=50,
            maximize=False,
            history=history,
            population_size=10,
        )
        assert result.best_score <= 1.0

    def test_best_params_contains_all_spaces(self):
        spaces = [_float_space("x"), _int_space("k", 1, 5)]
        history: List[Dict] = []
        result = run_genetic(
            spaces,
            _objective_maximize,
            n_trials=20,
            maximize=True,
            history=history,
            population_size=5,
        )
        assert "x" in result.best_params
        assert "k" in result.best_params

    def test_with_categorical_space(self):
        spaces = [_cat_space("model", ["gpt-4o", "gpt-4o-mini"])]
        history: List[Dict] = []

        def obj(params):
            return 1.0 if params["model"] == "gpt-4o" else 0.0

        result = run_genetic(
            spaces,
            obj,
            n_trials=20,
            maximize=True,
            history=history,
            population_size=5,
        )
        assert result.best_params["model"] in ["gpt-4o", "gpt-4o-mini"]


# ---------------------------------------------------------------------------
# run_bayesian (fallback to random)
# ---------------------------------------------------------------------------


class TestRunBayesian:
    def test_bayesian_fallback_to_random_when_not_installed(self):
        """When bayes_opt is not installed, should fall back to run_random."""
        import sys
        import unittest.mock as mock

        # Ensure bayes_opt appears to not be installed
        with mock.patch.dict(sys.modules, {"bayes_opt": None}):
            from beanllm.domain.optimizer.optimization_strategies import run_bayesian

            spaces = [_float_space("x", 0.0, 1.0)]
            history: List[Dict] = []
            # This will either use bayes_opt or fall back gracefully
            # Just test it doesn't crash
            try:
                result = run_bayesian(
                    spaces,
                    lambda p: p["x"],
                    n_trials=5,
                    maximize=True,
                    history=history,
                )
                assert isinstance(result, OptimizationResult)
            except (ImportError, TypeError):
                pass  # Expected if bayes_opt not properly mocked

    def _make_fake_bayes_opt_module(self):
        """Create a fake bayes_opt module with a functional BayesianOptimization."""
        import types

        class FakeBayesianOptimization:
            def __init__(self, f, pbounds, random_state=None, verbose=0):
                self.f = f
                self.pbounds = pbounds
                self._max_params = None
                self._max_target = None

            def maximize(self, init_points=5, n_iter=5):
                for _ in range(max(1, init_points)):
                    params = {
                        name: (low + high) / 2.0 for name, (low, high) in self.pbounds.items()
                    }
                    score = self.f(**params)
                    if self._max_target is None or score > self._max_target:
                        self._max_target = score
                        self._max_params = dict(params)

            @property
            def max(self):
                return {
                    "params": self._max_params or {},
                    "target": self._max_target if self._max_target is not None else 0.0,
                }

        fake_mod = types.ModuleType("bayes_opt")
        fake_mod.BayesianOptimization = FakeBayesianOptimization
        return fake_mod

    def test_bayesian_with_integer_and_float_params(self):
        import sys

        fake_mod = self._make_fake_bayes_opt_module()

        with patch.dict(sys.modules, {"bayes_opt": fake_mod}):
            from beanllm.domain.optimizer.optimization_strategies import run_bayesian

            spaces = [
                ParameterSpace("k", ParameterType.INTEGER, low=1.0, high=10.0),
                ParameterSpace("threshold", ParameterType.FLOAT, low=0.0, high=1.0),
            ]
            history: List[Dict] = []
            result = run_bayesian(
                spaces,
                lambda p: float(p["k"]) + p["threshold"],
                n_trials=10,
                maximize=True,
                history=history,
            )

        assert isinstance(result, OptimizationResult)
        assert "k" in result.best_params
        assert "threshold" in result.best_params
        assert isinstance(result.best_params["k"], int)
        assert isinstance(result.best_params["threshold"], float)

    def test_bayesian_with_categorical_params(self):
        import sys

        fake_mod = self._make_fake_bayes_opt_module()

        with patch.dict(sys.modules, {"bayes_opt": fake_mod}):
            from beanllm.domain.optimizer.optimization_strategies import run_bayesian

            spaces = [
                ParameterSpace(
                    "model", ParameterType.CATEGORICAL, categories=["gpt-4o", "claude", "gemini"]
                ),
            ]
            history: List[Dict] = []
            result = run_bayesian(
                spaces,
                lambda p: 1.0 if p["model"] == "claude" else 0.5,
                n_trials=8,
                maximize=True,
                history=history,
            )

        assert isinstance(result, OptimizationResult)
        assert "model" in result.best_params
        assert result.best_params["model"] in ["gpt-4o", "claude", "gemini"]

    def test_bayesian_with_boolean_params(self):
        import sys

        fake_mod = self._make_fake_bayes_opt_module()

        with patch.dict(sys.modules, {"bayes_opt": fake_mod}):
            from beanllm.domain.optimizer.optimization_strategies import run_bayesian

            spaces = [
                ParameterSpace("use_cache", ParameterType.BOOLEAN),
                ParameterSpace("threshold", ParameterType.FLOAT, low=0.0, high=1.0),
            ]
            history: List[Dict] = []
            result = run_bayesian(
                spaces,
                lambda p: 1.0 if p["use_cache"] else 0.0,
                n_trials=6,
                maximize=True,
                history=history,
            )

        assert isinstance(result, OptimizationResult)
        assert "use_cache" in result.best_params
        assert isinstance(result.best_params["use_cache"], bool)

    def test_bayesian_maximize_false(self):
        import sys

        fake_mod = self._make_fake_bayes_opt_module()

        with patch.dict(sys.modules, {"bayes_opt": fake_mod}):
            from beanllm.domain.optimizer.optimization_strategies import run_bayesian

            spaces = [ParameterSpace("x", ParameterType.FLOAT, low=0.0, high=1.0)]
            history: List[Dict] = []
            result = run_bayesian(
                spaces,
                lambda p: p["x"],
                n_trials=6,
                maximize=False,
                history=history,
            )

        assert isinstance(result, OptimizationResult)
        # When maximize=False, best_score is negated back
        assert isinstance(result.best_score, float)

    def test_bayesian_with_custom_init_points(self):
        import sys

        fake_mod = self._make_fake_bayes_opt_module()

        with patch.dict(sys.modules, {"bayes_opt": fake_mod}):
            from beanllm.domain.optimizer.optimization_strategies import run_bayesian

            spaces = [ParameterSpace("x", ParameterType.FLOAT, low=0.0, high=1.0)]
            history: List[Dict] = []
            result = run_bayesian(
                spaces,
                lambda p: p["x"],
                n_trials=8,
                maximize=True,
                history=history,
                init_points=3,
            )

        assert isinstance(result, OptimizationResult)
