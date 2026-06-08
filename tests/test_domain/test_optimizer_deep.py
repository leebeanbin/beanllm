"""
Comprehensive pytest tests for beanllm optimizer domain.

Covers:
- profiler.py: Profiler, ProfileResult, ProfileContext, ComponentMetrics, ComponentType
- ab_tester.py: ABTester, ABTestResult, compare_multiple_variants
- optimization_strategies.py: run_random, run_grid, run_genetic
- parameter_management.py: ParameterSpace, ParameterType, OptimizationResult (used as deps)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

# ── A/B Tester imports ──────────────────────────────────────────────────────
from beanllm.domain.optimizer.ab_tester import (
    ABTester,
    ABTestResult,
    compare_multiple_variants,
)

# ── Optimization strategy imports ───────────────────────────────────────────
from beanllm.domain.optimizer.optimization_strategies import (
    run_genetic,
    run_grid,
    run_random,
)

# ── Parameter management imports ────────────────────────────────────────────
from beanllm.domain.optimizer.parameter_management import (
    OptimizationResult,
    ParameterSpace,
    ParameterType,
)

# ── Profiler imports ────────────────────────────────────────────────────────
from beanllm.domain.optimizer.profiler import (
    ComponentMetrics,
    ComponentType,
    ProfileContext,
    Profiler,
    ProfileResult,
    profile_rag_pipeline,
)

# ===========================================================================
# Helpers / shared fixtures
# ===========================================================================


def _make_int_space(name: str = "k", low: float = 1, high: float = 10) -> ParameterSpace:
    return ParameterSpace(name=name, type=ParameterType.INTEGER, low=low, high=high)


def _make_float_space(name: str = "temp", low: float = 0.0, high: float = 1.0) -> ParameterSpace:
    return ParameterSpace(name=name, type=ParameterType.FLOAT, low=low, high=high)


def _make_cat_space(name: str = "model", categories: List[Any] = None) -> ParameterSpace:
    cats = categories or ["gpt-4o", "claude-3", "gemini"]
    return ParameterSpace(name=name, type=ParameterType.CATEGORICAL, categories=cats)


def _make_bool_space(name: str = "use_cache") -> ParameterSpace:
    return ParameterSpace(name=name, type=ParameterType.BOOLEAN)


def _simple_objective(params: Dict[str, Any]) -> float:
    """Objective: penalize large k, reward high temp."""
    k = params.get("k", 5)
    return 1.0 / (k + 1)


# ===========================================================================
# ComponentType tests
# ===========================================================================


class TestComponentType:
    def test_all_enum_values_exist(self) -> None:
        expected = {
            "EMBEDDING",
            "RETRIEVAL",
            "RERANKING",
            "GENERATION",
            "PREPROCESSING",
            "POSTPROCESSING",
            "TOTAL",
        }
        actual = {e.name for e in ComponentType}
        assert expected == actual

    def test_embedding_value(self) -> None:
        assert ComponentType.EMBEDDING.value == "embedding"

    def test_retrieval_value(self) -> None:
        assert ComponentType.RETRIEVAL.value == "retrieval"

    def test_generation_value(self) -> None:
        assert ComponentType.GENERATION.value == "generation"

    def test_total_value(self) -> None:
        assert ComponentType.TOTAL.value == "total"


# ===========================================================================
# ComponentMetrics tests
# ===========================================================================


class TestComponentMetrics:
    def test_default_values(self) -> None:
        m = ComponentMetrics(component_type=ComponentType.EMBEDDING)
        assert m.duration_ms == 0.0
        assert m.memory_mb == 0.0
        assert m.token_count == 0
        assert m.estimated_cost == 0.0
        assert m.metadata == {}

    def test_explicit_values(self) -> None:
        m = ComponentMetrics(
            component_type=ComponentType.GENERATION,
            duration_ms=200.5,
            memory_mb=128.0,
            token_count=500,
            estimated_cost=0.015,
            metadata={"model": "gpt-4o"},
        )
        assert m.duration_ms == 200.5
        assert m.memory_mb == 128.0
        assert m.token_count == 500
        assert m.estimated_cost == 0.015
        assert m.metadata["model"] == "gpt-4o"


# ===========================================================================
# ProfileResult tests
# ===========================================================================


class TestProfileResult:
    def test_empty_result(self) -> None:
        result = ProfileResult()
        assert result.total_duration_ms == 0.0
        assert result.total_cost == 0.0
        assert result.bottleneck is None
        assert result.recommendations == []

    def test_post_init_totals(self) -> None:
        components = {
            "embed": ComponentMetrics(
                component_type=ComponentType.EMBEDDING,
                duration_ms=100.0,
                estimated_cost=0.01,
            ),
            "retrieve": ComponentMetrics(
                component_type=ComponentType.RETRIEVAL,
                duration_ms=200.0,
                estimated_cost=0.0,
            ),
        }
        result = ProfileResult(components=components)
        assert result.total_duration_ms == pytest.approx(300.0)
        assert result.total_cost == pytest.approx(0.01)

    def test_post_init_bottleneck_is_slowest_component(self) -> None:
        components = {
            "fast": ComponentMetrics(component_type=ComponentType.EMBEDDING, duration_ms=50.0),
            "slow": ComponentMetrics(component_type=ComponentType.GENERATION, duration_ms=500.0),
        }
        result = ProfileResult(components=components)
        assert result.bottleneck == ComponentType.GENERATION

    def test_get_breakdown_empty(self) -> None:
        result = ProfileResult()
        assert result.get_breakdown() == {}

    def test_get_breakdown_percentages_sum_to_100(self) -> None:
        components = {
            "a": ComponentMetrics(component_type=ComponentType.EMBEDDING, duration_ms=30.0),
            "b": ComponentMetrics(component_type=ComponentType.RETRIEVAL, duration_ms=70.0),
        }
        result = ProfileResult(components=components)
        breakdown = result.get_breakdown()
        assert breakdown["a"] == pytest.approx(30.0)
        assert breakdown["b"] == pytest.approx(70.0)
        assert sum(breakdown.values()) == pytest.approx(100.0)

    def test_get_breakdown_single_component(self) -> None:
        components = {
            "only": ComponentMetrics(component_type=ComponentType.TOTAL, duration_ms=400.0)
        }
        result = ProfileResult(components=components)
        breakdown = result.get_breakdown()
        assert breakdown["only"] == pytest.approx(100.0)

    def test_bottleneck_when_single_component(self) -> None:
        components = {
            "solo": ComponentMetrics(component_type=ComponentType.PREPROCESSING, duration_ms=111.0)
        }
        result = ProfileResult(components=components)
        assert result.bottleneck == ComponentType.PREPROCESSING


# ===========================================================================
# Profiler tests
# ===========================================================================


class TestProfilerStartEnd:
    def test_start_records_start_time(self) -> None:
        profiler = Profiler()
        profiler.start("comp_a")
        assert "comp_a" in profiler._start_times
        assert "comp_a" in profiler._active_profiles

    def test_end_returns_metrics(self) -> None:
        profiler = Profiler()
        profiler.start("comp_a")
        time.sleep(0.01)  # ensure measurable duration
        metrics = profiler.end("comp_a")
        assert isinstance(metrics, ComponentMetrics)
        assert metrics.duration_ms > 0

    def test_end_without_start_returns_default(self) -> None:
        profiler = Profiler()
        metrics = profiler.end("ghost")
        assert metrics.component_type == ComponentType.TOTAL

    def test_end_removes_from_active_profiles(self) -> None:
        profiler = Profiler()
        profiler.start("comp_a")
        profiler.end("comp_a")
        assert "comp_a" not in profiler._active_profiles

    def test_multiple_components(self) -> None:
        profiler = Profiler()
        for name in ("embedding", "retrieval", "generation"):
            profiler.start(name)
            profiler.end(name)
        result = profiler.get_result()
        assert len(result.components) == 3


class TestProfilerContextManager:
    def test_basic_context_manager(self) -> None:
        profiler = Profiler()
        with profiler.profile("embed_ctx"):
            pass
        assert "embed_ctx" in profiler._metrics

    def test_context_manager_returns_profile_context(self) -> None:
        profiler = Profiler()
        ctx = profiler.profile("my_comp")
        assert isinstance(ctx, ProfileContext)

    def test_context_manager_sets_metrics_after_exit(self) -> None:
        profiler = Profiler()
        with profiler.profile("embed_ctx") as ctx:
            pass
        assert ctx.metrics is not None
        assert isinstance(ctx.metrics, ComponentMetrics)

    def test_context_manager_set_tokens(self) -> None:
        # set_tokens must be called after the context exits (after profiler.end() stores
        # the metrics object in _metrics), so we call it on the returned context.
        profiler = Profiler()
        with profiler.profile("generation") as ctx:
            pass
        ctx.set_tokens(1000)
        assert profiler._metrics["generation"].token_count == 1000
        assert profiler._metrics["generation"].estimated_cost > 0

    def test_context_manager_set_memory(self) -> None:
        # set_memory must be called after the context exits so that the component
        # entry already exists in profiler._metrics.
        profiler = Profiler()
        with profiler.profile("embedding") as ctx:
            pass
        ctx.set_memory(256.0)
        assert profiler._metrics["embedding"].memory_mb == 256.0


class TestProfilerSetTokensSetMemory:
    def test_set_tokens_calculates_cost(self) -> None:
        profiler = Profiler()
        profiler.start("gen")
        profiler.end("gen")
        profiler.set_tokens("gen", 2000)
        assert profiler._metrics["gen"].token_count == 2000
        # cost = 2000 / 1000 * 0.03
        assert profiler._metrics["gen"].estimated_cost == pytest.approx(0.06)

    def test_set_tokens_on_unknown_component_is_no_op(self) -> None:
        profiler = Profiler()
        profiler.set_tokens("nonexistent", 100)  # should not raise

    def test_set_memory_on_known_component(self) -> None:
        profiler = Profiler()
        profiler.start("retrieve")
        profiler.end("retrieve")
        profiler.set_memory("retrieve", 512.0)
        assert profiler._metrics["retrieve"].memory_mb == 512.0

    def test_set_memory_on_unknown_component_is_no_op(self) -> None:
        profiler = Profiler()
        profiler.set_memory("ghost", 100.0)  # should not raise


class TestProfilerReset:
    def test_reset_clears_all_state(self) -> None:
        profiler = Profiler()
        profiler.start("comp")
        profiler.end("comp")
        profiler.reset()
        assert profiler._metrics == {}
        assert profiler._start_times == {}
        assert profiler._active_profiles == []


class TestProfilerGetResult:
    def test_get_result_returns_profile_result(self) -> None:
        profiler = Profiler()
        profiler.start("embed")
        profiler.end("embed")
        result = profiler.get_result()
        assert isinstance(result, ProfileResult)

    def test_get_result_includes_recommendations(self) -> None:
        profiler = Profiler()
        profiler.start("embed")
        profiler.end("embed")
        result = profiler.get_result()
        assert isinstance(result.recommendations, list)


class TestProfilerInferComponentType:
    def setup_method(self) -> None:
        self.profiler = Profiler()

    def _infer(self, name: str) -> ComponentType:
        return self.profiler._infer_component_type(name)

    def test_infer_embedding(self) -> None:
        assert self._infer("embed_docs") == ComponentType.EMBEDDING
        assert self._infer("EmbedQuery") == ComponentType.EMBEDDING

    def test_infer_retrieval_search(self) -> None:
        assert self._infer("retrieval_step") == ComponentType.RETRIEVAL
        assert self._infer("similarity_search") == ComponentType.RETRIEVAL

    def test_infer_reranking(self) -> None:
        assert self._infer("rerank_results") == ComponentType.RERANKING

    def test_infer_generation(self) -> None:
        assert self._infer("generate_answer") == ComponentType.GENERATION
        assert self._infer("llm_call") == ComponentType.GENERATION

    def test_infer_preprocessing(self) -> None:
        assert self._infer("preprocess_text") == ComponentType.PREPROCESSING

    def test_infer_postprocessing(self) -> None:
        assert self._infer("postprocess_output") == ComponentType.POSTPROCESSING

    def test_infer_total(self) -> None:
        assert self._infer("total") == ComponentType.TOTAL
        assert self._infer("TOTAL") == ComponentType.TOTAL

    def test_infer_unknown_falls_back_to_total(self) -> None:
        assert self._infer("unknown_component_xyz") == ComponentType.TOTAL


class TestProfilerGenerateRecommendations:
    def test_recommendation_for_slow_embedding(self) -> None:
        profiler = Profiler()
        components = {
            "embedding": ComponentMetrics(
                component_type=ComponentType.EMBEDDING, duration_ms=900.0
            ),
            "other": ComponentMetrics(component_type=ComponentType.RETRIEVAL, duration_ms=100.0),
        }
        result = ProfileResult(components=components)
        recs = profiler._generate_recommendations(result)
        assert any("Embedding" in r for r in recs)

    def test_recommendation_for_slow_retrieval(self) -> None:
        profiler = Profiler()
        components = {
            "retrieval": ComponentMetrics(
                component_type=ComponentType.RETRIEVAL, duration_ms=900.0
            ),
            "other": ComponentMetrics(component_type=ComponentType.EMBEDDING, duration_ms=100.0),
        }
        result = ProfileResult(components=components)
        recs = profiler._generate_recommendations(result)
        assert any("Retrieval" in r for r in recs)

    def test_recommendation_for_slow_generation(self) -> None:
        profiler = Profiler()
        components = {
            "generation": ComponentMetrics(
                component_type=ComponentType.GENERATION, duration_ms=900.0
            ),
            "other": ComponentMetrics(component_type=ComponentType.EMBEDDING, duration_ms=100.0),
        }
        result = ProfileResult(components=components)
        recs = profiler._generate_recommendations(result)
        assert any("Generation" in r for r in recs)

    def test_recommendation_for_slow_reranking(self) -> None:
        profiler = Profiler()
        components = {
            "reranking": ComponentMetrics(
                component_type=ComponentType.RERANKING, duration_ms=900.0
            ),
            "other": ComponentMetrics(component_type=ComponentType.EMBEDDING, duration_ms=100.0),
        }
        result = ProfileResult(components=components)
        recs = profiler._generate_recommendations(result)
        assert any("Reranking" in r for r in recs)

    def test_recommendation_for_high_cost(self) -> None:
        profiler = Profiler()
        components = {
            "gen": ComponentMetrics(
                component_type=ComponentType.GENERATION,
                duration_ms=100.0,
                estimated_cost=0.50,
            )
        }
        result = ProfileResult(components=components)
        recs = profiler._generate_recommendations(result)
        assert any("cost" in r.lower() for r in recs)

    def test_recommendation_for_high_latency(self) -> None:
        profiler = Profiler()
        components = {
            "total": ComponentMetrics(component_type=ComponentType.TOTAL, duration_ms=6000.0)
        }
        result = ProfileResult(components=components)
        recs = profiler._generate_recommendations(result)
        assert any("latency" in r.lower() for r in recs)

    def test_no_recommendations_for_fast_cheap_pipeline(self) -> None:
        # Each component occupies exactly 50% of total time, which is above the
        # 40% threshold in _generate_recommendations, so both DO produce recs.
        # Use a 3-component mix where no single component exceeds 40% and cost is low.
        profiler = Profiler()
        components = {
            "embed": ComponentMetrics(
                component_type=ComponentType.EMBEDDING,
                duration_ms=30.0,
                estimated_cost=0.001,
            ),
            "retrieve": ComponentMetrics(component_type=ComponentType.RETRIEVAL, duration_ms=40.0),
            "generate": ComponentMetrics(component_type=ComponentType.GENERATION, duration_ms=30.0),
        }
        result = ProfileResult(components=components)
        recs = profiler._generate_recommendations(result)
        assert recs == []


class TestProfileRagPipeline:
    def test_profile_rag_pipeline_returns_profile_result(self) -> None:
        def dummy_rag(query: str) -> str:
            return f"Answer to: {query}"

        result = profile_rag_pipeline(dummy_rag, "test query")
        assert isinstance(result, ProfileResult)
        assert "total" in result.components

    def test_profile_rag_pipeline_measures_time(self) -> None:
        def slow_rag(query: str) -> str:
            time.sleep(0.05)
            return "answer"

        result = profile_rag_pipeline(slow_rag, "test")
        assert result.total_duration_ms >= 40  # at least 40ms


# ===========================================================================
# ABTestResult tests
# ===========================================================================


class TestABTestResult:
    def test_basic_creation(self) -> None:
        result = ABTestResult(
            variant_a_name="A",
            variant_b_name="B",
            variant_a_mean=0.70,
            variant_b_mean=0.80,
        )
        assert result.variant_a_name == "A"
        assert result.variant_b_mean == 0.80

    def test_post_init_lift_calculation(self) -> None:
        result = ABTestResult(
            variant_a_name="A",
            variant_b_name="B",
            variant_a_mean=0.80,
            variant_b_mean=0.90,
        )
        expected_lift = (0.90 - 0.80) / 0.80 * 100
        assert result.lift == pytest.approx(expected_lift)

    def test_post_init_lift_zero_when_a_mean_zero(self) -> None:
        result = ABTestResult(
            variant_a_name="A",
            variant_b_name="B",
            variant_a_mean=0.0,
            variant_b_mean=0.5,
        )
        # lift stays 0.0 because variant_a_mean == 0
        assert result.lift == pytest.approx(0.0)

    def test_post_init_winner_b_when_significant_and_b_higher(self) -> None:
        result = ABTestResult(
            variant_a_name="A",
            variant_b_name="B",
            variant_a_mean=0.60,
            variant_b_mean=0.80,
            is_significant=True,
        )
        assert result.winner == "B"

    def test_post_init_winner_a_when_significant_and_a_higher(self) -> None:
        result = ABTestResult(
            variant_a_name="A",
            variant_b_name="B",
            variant_a_mean=0.90,
            variant_b_mean=0.60,
            is_significant=True,
        )
        assert result.winner == "A"

    def test_post_init_winner_tie_when_not_significant(self) -> None:
        result = ABTestResult(
            variant_a_name="A",
            variant_b_name="B",
            variant_a_mean=0.70,
            variant_b_mean=0.75,
            is_significant=False,
        )
        assert result.winner == "tie"

    def test_default_confidence_level(self) -> None:
        result = ABTestResult(
            variant_a_name="A",
            variant_b_name="B",
            variant_a_mean=0.5,
            variant_b_mean=0.6,
        )
        assert result.confidence_level == 0.95

    def test_sample_sizes_stored(self) -> None:
        result = ABTestResult(
            variant_a_name="A",
            variant_b_name="B",
            variant_a_mean=0.5,
            variant_b_mean=0.6,
            sample_size_a=50,
            sample_size_b=50,
        )
        assert result.sample_size_a == 50
        assert result.sample_size_b == 50


# ===========================================================================
# ABTester tests
# ===========================================================================


class TestABTester:
    def setup_method(self) -> None:
        self.tester = ABTester()

    def _make_const_fn(self, value: float):
        return lambda q: value

    def _identity_eval(self, result: float) -> float:
        return result

    def test_run_test_returns_ab_test_result(self) -> None:
        queries = [1, 2, 3, 4, 5]
        result = self.tester.run_test(
            variant_a=self._make_const_fn(0.7),
            variant_b=self._make_const_fn(0.8),
            evaluation_fn=self._identity_eval,
            queries=queries,
        )
        assert isinstance(result, ABTestResult)

    def test_run_test_correct_means(self) -> None:
        queries = list(range(10))
        result = self.tester.run_test(
            variant_a=self._make_const_fn(0.5),
            variant_b=self._make_const_fn(0.9),
            evaluation_fn=self._identity_eval,
            queries=queries,
            variant_a_name="Baseline",
            variant_b_name="New",
        )
        assert result.variant_a_mean == pytest.approx(0.5)
        assert result.variant_b_mean == pytest.approx(0.9)
        assert result.variant_a_name == "Baseline"
        assert result.variant_b_name == "New"

    def test_run_test_sample_sizes(self) -> None:
        queries = list(range(15))
        result = self.tester.run_test(
            variant_a=self._make_const_fn(0.6),
            variant_b=self._make_const_fn(0.7),
            evaluation_fn=self._identity_eval,
            queries=queries,
        )
        assert result.sample_size_a == 15
        assert result.sample_size_b == 15

    def test_run_test_variant_error_records_zero(self) -> None:
        def error_variant(q: Any) -> float:
            raise RuntimeError("boom")

        queries = list(range(5))
        result = self.tester.run_test(
            variant_a=error_variant,
            variant_b=self._make_const_fn(0.8),
            evaluation_fn=self._identity_eval,
            queries=queries,
        )
        assert result.variant_a_mean == pytest.approx(0.0)

    def test_run_test_stores_correct_variant_names(self) -> None:
        queries = [1]
        result = self.tester.run_test(
            variant_a=self._make_const_fn(0.5),
            variant_b=self._make_const_fn(0.6),
            evaluation_fn=self._identity_eval,
            queries=queries,
            variant_a_name="Alpha",
            variant_b_name="Beta",
        )
        assert result.variant_a_name == "Alpha"
        assert result.variant_b_name == "Beta"

    def test_t_test_identical_scores_returns_high_p(self) -> None:
        scores = [0.7] * 10
        p = self.tester._t_test(scores, scores)
        # identical => p should be 1.0 (pooled_se is 0)
        assert p == pytest.approx(1.0)

    def test_t_test_very_different_scores_returns_low_p(self) -> None:
        # Use non-constant, clearly separated samples so pooled_se > 0
        # and df > 30 triggers the normal approximation path.
        import random as _rnd

        _rnd.seed(0)
        a = [0.10 + _rnd.uniform(0, 0.05) for _ in range(40)]
        b = [0.90 + _rnd.uniform(0, 0.05) for _ in range(40)]
        p = self.tester._t_test(a, b)
        # The two-tailed p-value should be well below 0.05
        assert p < 0.05

    def test_t_test_single_sample_returns_1(self) -> None:
        assert self.tester._t_test([0.5], [0.5]) == 1.0

    def test_calculate_required_sample_size_is_positive_int(self) -> None:
        n = self.tester.calculate_required_sample_size(
            baseline_mean=0.75,
            baseline_std=0.15,
            minimum_detectable_effect=5.0,
        )
        assert isinstance(n, int)
        assert n > 0

    def test_calculate_required_sample_size_larger_effect_smaller_n(self) -> None:
        n_small_effect = self.tester.calculate_required_sample_size(
            baseline_mean=0.75, baseline_std=0.15, minimum_detectable_effect=1.0
        )
        n_large_effect = self.tester.calculate_required_sample_size(
            baseline_mean=0.75, baseline_std=0.15, minimum_detectable_effect=20.0
        )
        assert n_large_effect < n_small_effect

    def test_run_test_custom_confidence_level(self) -> None:
        queries = list(range(5))
        result = self.tester.run_test(
            variant_a=self._make_const_fn(0.5),
            variant_b=self._make_const_fn(0.6),
            evaluation_fn=self._identity_eval,
            queries=queries,
            confidence_level=0.90,
        )
        assert result.confidence_level == 0.90

    def test_run_test_significance_detected_with_identical_values(self) -> None:
        """Identical scores → not significant."""
        queries = list(range(10))
        result = self.tester.run_test(
            variant_a=self._make_const_fn(0.7),
            variant_b=self._make_const_fn(0.7),
            evaluation_fn=self._identity_eval,
            queries=queries,
        )
        # p_value is 1.0, not significant
        assert result.p_value == pytest.approx(1.0)
        assert result.is_significant is False
        assert result.winner == "tie"


class TestABTesterNormalApproximation:
    def test_normal_p_value_positive_z(self) -> None:
        tester = ABTester()
        p = tester._normal_distribution_p_value(2.0)
        # z=2, p ≈ 0.023 (one-tailed)
        assert 0.0 < p < 0.05

    def test_normal_p_value_zero_z(self) -> None:
        tester = ABTester()
        p = tester._normal_distribution_p_value(0.0)
        assert p == pytest.approx(0.5, abs=0.01)

    def test_t_distribution_large_df_uses_normal_approx(self) -> None:
        tester = ABTester()
        # df > 30 triggers normal approximation
        p = tester._t_distribution_p_value(2.5, df=50)
        assert 0 < p < 1

    def test_t_distribution_small_df_uses_lookup(self) -> None:
        tester = ABTester()
        # t_stat > critical_value → p ≈ 0.01
        p_sig = tester._t_distribution_p_value(3.0, df=10)
        assert p_sig == pytest.approx(0.01)
        # t_stat < critical_value → p ≈ 0.10
        p_not_sig = tester._t_distribution_p_value(1.0, df=10)
        assert p_not_sig == pytest.approx(0.10)


# ===========================================================================
# compare_multiple_variants tests
# ===========================================================================


class TestCompareMultipleVariants:
    def test_returns_all_pairwise_comparisons(self) -> None:
        variants = {
            "v1": lambda q: 0.6,
            "v2": lambda q: 0.7,
            "v3": lambda q: 0.8,
        }
        queries = list(range(5))
        results = compare_multiple_variants(
            variants=variants, evaluation_fn=lambda x: x, queries=queries
        )
        assert "v1_vs_v2" in results
        assert "v1_vs_v3" in results
        assert "v2_vs_v3" in results
        assert len(results) == 3

    def test_results_are_ab_test_result_instances(self) -> None:
        variants = {"a": lambda q: 0.5, "b": lambda q: 0.6}
        results = compare_multiple_variants(
            variants=variants, evaluation_fn=lambda x: x, queries=[1, 2, 3]
        )
        for r in results.values():
            assert isinstance(r, ABTestResult)

    def test_two_variants_produces_one_comparison(self) -> None:
        variants = {"x": lambda q: 0.4, "y": lambda q: 0.8}
        results = compare_multiple_variants(
            variants=variants, evaluation_fn=lambda x: x, queries=[1, 2, 3]
        )
        assert len(results) == 1
        assert "x_vs_y" in results


# ===========================================================================
# run_random tests
# ===========================================================================


class TestRunRandom:
    def _spaces(self) -> List[ParameterSpace]:
        return [_make_int_space("k", 1, 10)]

    def test_returns_optimization_result(self) -> None:
        history: List[Dict] = []
        result = run_random(
            param_spaces=self._spaces(),
            objective_fn=lambda p: float(p["k"]),
            n_trials=5,
            maximize=True,
            history=history,
        )
        assert isinstance(result, OptimizationResult)

    def test_correct_number_of_trials(self) -> None:
        history: List[Dict] = []
        run_random(
            param_spaces=self._spaces(),
            objective_fn=lambda p: 1.0,
            n_trials=8,
            maximize=True,
            history=history,
        )
        assert len(history) == 8

    def test_maximize_finds_better_score(self) -> None:
        history: List[Dict] = []
        result = run_random(
            param_spaces=[_make_int_space("k", 1, 100)],
            objective_fn=lambda p: float(p["k"]),
            n_trials=50,
            maximize=True,
            history=history,
        )
        # With 50 trials maximizing k, best should be well above 1
        assert result.best_score > 1.0

    def test_minimize_finds_lower_score(self) -> None:
        history: List[Dict] = []
        result = run_random(
            param_spaces=[_make_int_space("k", 1, 100)],
            objective_fn=lambda p: float(p["k"]),
            n_trials=50,
            maximize=False,
            history=history,
        )
        # Minimizing k, best should be near 1
        assert result.best_score < 50.0

    def test_best_params_key_present(self) -> None:
        history: List[Dict] = []
        result = run_random(
            param_spaces=self._spaces(),
            objective_fn=lambda p: float(p["k"]),
            n_trials=5,
            maximize=True,
            history=history,
        )
        assert "k" in result.best_params

    def test_history_populated(self) -> None:
        history: List[Dict] = []
        run_random(
            param_spaces=self._spaces(),
            objective_fn=lambda p: 1.0,
            n_trials=7,
            maximize=True,
            history=history,
        )
        assert len(history) == 7
        assert all("params" in h and "score" in h for h in history)

    def test_categorical_space(self) -> None:
        history: List[Dict] = []
        result = run_random(
            param_spaces=[_make_cat_space("model", ["a", "b", "c"])],
            objective_fn=lambda p: 1.0 if p["model"] == "a" else 0.0,
            n_trials=10,
            maximize=True,
            history=history,
        )
        assert result.best_params["model"] in ["a", "b", "c"]

    def test_boolean_space(self) -> None:
        history: List[Dict] = []
        result = run_random(
            param_spaces=[_make_bool_space("use_cache")],
            objective_fn=lambda p: 1.0 if p["use_cache"] else 0.0,
            n_trials=20,
            maximize=True,
            history=history,
        )
        assert isinstance(result.best_params["use_cache"], bool)

    def test_float_space(self) -> None:
        history: List[Dict] = []
        result = run_random(
            param_spaces=[_make_float_space("temp", 0.0, 2.0)],
            objective_fn=lambda p: p["temp"],
            n_trials=20,
            maximize=True,
            history=history,
        )
        assert 0.0 <= result.best_params["temp"] <= 2.0

    def test_total_trials_field(self) -> None:
        history: List[Dict] = []
        result = run_random(
            param_spaces=self._spaces(),
            objective_fn=lambda p: 1.0,
            n_trials=6,
            maximize=True,
            history=history,
        )
        assert result.total_trials == 6


# ===========================================================================
# run_grid tests
# ===========================================================================


class TestRunGrid:
    def _spaces(self) -> List[ParameterSpace]:
        return [_make_int_space("k", 1, 5)]

    def test_returns_optimization_result(self) -> None:
        history: List[Dict] = []
        result = run_grid(
            param_spaces=self._spaces(),
            objective_fn=lambda p: float(p["k"]),
            maximize=True,
            history=history,
        )
        assert isinstance(result, OptimizationResult)

    def test_maximize_finds_max(self) -> None:
        history: List[Dict] = []
        result = run_grid(
            param_spaces=self._spaces(),
            objective_fn=lambda p: float(p["k"]),
            maximize=True,
            history=history,
        )
        assert result.best_score == max(h["score"] for h in history)

    def test_minimize_finds_min(self) -> None:
        history: List[Dict] = []
        result = run_grid(
            param_spaces=self._spaces(),
            objective_fn=lambda p: float(p["k"]),
            maximize=False,
            history=history,
        )
        assert result.best_score == min(h["score"] for h in history)

    def test_float_grid(self) -> None:
        history: List[Dict] = []
        result = run_grid(
            param_spaces=[_make_float_space("t", 0.0, 1.0)],
            objective_fn=lambda p: p["t"],
            maximize=True,
            history=history,
            grid_size=4,
        )
        assert result.best_score == pytest.approx(1.0)

    def test_categorical_grid_tries_all_categories(self) -> None:
        history: List[Dict] = []
        cats = ["a", "b", "c"]
        run_grid(
            param_spaces=[_make_cat_space("m", cats)],
            objective_fn=lambda p: 1.0,
            maximize=True,
            history=history,
        )
        tried_models = {h["params"]["m"] for h in history}
        assert tried_models == set(cats)

    def test_boolean_grid_tries_both(self) -> None:
        history: List[Dict] = []
        run_grid(
            param_spaces=[_make_bool_space("flag")],
            objective_fn=lambda p: 1.0 if p["flag"] else 0.0,
            maximize=True,
            history=history,
        )
        tried = {h["params"]["flag"] for h in history}
        assert True in tried
        assert False in tried

    def test_best_params_key_present(self) -> None:
        history: List[Dict] = []
        result = run_grid(
            param_spaces=self._spaces(),
            objective_fn=lambda p: float(p["k"]),
            maximize=True,
            history=history,
        )
        assert "k" in result.best_params

    def test_multi_param_grid(self) -> None:
        history: List[Dict] = []
        spaces = [
            _make_int_space("k", 1, 3),
            _make_float_space("t", 0.0, 1.0),
        ]
        result = run_grid(
            param_spaces=spaces,
            objective_fn=lambda p: float(p["k"]) + p["t"],
            maximize=True,
            history=history,
            grid_size=3,
        )
        assert "k" in result.best_params
        assert "t" in result.best_params
        assert result.best_score > 0


# ===========================================================================
# run_genetic tests
# ===========================================================================


class TestRunGenetic:
    def _spaces(self) -> List[ParameterSpace]:
        return [_make_int_space("k", 1, 20)]

    def test_returns_optimization_result(self) -> None:
        history: List[Dict] = []
        result = run_genetic(
            param_spaces=self._spaces(),
            objective_fn=lambda p: float(p["k"]),
            n_trials=40,
            maximize=True,
            history=history,
            population_size=10,
        )
        assert isinstance(result, OptimizationResult)

    def test_maximize_improves_score(self) -> None:
        history: List[Dict] = []
        result = run_genetic(
            param_spaces=[_make_int_space("k", 1, 100)],
            objective_fn=lambda p: float(p["k"]),
            n_trials=100,
            maximize=True,
            history=history,
            population_size=10,
        )
        # With genetic search, we expect to find values higher than 50 on average
        assert result.best_score > 1.0

    def test_minimize_finds_lower_score(self) -> None:
        history: List[Dict] = []
        result = run_genetic(
            param_spaces=[_make_int_space("k", 1, 100)],
            objective_fn=lambda p: float(p["k"]),
            n_trials=100,
            maximize=False,
            history=history,
            population_size=10,
        )
        assert result.best_score < 100.0

    def test_history_is_populated(self) -> None:
        history: List[Dict] = []
        run_genetic(
            param_spaces=self._spaces(),
            objective_fn=lambda p: 1.0,
            n_trials=20,
            maximize=True,
            history=history,
            population_size=10,
        )
        assert len(history) > 0

    def test_best_params_key_present(self) -> None:
        history: List[Dict] = []
        result = run_genetic(
            param_spaces=self._spaces(),
            objective_fn=lambda p: float(p["k"]),
            n_trials=20,
            maximize=True,
            history=history,
            population_size=10,
        )
        assert "k" in result.best_params

    def test_total_trials_matches_history(self) -> None:
        history: List[Dict] = []
        result = run_genetic(
            param_spaces=self._spaces(),
            objective_fn=lambda p: 1.0,
            n_trials=20,
            maximize=True,
            history=history,
            population_size=10,
        )
        assert result.total_trials == len(history)

    def test_custom_mutation_rate(self) -> None:
        history: List[Dict] = []
        result = run_genetic(
            param_spaces=self._spaces(),
            objective_fn=lambda p: float(p["k"]),
            n_trials=20,
            maximize=True,
            history=history,
            population_size=5,
            mutation_rate=0.5,
        )
        assert isinstance(result, OptimizationResult)

    def test_multi_param_genetic(self) -> None:
        history: List[Dict] = []
        spaces = [
            _make_int_space("k", 1, 10),
            _make_float_space("t", 0.0, 1.0),
        ]
        result = run_genetic(
            param_spaces=spaces,
            objective_fn=lambda p: float(p["k"]) * p["t"],
            n_trials=40,
            maximize=True,
            history=history,
            population_size=10,
        )
        assert "k" in result.best_params
        assert "t" in result.best_params

    def test_genetic_categorical_space(self) -> None:
        history: List[Dict] = []
        result = run_genetic(
            param_spaces=[_make_cat_space("m", ["a", "b", "c"])],
            objective_fn=lambda p: 1.0 if p["m"] == "a" else 0.0,
            n_trials=30,
            maximize=True,
            history=history,
            population_size=10,
        )
        assert result.best_params["m"] in ["a", "b", "c"]
        assert result.best_score >= 0.0


# ===========================================================================
# ParameterSpace edge-case tests (used as dependency above)
# ===========================================================================


class TestParameterSpace:
    def test_integer_space_requires_low_high(self) -> None:
        with pytest.raises(ValueError, match="low and high are required"):
            ParameterSpace(name="k", type=ParameterType.INTEGER)

    def test_float_space_requires_low_high(self) -> None:
        with pytest.raises(ValueError, match="low and high are required"):
            ParameterSpace(name="t", type=ParameterType.FLOAT)

    def test_categorical_requires_categories(self) -> None:
        with pytest.raises(ValueError, match="categories are required"):
            ParameterSpace(name="m", type=ParameterType.CATEGORICAL)

    def test_boolean_space_needs_no_bounds(self) -> None:
        sp = ParameterSpace(name="flag", type=ParameterType.BOOLEAN)
        sample = sp.sample()
        assert isinstance(sample, bool)

    def test_integer_sample_in_range(self) -> None:
        sp = _make_int_space("k", 3, 7)
        for _ in range(20):
            v = sp.sample()
            assert 3 <= v <= 7
            assert isinstance(v, int)

    def test_float_sample_in_range(self) -> None:
        sp = _make_float_space("t", 0.2, 0.8)
        for _ in range(20):
            v = sp.sample()
            assert 0.2 <= v <= 0.8

    def test_categorical_sample_in_categories(self) -> None:
        cats = ["x", "y", "z"]
        sp = _make_cat_space("m", cats)
        for _ in range(20):
            assert sp.sample() in cats
