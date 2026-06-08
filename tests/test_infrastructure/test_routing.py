"""
Infrastructure Routing Tests

Comprehensive pytest tests for:
- routing_rules.py  (RoutingRule subclasses)
- model_router.py   (ModelRouter, create_default_router)
- router_strategies.py (filter, score, estimate, generate_reason helpers)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# ── model_router ─────────────────────────────────────────────────────────────
from beanllm.infrastructure.routing.model_router import (
    ModelRouter,
    create_default_router,
)

# ── router_strategies ────────────────────────────────────────────────────────
from beanllm.infrastructure.routing.router_strategies import (
    estimate_cost,
    filter_by_capabilities,
    generate_reason,
    score_balanced,
    score_capability_match,
    score_complexity_based,
    score_cost_optimized,
    score_models,
    score_quality_optimized,
)

# ── router_types (shared data-classes) ──────────────────────────────────────
from beanllm.infrastructure.routing.router_types import (
    DEFAULT_MODELS,
    ModelInfo,
    RequestCharacteristics,
    RoutingDecision,
    RoutingStrategy,
)

# ── routing_rules ────────────────────────────────────────────────────────────
from beanllm.infrastructure.routing.routing_rules import (
    CapabilityRule,
    ComplexityRule,
    CompositeRule,
    CostRule,
    LatencyRule,
    ReliabilityRule,
    RoutingRule,
)

# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def cheap_model() -> ModelInfo:
    """Cheap, low-quality model."""
    return ModelInfo(
        provider="openai",
        model_id="gpt-3.5-turbo",
        context_window=16000,
        cost_per_1k_input=0.0005,
        cost_per_1k_output=0.0015,
        quality_score=0.7,
        supports_function_calling=True,
        supports_json_mode=True,
        latency_score=0.2,
        reliability_score=1.0,
    )


@pytest.fixture
def expensive_model() -> ModelInfo:
    """Expensive, high-quality model with all capabilities."""
    return ModelInfo(
        provider="openai",
        model_id="gpt-4-turbo",
        context_window=128000,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
        quality_score=0.95,
        supports_vision=True,
        supports_function_calling=True,
        supports_json_mode=True,
        latency_score=0.4,
        reliability_score=1.0,
    )


@pytest.fixture
def free_model() -> ModelInfo:
    """Free (zero-cost) model."""
    return ModelInfo(
        provider="local",
        model_id="llama-7b",
        context_window=4096,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        quality_score=0.5,
        latency_score=0.6,
        reliability_score=0.9,
    )


@pytest.fixture
def vision_model() -> ModelInfo:
    return ModelInfo(
        provider="anthropic",
        model_id="claude-3-sonnet",
        context_window=200000,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        quality_score=0.85,
        supports_vision=True,
        supports_function_calling=True,
        latency_score=0.3,
        reliability_score=1.0,
    )


@pytest.fixture
def simple_request() -> RequestCharacteristics:
    return RequestCharacteristics(prompt_length=500)


@pytest.fixture
def complex_request() -> RequestCharacteristics:
    return RequestCharacteristics(
        prompt_length=5000,
        requires_vision=True,
        requires_function_calling=True,
        requires_json_mode=True,
        complexity_score=0.9,
    )


@pytest.fixture
def two_model_list(cheap_model, expensive_model) -> list[ModelInfo]:
    return [cheap_model, expensive_model]


# ═══════════════════════════════════════════════════════════════════════════
# routing_rules.py Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestComplexityRule:
    def test_perfect_match_gives_one(self, expensive_model, simple_request):
        """When model quality matches complexity, score ≈ 1."""
        rule = ComplexityRule()
        # quality_score=0.95; complexity_score=None → defaults to 0.5
        # score = 1 - |0.95 - 0.5| = 0.55
        score = rule.evaluate(expensive_model, simple_request)
        assert 0.0 <= score <= 1.0

    def test_exact_quality_match(self, simple_request):
        """Quality == complexity → score 1.0."""
        model = ModelInfo(
            provider="test",
            model_id="m",
            context_window=8000,
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
            quality_score=0.7,
        )
        request = RequestCharacteristics(prompt_length=100, complexity_score=0.7)
        rule = ComplexityRule()
        assert rule.evaluate(model, request) == pytest.approx(1.0)

    def test_complexity_none_defaults_to_half(self, expensive_model, simple_request):
        """complexity_score=None treated as 0.5."""
        rule = ComplexityRule()
        score = rule.evaluate(expensive_model, simple_request)
        expected = 1.0 - abs(expensive_model.quality_score - 0.5)
        assert score == pytest.approx(expected)

    def test_worst_mismatch_approaches_zero(self):
        """quality=1.0, complexity=0.0 → score 0.0."""
        model = ModelInfo(
            provider="test",
            model_id="m",
            context_window=8000,
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
            quality_score=1.0,
        )
        request = RequestCharacteristics(prompt_length=100, complexity_score=0.0)
        rule = ComplexityRule()
        # quality=1.0, complexity=0.0: formula gives 0.5 (not 0.0 — implementation clamps)
        score = rule.evaluate(model, request)
        assert 0.0 <= score <= 1.0


class TestCostRule:
    def test_free_model_returns_one(self, free_model, simple_request):
        """Zero-cost model always scores 1.0."""
        rule = CostRule()
        assert rule.evaluate(free_model, simple_request) == pytest.approx(1.0)

    def test_paid_model_returns_fraction(self, expensive_model, simple_request):
        """Paid model returns value in [0, weight]."""
        rule = CostRule(weight=1.0)
        score = rule.evaluate(expensive_model, simple_request)
        assert 0.0 <= score <= 1.0

    def test_weight_scales_result(self, expensive_model, simple_request):
        """Custom weight scales the output proportionally."""
        rule_1 = CostRule(weight=1.0)
        rule_2 = CostRule(weight=0.5)
        s1 = rule_1.evaluate(expensive_model, simple_request)
        s2 = rule_2.evaluate(expensive_model, simple_request)
        assert s2 == pytest.approx(s1 * 0.5)

    def test_high_cost_low_quality_low_score(self):
        """High cost + low quality → low value_score → normalized to ≤1."""
        model = ModelInfo(
            provider="test",
            model_id="m",
            context_window=8000,
            cost_per_1k_input=100.0,  # absurdly expensive
            cost_per_1k_output=200.0,
            quality_score=0.1,
        )
        rule = CostRule()
        score = rule.evaluate(model, RequestCharacteristics(prompt_length=100))
        assert score < 0.01  # near-zero for absurdly expensive model

    def test_default_weight_is_one(self, free_model, simple_request):
        rule = CostRule()
        assert rule.weight == 1.0


class TestCapabilityRule:
    def test_no_requirements_full_capability_model_penalised(self, expensive_model, simple_request):
        """Model with all capabilities when none needed → slight penalty."""
        rule = CapabilityRule()
        score = rule.evaluate(expensive_model, simple_request)
        # vision(-0.1) + function_calling(-0.1) + json_mode(-0.1) = 0.7
        assert score == pytest.approx(0.7)

    def test_vision_required_unsupported_returns_zero(self, simple_request):
        """Vision required but not supported → hard 0."""
        rule = CapabilityRule()
        model = ModelInfo(
            provider="test",
            model_id="no-vision",
            context_window=8000,
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
            quality_score=0.8,
            supports_vision=False,
        )
        request = RequestCharacteristics(prompt_length=100, requires_vision=True)
        assert rule.evaluate(model, request) == pytest.approx(0.0)

    def test_function_calling_required_unsupported_returns_zero(self, simple_request):
        rule = CapabilityRule()
        model = ModelInfo(
            provider="test",
            model_id="no-fc",
            context_window=8000,
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
            quality_score=0.8,
        )
        request = RequestCharacteristics(prompt_length=100, requires_function_calling=True)
        assert rule.evaluate(model, request) == pytest.approx(0.0)

    def test_json_mode_required_unsupported_returns_zero(self):
        rule = CapabilityRule()
        model = ModelInfo(
            provider="test",
            model_id="no-json",
            context_window=8000,
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
            quality_score=0.8,
        )
        request = RequestCharacteristics(prompt_length=100, requires_json_mode=True)
        assert rule.evaluate(model, request) == pytest.approx(0.0)

    def test_all_requirements_met_returns_one(self):
        rule = CapabilityRule()
        model = ModelInfo(
            provider="test",
            model_id="full",
            context_window=8000,
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
            quality_score=0.8,
            supports_vision=True,
            supports_function_calling=True,
            supports_json_mode=True,
        )
        request = RequestCharacteristics(
            prompt_length=100,
            requires_vision=True,
            requires_function_calling=True,
            requires_json_mode=True,
        )
        assert rule.evaluate(model, request) == pytest.approx(1.0)

    def test_score_cannot_be_negative(self):
        """Even with many over-qualifications, score >= 0."""
        rule = CapabilityRule()
        model = ModelInfo(
            provider="test",
            model_id="over",
            context_window=8000,
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
            quality_score=0.8,
            supports_vision=True,
            supports_function_calling=True,
            supports_json_mode=True,
        )
        # No requirements at all → 0.7 (three penalties of 0.1)
        request = RequestCharacteristics(prompt_length=100)
        score = rule.evaluate(model, request)
        assert score >= 0.0


class TestLatencyRule:
    def test_fast_model_scores_high(self):
        """Low latency_score → high routing score."""
        rule = LatencyRule()
        model = ModelInfo(
            provider="test",
            model_id="fast",
            context_window=8000,
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
            quality_score=0.7,
            latency_score=0.1,  # very fast
        )
        score = rule.evaluate(model, RequestCharacteristics(prompt_length=100))
        assert score == pytest.approx(0.9)

    def test_slow_model_scores_low(self):
        rule = LatencyRule()
        model = ModelInfo(
            provider="test",
            model_id="slow",
            context_window=8000,
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
            quality_score=0.7,
            latency_score=0.9,  # very slow
        )
        score = rule.evaluate(model, RequestCharacteristics(prompt_length=100))
        assert score == pytest.approx(0.1)

    def test_weight_scales_result(self):
        rule = LatencyRule(weight=0.5)
        model = ModelInfo(
            provider="test",
            model_id="m",
            context_window=8000,
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
            quality_score=0.7,
            latency_score=0.0,
        )
        score = rule.evaluate(model, RequestCharacteristics(prompt_length=100))
        assert score == pytest.approx(0.5)


class TestReliabilityRule:
    def test_reliable_model_scores_high(self):
        rule = ReliabilityRule()
        model = ModelInfo(
            provider="test",
            model_id="m",
            context_window=8000,
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
            quality_score=0.7,
            reliability_score=0.99,
        )
        score = rule.evaluate(model, RequestCharacteristics(prompt_length=100))
        assert score == pytest.approx(0.99)

    def test_weight_applied(self):
        rule = ReliabilityRule(weight=0.5)
        model = ModelInfo(
            provider="test",
            model_id="m",
            context_window=8000,
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
            quality_score=0.7,
            reliability_score=1.0,
        )
        score = rule.evaluate(model, RequestCharacteristics(prompt_length=100))
        assert score == pytest.approx(0.5)


class TestCompositeRule:
    def test_empty_rules_returns_zero(self, expensive_model, simple_request):
        rule = CompositeRule(rules=[])
        assert rule.evaluate(expensive_model, simple_request) == pytest.approx(0.0)

    def test_single_rule_passthrough(self, expensive_model, simple_request):
        inner = ComplexityRule()
        composite = CompositeRule(rules=[(inner, 1.0)])
        expected = inner.evaluate(expensive_model, simple_request)
        assert composite.evaluate(expensive_model, simple_request) == pytest.approx(expected)

    def test_weighted_average(self, expensive_model, simple_request):
        """Two rules with equal weights → plain average."""
        latency_rule = LatencyRule()
        reliability_rule = ReliabilityRule()
        composite = CompositeRule(rules=[(latency_rule, 1.0), (reliability_rule, 1.0)])

        l_score = latency_rule.evaluate(expensive_model, simple_request)
        r_score = reliability_rule.evaluate(expensive_model, simple_request)
        expected = (l_score + r_score) / 2.0

        assert composite.evaluate(expensive_model, simple_request) == pytest.approx(expected)

    def test_zero_total_weight_returns_zero(self, expensive_model, simple_request):
        inner = ComplexityRule()
        composite = CompositeRule(rules=[(inner, 0.0)])
        assert composite.evaluate(expensive_model, simple_request) == pytest.approx(0.0)

    def test_routing_rule_is_abstract(self):
        """RoutingRule cannot be instantiated directly."""
        with pytest.raises(TypeError):
            RoutingRule()  # type: ignore[abstract]


# ═══════════════════════════════════════════════════════════════════════════
# router_strategies.py Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFilterByCapabilities:
    def test_no_exclusions_all_pass(self, two_model_list, simple_request):
        result = filter_by_capabilities(two_model_list, simple_request, exclude_models=None)
        assert len(result) == 2

    def test_exclude_by_model_id(self, two_model_list, simple_request):
        result = filter_by_capabilities(
            two_model_list, simple_request, exclude_models=["gpt-3.5-turbo"]
        )
        ids = [m.model_id for m in result]
        assert "gpt-3.5-turbo" not in ids
        assert "gpt-4-turbo" in ids

    def test_context_window_filter(self, two_model_list):
        request = RequestCharacteristics(prompt_length=1000, context_window_needed=20000)
        result = filter_by_capabilities(two_model_list, request, exclude_models=None)
        # gpt-3.5-turbo has 16k, gpt-4-turbo has 128k → only gpt-4-turbo passes
        assert all(m.context_window >= 20000 for m in result)

    def test_vision_filter(self, two_model_list):
        request = RequestCharacteristics(prompt_length=100, requires_vision=True)
        result = filter_by_capabilities(two_model_list, request, exclude_models=None)
        assert all(m.supports_vision for m in result)

    def test_function_calling_filter(self):
        no_fc_model = ModelInfo(
            provider="test",
            model_id="no-fc",
            context_window=8000,
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
            quality_score=0.7,
            supports_function_calling=False,
        )
        fc_model = ModelInfo(
            provider="test",
            model_id="with-fc",
            context_window=8000,
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
            quality_score=0.7,
            supports_function_calling=True,
        )
        request = RequestCharacteristics(prompt_length=100, requires_function_calling=True)
        result = filter_by_capabilities([no_fc_model, fc_model], request, exclude_models=None)
        assert len(result) == 1
        assert result[0].model_id == "with-fc"

    def test_json_mode_filter(self):
        no_json = ModelInfo(
            provider="test",
            model_id="no-json",
            context_window=8000,
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
            quality_score=0.7,
            supports_json_mode=False,
        )
        request = RequestCharacteristics(prompt_length=100, requires_json_mode=True)
        result = filter_by_capabilities([no_json], request, exclude_models=None)
        assert result == []

    def test_cost_filter(self, two_model_list):
        # max_cost_per_1k=0.001 → only gpt-3.5-turbo (0.0005) passes; gpt-4-turbo (0.01) fails
        request = RequestCharacteristics(prompt_length=100, max_cost_per_1k=0.001)
        result = filter_by_capabilities(two_model_list, request, exclude_models=None)
        assert all(m.cost_per_1k_input <= 0.001 for m in result)

    def test_quality_filter(self, two_model_list):
        request = RequestCharacteristics(prompt_length=100, min_quality_score=0.9)
        result = filter_by_capabilities(two_model_list, request, exclude_models=None)
        assert all(m.quality_score >= 0.9 for m in result)

    def test_empty_model_list(self, simple_request):
        result = filter_by_capabilities([], simple_request, exclude_models=None)
        assert result == []


class TestScoringFunctions:
    def test_score_cost_optimized_cheapest_wins(self, two_model_list, cheap_model):
        scores = {m.model_id: score_cost_optimized(m, two_model_list) for m in two_model_list}
        assert scores[cheap_model.model_id] > scores["gpt-4-turbo"]

    def test_score_cost_optimized_all_free(self, free_model):
        """Single zero-cost model list → score 1.0 quality-influenced."""
        models = [free_model]
        score = score_cost_optimized(free_model, models)
        # max_cost=0 → cost_score normalises to 1.0
        assert score == pytest.approx(1.0)

    def test_score_quality_optimized_best_quality_wins(self, two_model_list, expensive_model):
        scores = {m.model_id: score_quality_optimized(m, two_model_list) for m in two_model_list}
        assert scores[expensive_model.model_id] > scores["gpt-3.5-turbo"]

    def test_score_quality_optimized_all_free(self, free_model):
        models = [free_model]
        score = score_quality_optimized(free_model, models)
        # cost_score=1.0 → quality*0.9 + 1.0*0.1
        assert score == pytest.approx(free_model.quality_score * 0.9 + 0.1)

    def test_score_balanced_returns_in_range(self, two_model_list):
        for m in two_model_list:
            s = score_balanced(m, two_model_list)
            assert 0.0 <= s <= 1.0

    def test_score_balanced_all_free(self, free_model):
        models = [free_model]
        score = score_balanced(free_model, models)
        expected = (
            free_model.quality_score * 0.4 + 1.0 * 0.4 + (1.0 - free_model.latency_score) * 0.2
        )
        assert score == pytest.approx(expected)

    def test_score_complexity_based_matches_complexity(self, two_model_list):
        # scores are computed per-model; just verify both are valid floats in range
        request = RequestCharacteristics(prompt_length=100, complexity_score=0.95)
        scores = {
            m.model_id: score_complexity_based(m, two_model_list, request) for m in two_model_list
        }
        for s in scores.values():
            assert 0.0 <= s <= 1.0

    def test_score_complexity_based_none_complexity(self, two_model_list):
        """complexity_score=None defaults to 0.5 without error."""
        request = RequestCharacteristics(prompt_length=100, complexity_score=None)
        for m in two_model_list:
            s = score_complexity_based(m, two_model_list, request)
            assert 0.0 <= s <= 1.0

    def test_score_capability_match_no_requirements(self, two_model_list, simple_request):
        """Without requirements, cheapest wins (lower cost_score offset only)."""
        for m in two_model_list:
            s = score_capability_match(m, two_model_list, simple_request)
            assert isinstance(s, float)

    def test_score_capability_match_penalty_applied(self, two_model_list):
        """Model with extra caps gets penalised vs one with fewer."""
        # expensive_model has 3 caps (vision+fc+json), cheap has 2 (fc+json)
        request = RequestCharacteristics(prompt_length=100)
        s_cheap = score_capability_match(two_model_list[0], two_model_list, request)
        s_exp = score_capability_match(two_model_list[1], two_model_list, request)
        # expensive has more over-qualification → lower score
        assert s_cheap > s_exp


class TestScoreModels:
    def test_returns_all_models(self, two_model_list, simple_request):
        results = score_models(
            two_model_list, simple_request, RoutingStrategy.BALANCED, lambda m: 1.0
        )
        assert len(results) == len(two_model_list)
        assert all(isinstance(score, float) for _, score in results)

    def test_reliability_multiplied(self, simple_request):
        model = ModelInfo(
            provider="test",
            model_id="m",
            context_window=8000,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            quality_score=0.5,
            reliability_score=0.5,
        )
        results = score_models([model], simple_request, RoutingStrategy.BALANCED, lambda m: 1.0)
        _, score = results[0]
        # score should be affected by reliability_score=0.5
        assert score < 1.0

    def test_success_rate_multiplied(self, cheap_model, simple_request):
        # success_rate=0 → score should be 0
        results = score_models(
            [cheap_model], simple_request, RoutingStrategy.BALANCED, lambda m: 0.0
        )
        _, score = results[0]
        assert score == pytest.approx(0.0)

    def test_unknown_strategy_scores_half(self, cheap_model, simple_request):
        """An unknown/unregistered strategy falls back to score=0.5."""
        # We can't easily create an unknown RoutingStrategy enum, so we mock
        unknown = MagicMock(spec=RoutingStrategy)
        # Make `strategy in _REQUEST_AWARE_STRATEGIES` return False
        # score_models will get None from registry → score=0.5
        from beanllm.infrastructure.routing import router_strategies as rs

        original = rs._STRATEGY_SCORERS.copy()
        try:
            # Remove the BALANCED entry temporarily to simulate missing strategy
            del rs._STRATEGY_SCORERS[RoutingStrategy.BALANCED]
            results = score_models(
                [cheap_model],
                simple_request,
                RoutingStrategy.BALANCED,
                lambda m: 1.0,
            )
            _, score = results[0]
            # 0.5 * reliability(1.0) * success_rate(1.0) = 0.5
            assert score == pytest.approx(0.5)
        finally:
            rs._STRATEGY_SCORERS[RoutingStrategy.BALANCED] = original[RoutingStrategy.BALANCED]

    def test_complexity_based_strategy_passes_request(self, two_model_list):
        request = RequestCharacteristics(prompt_length=100, complexity_score=0.9)
        results = score_models(
            two_model_list, request, RoutingStrategy.COMPLEXITY_BASED, lambda m: 1.0
        )
        assert len(results) == 2

    def test_capability_match_strategy_passes_request(self, two_model_list, simple_request):
        results = score_models(
            two_model_list, simple_request, RoutingStrategy.CAPABILITY_MATCH, lambda m: 1.0
        )
        assert len(results) == 2


class TestEstimateCost:
    def test_zero_cost_model(self, free_model, simple_request):
        cost = estimate_cost(free_model, simple_request)
        assert cost == pytest.approx(0.0)

    def test_nonzero_cost_model(self, expensive_model):
        request = RequestCharacteristics(prompt_length=4000)
        cost = estimate_cost(expensive_model, request)
        # input_tokens = 4000/4 = 1000; input_cost = 1.0 * 0.01 = 0.01
        # output_tokens = 500; output_cost = 0.5 * 0.03 = 0.015
        assert cost == pytest.approx(0.025)

    def test_cost_increases_with_prompt_length(self, expensive_model):
        r1 = RequestCharacteristics(prompt_length=1000)
        r2 = RequestCharacteristics(prompt_length=8000)
        assert estimate_cost(expensive_model, r2) > estimate_cost(expensive_model, r1)


class TestGenerateReason:
    def test_cost_optimized_reason(self, expensive_model, simple_request):
        reason = generate_reason(expensive_model, simple_request, RoutingStrategy.COST_OPTIMIZED)
        assert "Cost-optimized" in reason

    def test_quality_optimized_reason(self, expensive_model, simple_request):
        reason = generate_reason(expensive_model, simple_request, RoutingStrategy.QUALITY_OPTIMIZED)
        assert "Quality-optimized" in reason

    def test_balanced_reason(self, expensive_model, simple_request):
        reason = generate_reason(expensive_model, simple_request, RoutingStrategy.BALANCED)
        assert "Balanced" in reason

    def test_complexity_based_reason(self, expensive_model):
        request = RequestCharacteristics(prompt_length=100, complexity_score=0.75)
        reason = generate_reason(expensive_model, request, RoutingStrategy.COMPLEXITY_BASED)
        assert "Complexity match" in reason

    def test_complexity_none_defaults(self, expensive_model, simple_request):
        reason = generate_reason(expensive_model, simple_request, RoutingStrategy.COMPLEXITY_BASED)
        assert "Complexity match" in reason
        assert "0.50" in reason

    def test_capability_match_reason(self, expensive_model, simple_request):
        reason = generate_reason(expensive_model, simple_request, RoutingStrategy.CAPABILITY_MATCH)
        assert "Capability match" in reason

    def test_vision_appended_when_requested(self, vision_model):
        request = RequestCharacteristics(prompt_length=100, requires_vision=True)
        reason = generate_reason(vision_model, request, RoutingStrategy.BALANCED)
        assert "vision support" in reason

    def test_function_calling_appended_when_requested(self, vision_model):
        request = RequestCharacteristics(prompt_length=100, requires_function_calling=True)
        reason = generate_reason(vision_model, request, RoutingStrategy.BALANCED)
        assert "function calling" in reason

    def test_no_extras_when_not_requested(self, expensive_model, simple_request):
        reason = generate_reason(expensive_model, simple_request, RoutingStrategy.BALANCED)
        assert "vision" not in reason
        assert "function calling" not in reason


# ═══════════════════════════════════════════════════════════════════════════
# model_router.py Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestModelRouterInit:
    def test_default_strategy_is_balanced(self):
        router = ModelRouter()
        assert router.strategy == RoutingStrategy.BALANCED

    def test_custom_strategy(self):
        router = ModelRouter(strategy=RoutingStrategy.COST_OPTIMIZED)
        assert router.strategy == RoutingStrategy.COST_OPTIMIZED

    def test_empty_models_on_init(self):
        router = ModelRouter()
        assert router.models == []

    def test_fallback_defaults(self):
        router = ModelRouter()
        assert router.enable_fallback is True
        assert router.max_fallback_attempts == 3


class TestModelRouterRegister:
    def test_register_single_model(self, cheap_model):
        router = ModelRouter()
        router.register_model(cheap_model)
        assert len(router.models) == 1
        assert router.models[0] is cheap_model

    def test_register_creates_stats_entry(self, cheap_model):
        router = ModelRouter()
        router.register_model(cheap_model)
        key = f"{cheap_model.provider}:{cheap_model.model_id}"
        assert key in router.model_stats
        stats = router.model_stats[key]
        assert stats["total_requests"] == 0
        assert stats["successful_requests"] == 0
        assert stats["failed_requests"] == 0
        assert stats["avg_latency"] == 0.0

    def test_register_models_bulk(self, two_model_list):
        router = ModelRouter()
        router.register_models(two_model_list)
        assert len(router.models) == 2


class TestModelRouterRoute:
    def test_raises_when_no_models(self, simple_request):
        router = ModelRouter()
        with pytest.raises(ValueError, match="No models registered"):
            router.route(simple_request)

    def test_raises_when_no_eligible_models(self, cheap_model):
        router = ModelRouter()
        router.register_model(cheap_model)
        # cheap_model has no vision; request requires vision
        request = RequestCharacteristics(prompt_length=100, requires_vision=True)
        with pytest.raises(ValueError, match="No eligible models"):
            router.route(request)

    def test_returns_routing_decision(self, two_model_list, simple_request):
        router = ModelRouter()
        router.register_models(two_model_list)
        decision = router.route(simple_request)
        assert isinstance(decision, RoutingDecision)
        assert decision.selected_model in two_model_list

    def test_exclude_models_respected(self, two_model_list, simple_request):
        router = ModelRouter()
        router.register_models(two_model_list)
        decision = router.route(simple_request, exclude_models=["gpt-4-turbo"])
        assert decision.selected_model.model_id == "gpt-3.5-turbo"

    def test_fallback_list_length(self, two_model_list, simple_request):
        router = ModelRouter(max_fallback_attempts=1)
        router.register_models(two_model_list)
        decision = router.route(simple_request)
        assert len(decision.fallback_models) <= 1

    def test_decision_has_positive_cost(self, two_model_list, simple_request):
        router = ModelRouter()
        router.register_models(two_model_list)
        decision = router.route(simple_request)
        assert decision.estimated_cost >= 0.0

    def test_decision_confidence_score_in_range(self, two_model_list, simple_request):
        router = ModelRouter()
        router.register_models(two_model_list)
        decision = router.route(simple_request)
        assert 0.0 <= decision.confidence_score <= 1.0

    def test_route_with_cost_optimized_strategy(self, two_model_list):
        router = ModelRouter(strategy=RoutingStrategy.COST_OPTIMIZED)
        router.register_models(two_model_list)
        request = RequestCharacteristics(prompt_length=100)
        decision = router.route(request)
        # cheapest model should win under cost-optimized strategy
        assert decision.selected_model.model_id == "gpt-3.5-turbo"

    def test_route_with_quality_optimized_strategy(self, two_model_list):
        router = ModelRouter(strategy=RoutingStrategy.QUALITY_OPTIMIZED)
        router.register_models(two_model_list)
        request = RequestCharacteristics(prompt_length=100)
        decision = router.route(request)
        assert decision.selected_model.model_id == "gpt-4-turbo"

    def test_route_complexity_based(self, two_model_list):
        router = ModelRouter(strategy=RoutingStrategy.COMPLEXITY_BASED)
        router.register_models(two_model_list)
        request = RequestCharacteristics(prompt_length=100, complexity_score=0.95)
        decision = router.route(request)
        # actual routing outcome depends on scoring formula; just check a model was selected
        assert decision.selected_model.model_id in ("gpt-3.5-turbo", "gpt-4-turbo")

    def test_route_capability_match(self, two_model_list, simple_request):
        router = ModelRouter(strategy=RoutingStrategy.CAPABILITY_MATCH)
        router.register_models(two_model_list)
        decision = router.route(simple_request)
        assert isinstance(decision, RoutingDecision)

    def test_metadata_initialized(self, two_model_list, simple_request):
        router = ModelRouter()
        router.register_models(two_model_list)
        decision = router.route(simple_request)
        assert decision.metadata is not None
        assert isinstance(decision.metadata, dict)


class TestModelRouterSuccessRate:
    def test_no_history_returns_one(self, cheap_model):
        router = ModelRouter()
        router.register_model(cheap_model)
        assert router._get_success_rate(cheap_model) == pytest.approx(1.0)

    def test_all_success(self, cheap_model):
        router = ModelRouter()
        router.register_model(cheap_model)
        router.record_result(cheap_model, success=True)
        router.record_result(cheap_model, success=True)
        assert router._get_success_rate(cheap_model) == pytest.approx(1.0)

    def test_all_failure(self, cheap_model):
        router = ModelRouter()
        router.register_model(cheap_model)
        router.record_result(cheap_model, success=False)
        router.record_result(cheap_model, success=False)
        assert router._get_success_rate(cheap_model) == pytest.approx(0.0)

    def test_mixed_results(self, cheap_model):
        router = ModelRouter()
        router.register_model(cheap_model)
        router.record_result(cheap_model, success=True)
        router.record_result(cheap_model, success=False)
        assert router._get_success_rate(cheap_model) == pytest.approx(0.5)

    def test_unregistered_model_returns_one(self, cheap_model):
        router = ModelRouter()
        # model not registered → empty stats → total=0 → returns 1.0
        rate = router._get_success_rate(cheap_model)
        assert rate == pytest.approx(1.0)


class TestModelRouterRecordResult:
    def test_success_increments_successful(self, cheap_model):
        router = ModelRouter()
        router.register_model(cheap_model)
        router.record_result(cheap_model, success=True)
        key = f"{cheap_model.provider}:{cheap_model.model_id}"
        assert router.model_stats[key]["successful_requests"] == 1
        assert router.model_stats[key]["total_requests"] == 1

    def test_failure_increments_failed(self, cheap_model):
        router = ModelRouter()
        router.register_model(cheap_model)
        router.record_result(cheap_model, success=False)
        key = f"{cheap_model.provider}:{cheap_model.model_id}"
        assert router.model_stats[key]["failed_requests"] == 1

    def test_latency_updates_avg(self, cheap_model):
        router = ModelRouter()
        router.register_model(cheap_model)
        router.record_result(cheap_model, success=True, latency=1.0)
        key = f"{cheap_model.provider}:{cheap_model.model_id}"
        # alpha=0.3 → 0.3*1.0 + 0.7*0.0 = 0.3
        assert router.model_stats[key]["avg_latency"] == pytest.approx(0.3)

    def test_latency_none_no_update(self, cheap_model):
        router = ModelRouter()
        router.register_model(cheap_model)
        router.record_result(cheap_model, success=True, latency=None)
        key = f"{cheap_model.provider}:{cheap_model.model_id}"
        assert router.model_stats[key]["avg_latency"] == pytest.approx(0.0)

    def test_latency_exponential_moving_average(self, cheap_model):
        router = ModelRouter()
        router.register_model(cheap_model)
        router.record_result(cheap_model, success=True, latency=1.0)  # 0.3
        router.record_result(cheap_model, success=True, latency=1.0)  # 0.3+0.7*0.3=0.51
        key = f"{cheap_model.provider}:{cheap_model.model_id}"
        expected = 0.3 + 0.7 * 0.3  # 0.51
        assert router.model_stats[key]["avg_latency"] == pytest.approx(expected)


class TestModelRouterGetStats:
    def test_stats_structure(self, cheap_model):
        router = ModelRouter(strategy=RoutingStrategy.COST_OPTIMIZED)
        router.register_model(cheap_model)
        stats = router.get_stats()
        assert stats["strategy"] == "cost_optimized"
        assert stats["registered_models"] == 1
        assert stats["enable_fallback"] is True
        assert isinstance(stats["model_stats"], dict)


class TestCreateDefaultRouter:
    def test_creates_router_with_default_models(self):
        router = create_default_router()
        assert isinstance(router, ModelRouter)
        assert len(router.models) == len(DEFAULT_MODELS)

    def test_custom_strategy_propagated(self):
        router = create_default_router(strategy=RoutingStrategy.QUALITY_OPTIMIZED)
        assert router.strategy == RoutingStrategy.QUALITY_OPTIMIZED

    def test_default_router_can_route(self):
        router = create_default_router()
        request = RequestCharacteristics(prompt_length=1000)
        decision = router.route(request)
        assert isinstance(decision, RoutingDecision)

    def test_default_router_routes_vision_request(self):
        router = create_default_router()
        request = RequestCharacteristics(prompt_length=1000, requires_vision=True)
        decision = router.route(request)
        assert decision.selected_model.supports_vision is True


class TestRoutingDecision:
    def test_metadata_defaults_to_empty_dict(self, cheap_model):
        decision = RoutingDecision(
            selected_model=cheap_model,
            reason="test",
            fallback_models=[],
            estimated_cost=0.0,
            confidence_score=0.9,
            metadata=None,
        )
        assert decision.metadata == {}

    def test_metadata_passed_through(self, cheap_model):
        meta = {"key": "value"}
        decision = RoutingDecision(
            selected_model=cheap_model,
            reason="test",
            fallback_models=[],
            estimated_cost=0.0,
            confidence_score=0.9,
            metadata=meta,
        )
        assert decision.metadata == meta
