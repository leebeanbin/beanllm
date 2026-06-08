"""Tests for utils/token_pricing.py — ModelPricing, ModelContextWindow, CostEstimate, CostEstimator."""

import pytest

from beanllm.utils.token_pricing import (
    CostEstimate,
    CostEstimator,
    ModelContextWindow,
    ModelPricing,
    estimate_cost,
    get_cheapest_model,
    get_context_window,
)

# ---------------------------------------------------------------------------
# ModelPricing
# ---------------------------------------------------------------------------


class TestModelPricing:
    def test_known_openai_model(self):
        pricing = ModelPricing.get_pricing("gpt-4o")
        assert pricing is not None
        assert "input" in pricing
        assert "output" in pricing
        assert pricing["input"] == pytest.approx(2.50)

    def test_known_anthropic_model(self):
        pricing = ModelPricing.get_pricing("claude-3-5-sonnet-20241022")
        assert pricing is not None
        assert pricing["input"] == pytest.approx(3.00)

    def test_known_google_model(self):
        pricing = ModelPricing.get_pricing("gemini-1.5-pro")
        assert pricing is not None

    def test_known_ollama_model_is_free(self):
        pricing = ModelPricing.get_pricing("llama3.2")
        assert pricing is not None
        assert pricing["input"] == 0.0
        assert pricing["output"] == 0.0

    def test_unknown_model_returns_none(self):
        pricing = ModelPricing.get_pricing("totally-unknown-model-xyz")
        assert pricing is None

    def test_partial_match(self):
        pricing = ModelPricing.get_pricing("gpt-4o-mini-2024-07-18")
        assert pricing is not None

    def test_embedding_model_no_output_cost(self):
        pricing = ModelPricing.get_pricing("text-embedding-3-small")
        assert pricing is not None
        assert pricing["output"] == 0.0

    def test_all_models_has_entries(self):
        assert len(ModelPricing.ALL_MODELS) > 10


# ---------------------------------------------------------------------------
# ModelContextWindow
# ---------------------------------------------------------------------------


class TestModelContextWindow:
    def test_gpt4o_context_window(self):
        window = ModelContextWindow.get_context_window("gpt-4o")
        assert window == 128000

    def test_claude_large_context_window(self):
        window = ModelContextWindow.get_context_window("claude-3-5-sonnet-20241022")
        assert window == 200000

    def test_gemini_very_large_context(self):
        window = ModelContextWindow.get_context_window("gemini-1.5-pro")
        assert window == 2000000

    def test_unknown_model_returns_default(self):
        window = ModelContextWindow.get_context_window("totally-unknown-model-xyz")
        assert isinstance(window, int)
        assert window > 0

    def test_partial_match(self):
        window = ModelContextWindow.get_context_window("gpt-4o-mini-2024")
        assert window > 0


# ---------------------------------------------------------------------------
# CostEstimate
# ---------------------------------------------------------------------------


class TestCostEstimate:
    def test_str_representation(self):
        estimate = CostEstimate(
            input_tokens=1000,
            output_tokens=500,
            input_cost=0.0025,
            output_cost=0.005,
            total_cost=0.0075,
            model="gpt-4o",
        )
        s = str(estimate)
        assert "gpt-4o" in s
        assert "1,000" in s

    def test_total_cost_field(self):
        estimate = CostEstimate(
            input_tokens=100,
            output_tokens=50,
            input_cost=0.001,
            output_cost=0.0005,
            total_cost=0.0015,
            model="test-model",
        )
        assert estimate.total_cost == pytest.approx(0.0015)

    def test_default_currency_is_usd(self):
        estimate = CostEstimate(
            input_tokens=100,
            output_tokens=50,
            input_cost=0.01,
            output_cost=0.005,
            total_cost=0.015,
            model="test",
        )
        assert estimate.currency == "USD"


# ---------------------------------------------------------------------------
# CostEstimator
# ---------------------------------------------------------------------------


class TestCostEstimator:
    def test_estimate_with_direct_token_counts(self):
        estimator = CostEstimator(model="gpt-4o")
        result = estimator.estimate_cost(input_tokens=1000, output_tokens=500)
        assert isinstance(result, CostEstimate)
        assert result.input_tokens == 1000
        assert result.output_tokens == 500
        assert result.total_cost >= 0.0

    def test_estimate_with_text(self):
        estimator = CostEstimator(model="gpt-4o")
        result = estimator.estimate_cost(
            input_text="Hello, world!",
            output_text="Hi there!",
        )
        assert isinstance(result, CostEstimate)
        assert result.input_tokens > 0

    def test_estimate_zero_tokens(self):
        estimator = CostEstimator(model="gpt-4o")
        result = estimator.estimate_cost(input_tokens=0, output_tokens=0)
        assert result.total_cost == 0.0

    def test_estimate_unknown_model_warns(self):
        estimator = CostEstimator(model="unknown-model-xyz")
        with pytest.warns(UserWarning, match="Pricing not found"):
            result = estimator.estimate_cost(input_tokens=100, output_tokens=50)
        assert result.total_cost == 0.0

    def test_compare_models_sorted_by_cost(self):
        estimator = CostEstimator(model="gpt-4o")
        models = ["gpt-4o", "gpt-4o-mini"]
        results = estimator.compare_models(models, input_text="Test input", output_tokens=100)
        assert len(results) == 2
        # Sorted by total_cost ascending
        assert results[0].total_cost <= results[1].total_cost

    def test_estimate_with_messages(self):
        estimator = CostEstimator(model="gpt-4o")
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = estimator.estimate_cost(messages=messages)
        assert result.input_tokens > 0

    def test_model_stored_in_result(self):
        estimator = CostEstimator(model="gpt-4o-mini")
        result = estimator.estimate_cost(input_tokens=100, output_tokens=50)
        assert result.model == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    def test_estimate_cost_returns_cost_estimate(self):
        result = estimate_cost("Hello world", "Hi there", model="gpt-4o")
        assert isinstance(result, CostEstimate)

    def test_estimate_cost_no_output_text(self):
        result = estimate_cost("Hello world", model="gpt-4o")
        assert result.output_tokens == 0

    def test_get_cheapest_model_returns_string(self):
        cheapest = get_cheapest_model("Test input", output_tokens=100)
        assert isinstance(cheapest, str)
        assert len(cheapest) > 0

    def test_get_cheapest_model_custom_models(self):
        cheapest = get_cheapest_model(
            "Test",
            output_tokens=100,
            models=["gpt-4o-mini", "gpt-4o"],
        )
        assert cheapest in ["gpt-4o-mini", "gpt-4o"]

    def test_get_context_window_known_model(self):
        window = get_context_window("gpt-4o")
        assert window == 128000

    def test_get_context_window_unknown_model(self):
        window = get_context_window("unknown-model-xyz")
        assert isinstance(window, int)
        assert window > 0
