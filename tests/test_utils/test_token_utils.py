"""
Token Counter 및 Token Pricing 테스트
"""

import pytest

from beanllm.utils.token_counter import (
    TokenCounter,
    count_message_tokens,
    count_tokens,
)
from beanllm.utils.token_pricing import (
    CostEstimate,
    CostEstimator,
    ModelContextWindow,
    ModelPricing,
    estimate_cost,
    get_cheapest_model,
    get_context_window,
)


class TestTokenCounter:
    @pytest.fixture
    def counter(self) -> TokenCounter:
        return TokenCounter(model="gpt-4o")

    def test_count_tokens_returns_positive(self, counter: TokenCounter) -> None:
        count = counter.count_tokens("Hello, world!")
        assert count > 0

    def test_count_tokens_empty_string(self, counter: TokenCounter) -> None:
        count = counter.count_tokens("")
        assert count >= 0

    def test_count_tokens_longer_text_more_tokens(self, counter: TokenCounter) -> None:
        short = counter.count_tokens("Hi")
        long = counter.count_tokens(
            "This is a much longer piece of text that should have more tokens than just hi"
        )
        assert long > short

    def test_count_tokens_from_messages(self, counter: TokenCounter) -> None:
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        count = counter.count_tokens_from_messages(messages)
        assert count > 0

    def test_count_tokens_from_messages_empty(self, counter: TokenCounter) -> None:
        count = counter.count_tokens_from_messages([])
        # Empty messages still has the priming tokens
        assert count >= 0

    def test_estimate_tokens(self, counter: TokenCounter) -> None:
        text = "Hello world, this is a test"
        estimate = counter.estimate_tokens(text)
        # ~4 chars per token
        assert estimate >= 0

    def test_estimate_tokens_approximately_correct(self, counter: TokenCounter) -> None:
        text = "A" * 40  # 40 chars → ~10 tokens
        estimate = counter.estimate_tokens(text)
        assert estimate == 10  # exactly 40 // 4

    def test_get_available_tokens(self, counter: TokenCounter) -> None:
        messages = [{"role": "user", "content": "test"}]
        available = counter.get_available_tokens(messages, reserved=100)
        assert isinstance(available, int)
        assert available >= 0

    def test_get_available_tokens_with_large_reserved(self, counter: TokenCounter) -> None:
        messages = [{"role": "user", "content": "test"}]
        # If reserved is larger than context window, returns 0
        available = counter.get_available_tokens(messages, reserved=1_000_000)
        assert available == 0

    def test_different_model_initialization(self) -> None:
        counter_gpt4 = TokenCounter(model="gpt-4")
        counter_claude = TokenCounter(model="claude-3-5-sonnet-20241022")
        assert counter_gpt4.model == "gpt-4"
        assert counter_claude.model == "claude-3-5-sonnet-20241022"

    def test_message_with_name_key(self, counter: TokenCounter) -> None:
        messages = [
            {"role": "user", "name": "Alice", "content": "Hello"},
        ]
        count = counter.count_tokens_from_messages(messages)
        assert count > 0


class TestCountTokensFunction:
    def test_count_tokens_function(self) -> None:
        count = count_tokens("Hello world", model="gpt-4o")
        assert count > 0

    def test_count_tokens_function_different_models(self) -> None:
        text = "Sample text for counting"
        count1 = count_tokens(text, model="gpt-4o")
        count2 = count_tokens(text, model="gpt-3.5-turbo")
        # Both should return a positive count
        assert count1 > 0
        assert count2 > 0

    def test_count_message_tokens_function(self) -> None:
        messages = [{"role": "user", "content": "test message"}]
        count = count_message_tokens(messages, model="gpt-4o")
        assert count > 0


class TestModelPricing:
    def test_get_pricing_exact_match(self) -> None:
        pricing = ModelPricing.get_pricing("gpt-4o")
        assert pricing is not None
        assert "input" in pricing
        assert "output" in pricing

    def test_get_pricing_prefix_match(self) -> None:
        # "gpt-4o-mini-2024-07-18" should match "gpt-4o-mini"
        pricing = ModelPricing.get_pricing("gpt-4o-mini-2024-07-18")
        assert pricing is not None

    def test_get_pricing_unknown_model(self) -> None:
        pricing = ModelPricing.get_pricing("unknown-model-xyz-99")
        assert pricing is None

    def test_anthropic_pricing_exists(self) -> None:
        pricing = ModelPricing.get_pricing("claude-3-5-sonnet-20241022")
        assert pricing is not None
        assert pricing["input"] > 0

    def test_google_pricing_exists(self) -> None:
        pricing = ModelPricing.get_pricing("gemini-1.5-pro")
        assert pricing is not None

    def test_ollama_pricing_is_free(self) -> None:
        pricing = ModelPricing.get_pricing("llama3.2")
        assert pricing is not None
        assert pricing["input"] == 0.0
        assert pricing["output"] == 0.0

    def test_all_models_dict_not_empty(self) -> None:
        assert len(ModelPricing.ALL_MODELS) > 0


class TestModelContextWindow:
    def test_get_context_window_known_model(self) -> None:
        window = ModelContextWindow.get_context_window("gpt-4o")
        assert window == 128000

    def test_get_context_window_claude(self) -> None:
        window = ModelContextWindow.get_context_window("claude-3-5-sonnet-20241022")
        assert window == 200000

    def test_get_context_window_gemini(self) -> None:
        window = ModelContextWindow.get_context_window("gemini-1.5-pro")
        assert window == 2000000

    def test_get_context_window_unknown_returns_default(self) -> None:
        window = ModelContextWindow.get_context_window("unknown-model-xyz")
        assert window > 0  # Returns CLAUDE_DEFAULT_MAX_TOKENS

    def test_get_context_window_prefix_match(self) -> None:
        # "llama3.2:latest" should match "llama3.2"
        window = ModelContextWindow.get_context_window("llama3.2:latest")
        assert window == 128000


class TestCostEstimate:
    def test_create_cost_estimate(self) -> None:
        estimate = CostEstimate(
            input_tokens=1000,
            output_tokens=500,
            input_cost=0.0025,
            output_cost=0.005,
            total_cost=0.0075,
            model="gpt-4o",
        )
        assert estimate.input_tokens == 1000
        assert estimate.output_tokens == 500
        assert estimate.total_cost == pytest.approx(0.0075)

    def test_cost_estimate_str(self) -> None:
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


class TestCostEstimator:
    @pytest.fixture
    def estimator(self) -> CostEstimator:
        return CostEstimator(model="gpt-4o")

    def test_estimate_cost_with_text(self, estimator: CostEstimator) -> None:
        result = estimator.estimate_cost(
            input_text="Hello world",
            output_text="Hi there!",
        )
        assert isinstance(result, CostEstimate)
        assert result.total_cost >= 0

    def test_estimate_cost_with_tokens(self, estimator: CostEstimator) -> None:
        result = estimator.estimate_cost(
            input_tokens=1000,
            output_tokens=500,
        )
        assert result.input_tokens == 1000
        assert result.output_tokens == 500

    def test_estimate_cost_with_messages(self, estimator: CostEstimator) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        result = estimator.estimate_cost(messages=messages)
        assert result.input_tokens > 0

    def test_estimate_cost_no_output(self, estimator: CostEstimator) -> None:
        result = estimator.estimate_cost(input_tokens=100)
        assert result.output_tokens == 0

    def test_estimate_cost_unknown_model(self) -> None:
        estimator = CostEstimator(model="unknown-model-xyz")
        result = estimator.estimate_cost(input_tokens=100, output_tokens=50)
        assert result.total_cost == 0.0

    def test_compare_models(self, estimator: CostEstimator) -> None:
        models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
        estimates = estimator.compare_models(models, "Hello world", output_tokens=100)
        assert len(estimates) == 3
        # Sorted by cost (cheapest first)
        costs = [e.total_cost for e in estimates]
        assert costs == sorted(costs)


class TestConvenienceFunctions:
    def test_estimate_cost_function(self) -> None:
        result = estimate_cost("Hello world", "Hi!", model="gpt-4o")
        assert isinstance(result, CostEstimate)
        assert result.model == "gpt-4o"

    def test_estimate_cost_no_output(self) -> None:
        result = estimate_cost("Hello", model="gpt-4o")
        assert result.total_cost >= 0

    def test_get_cheapest_model(self) -> None:
        models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
        cheapest = get_cheapest_model("Hello world", output_tokens=100, models=models)
        assert cheapest in models

    def test_get_cheapest_model_default_models(self) -> None:
        cheapest = get_cheapest_model("Hello world", output_tokens=100)
        assert isinstance(cheapest, str)

    def test_get_context_window_function(self) -> None:
        window = get_context_window("gpt-4o")
        assert window == 128000
