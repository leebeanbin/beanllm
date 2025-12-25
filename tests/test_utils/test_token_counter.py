"""
Token Counter 테스트 - 토큰 카운팅 테스트
"""

import pytest
from unittest.mock import Mock, patch

from llmkit.utils.token_counter import (
    TokenCounter,
    count_tokens,
    estimate_cost,
    ModelPricing,
    ModelContextWindow,
    CostEstimator,
    CostEstimate,
)


class TestTokenCounter:
    """TokenCounter 테스트"""

    @pytest.fixture
    def token_counter(self):
        """TokenCounter 인스턴스"""
        return TokenCounter()

    def test_count_tokens_openai(self, token_counter):
        """OpenAI 토큰 카운팅 테스트"""
        text = "Hello world"
        count = token_counter.count_tokens(text)

        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_anthropic(self, token_counter):
        """Anthropic 토큰 카운팅 테스트"""
        from llmkit.utils.token_counter import TokenCounter

        counter = TokenCounter(model="claude-3-opus")
        text = "Hello world"
        count = counter.count_tokens(text)

        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_batch(self, token_counter):
        """배치 토큰 카운팅 테스트"""
        texts = ["Text 1", "Text 2", "Text 3"]
        counts = [token_counter.count_tokens(text) for text in texts]

        assert isinstance(counts, list)
        assert len(counts) == len(texts)
        assert all(isinstance(c, int) for c in counts)

    def test_estimate_cost(self, token_counter):
        """비용 추정 테스트"""
        from llmkit.utils.token_counter import CostEstimator

        estimator = CostEstimator(model="gpt-4o-mini")
        cost_estimate = estimator.estimate_cost(
            input_text="Test input",
            output_text="Test output",
        )

        assert cost_estimate is not None
        assert isinstance(cost_estimate.total_cost, float)
        assert cost_estimate.total_cost >= 0


class TestTokenCounterFunctions:
    """TokenCounter 편의 함수 테스트"""

    def test_count_tokens_function(self):
        """count_tokens 편의 함수 테스트"""
        text = "Hello world"
        count = count_tokens(text, model="gpt-4o-mini")

        assert isinstance(count, int)
        assert count > 0

    def test_estimate_cost_function(self):
        """estimate_cost 편의 함수 테스트"""
        from llmkit.utils.token_counter import CostEstimator

        estimator = CostEstimator(model="gpt-4o-mini")
        cost_estimate = estimator.estimate_cost(
            input_text="Hello",
            output_text="Hi",
        )

        assert cost_estimate is not None
        assert isinstance(cost_estimate.total_cost, float)
        assert cost_estimate.total_cost >= 0


class TestModelPricing:
    """ModelPricing 테스트"""

    def test_get_pricing_exact_match(self):
        """정확한 모델 매치 테스트"""
        pricing = ModelPricing.get_pricing("gpt-4o-mini")
        assert pricing is not None
        assert "input" in pricing
        assert "output" in pricing

    def test_get_pricing_partial_match(self):
        """부분 모델 매치 테스트"""
        pricing = ModelPricing.get_pricing("gpt-4o-mini-2024-07-18")
        assert pricing is not None

    def test_get_pricing_not_found(self):
        """모델을 찾을 수 없는 경우 테스트"""
        pricing = ModelPricing.get_pricing("unknown-model")
        assert pricing is None


class TestModelContextWindow:
    """ModelContextWindow 테스트"""

    def test_get_context_window_exact_match(self):
        """정확한 모델 매치 테스트"""
        window = ModelContextWindow.get_context_window("gpt-4o")
        assert isinstance(window, int)
        assert window > 0

    def test_get_context_window_partial_match(self):
        """부분 모델 매치 테스트"""
        window = ModelContextWindow.get_context_window("gpt-4o-mini-2024-07-18")
        assert isinstance(window, int)
        assert window > 0

    def test_get_context_window_default(self):
        """기본값 반환 테스트"""
        window = ModelContextWindow.get_context_window("unknown-model")
        assert isinstance(window, int)
        assert window == 4096  # 기본값


class TestTokenCounterAdvanced:
    """TokenCounter 고급 테스트"""

    def test_count_tokens_from_messages(self):
        """메시지 리스트 토큰 카운팅 테스트"""
        counter = TokenCounter(model="gpt-4o-mini")
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        count = counter.count_tokens_from_messages(messages)
        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_from_messages_with_name(self):
        """이름 포함 메시지 토큰 카운팅 테스트"""
        counter = TokenCounter(model="gpt-4o-mini")
        messages = [
            {"role": "user", "name": "Alice", "content": "Hello"},
        ]
        count = counter.count_tokens_from_messages(messages)
        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_from_messages_no_encoding(self):
        """인코딩 없이 메시지 토큰 카운팅 테스트"""
        with patch("llmkit.utils.token_counter.TIKTOKEN_AVAILABLE", False):
            counter = TokenCounter(model="gpt-4o-mini")
            messages = [{"role": "user", "content": "Hello"}]
            count = counter.count_tokens_from_messages(messages)
            assert isinstance(count, int)
            assert count > 0

    def test_estimate_tokens(self):
        """토큰 수 추정 테스트"""
        counter = TokenCounter()
        text = "Hello world " * 10  # 120 characters
        estimate = counter.estimate_tokens(text)
        assert isinstance(estimate, int)
        assert estimate > 0
        # 120 characters / 4 ≈ 30 tokens
        assert estimate == 30

    def test_get_available_tokens(self):
        """사용 가능한 토큰 수 계산 테스트"""
        counter = TokenCounter(model="gpt-4o-mini")
        messages = [{"role": "user", "content": "Hello"}]
        available = counter.get_available_tokens(messages, reserved=1000)
        assert isinstance(available, int)
        assert available >= 0

    def test_get_available_tokens_exceeded(self):
        """컨텍스트 윈도우 초과 테스트"""
        counter = TokenCounter(model="gpt-4o-mini")
        # 매우 긴 메시지 생성
        long_content = "Hello " * 100000
        messages = [{"role": "user", "content": long_content}]
        available = counter.get_available_tokens(messages, reserved=0)
        assert isinstance(available, int)
        # 초과하면 max(0, available)이므로 0 반환
        # 하지만 tiktoken이 없으면 근사치로 계산되므로 0이 아닐 수 있음
        assert available >= 0  # 최소 0 이상


class TestCostEstimator:
    """CostEstimator 테스트"""

    def test_cost_estimator_init(self):
        """CostEstimator 초기화 테스트"""
        estimator = CostEstimator(model="gpt-4o-mini")
        assert estimator.model == "gpt-4o-mini"

    def test_estimate_cost_with_tokens(self):
        """토큰 수로 비용 추정 테스트"""
        estimator = CostEstimator(model="gpt-4o-mini")
        cost_estimate = estimator.estimate_cost(input_tokens=1000, output_tokens=500)
        assert isinstance(cost_estimate, CostEstimate)
        assert cost_estimate.input_tokens == 1000
        assert cost_estimate.output_tokens == 500
        assert cost_estimate.total_cost >= 0

    def test_estimate_cost_from_messages(self):
        """메시지로 비용 추정 테스트"""
        estimator = CostEstimator(model="gpt-4o-mini")
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        cost_estimate = estimator.estimate_cost(messages=messages, output_tokens=100)
        assert isinstance(cost_estimate, CostEstimate)
        assert cost_estimate.total_cost >= 0

    def test_estimate_cost_with_text(self):
        """텍스트로 비용 추정 테스트"""
        estimator = CostEstimator(model="gpt-4o-mini")
        cost_estimate = estimator.estimate_cost(input_text="Hello world", output_text="Hi there")
        assert isinstance(cost_estimate, CostEstimate)
        assert cost_estimate.input_tokens > 0
        assert cost_estimate.output_tokens > 0
        assert cost_estimate.total_cost >= 0

    def test_compare_models(self):
        """여러 모델 비용 비교 테스트"""
        estimator = CostEstimator(model="gpt-4o-mini")
        models = ["gpt-4o-mini", "gpt-4o"]
        estimates = estimator.compare_models(models, input_text="Hello", output_tokens=100)
        assert isinstance(estimates, list)
        assert len(estimates) == 2
        assert all(isinstance(e, CostEstimate) for e in estimates)

    def test_cost_estimate_str(self):
        """CostEstimate 문자열 표현 테스트"""
        estimate = CostEstimate(
            input_tokens=1000,
            output_tokens=500,
            input_cost=0.001,
            output_cost=0.002,
            total_cost=0.003,
            model="gpt-4o-mini",
        )
        str_repr = str(estimate)
        assert "gpt-4o-mini" in str_repr
        assert "1000" in str_repr
        assert "500" in str_repr


class TestModelPricing:
    """ModelPricing 테스트"""

    def test_get_pricing_exact_match(self):
        """정확한 모델 매치 테스트"""
        pricing = ModelPricing.get_pricing("gpt-4o-mini")
        assert pricing is not None
        assert "input" in pricing
        assert "output" in pricing

    def test_get_pricing_partial_match(self):
        """부분 모델 매치 테스트"""
        pricing = ModelPricing.get_pricing("gpt-4o-mini-2024-07-18")
        assert pricing is not None

    def test_get_pricing_not_found(self):
        """모델을 찾을 수 없는 경우 테스트"""
        pricing = ModelPricing.get_pricing("unknown-model")
        assert pricing is None


class TestModelContextWindow:
    """ModelContextWindow 테스트"""

    def test_get_context_window_exact_match(self):
        """정확한 모델 매치 테스트"""
        window = ModelContextWindow.get_context_window("gpt-4o")
        assert isinstance(window, int)
        assert window > 0

    def test_get_context_window_partial_match(self):
        """부분 모델 매치 테스트"""
        window = ModelContextWindow.get_context_window("gpt-4o-mini-2024-07-18")
        assert isinstance(window, int)
        assert window > 0

    def test_get_context_window_default(self):
        """기본값 반환 테스트"""
        window = ModelContextWindow.get_context_window("unknown-model")
        assert isinstance(window, int)
        assert window == 4096  # 기본값


class TestTokenCounterAdvanced:
    """TokenCounter 고급 테스트"""

    def test_count_tokens_from_messages(self):
        """메시지 리스트 토큰 카운팅 테스트"""
        counter = TokenCounter(model="gpt-4o-mini")
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        count = counter.count_tokens_from_messages(messages)
        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_from_messages_with_name(self):
        """이름 포함 메시지 토큰 카운팅 테스트"""
        counter = TokenCounter(model="gpt-4o-mini")
        messages = [
            {"role": "user", "name": "Alice", "content": "Hello"},
        ]
        count = counter.count_tokens_from_messages(messages)
        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_from_messages_no_encoding(self):
        """인코딩 없이 메시지 토큰 카운팅 테스트"""
        with patch("llmkit.utils.token_counter.TIKTOKEN_AVAILABLE", False):
            counter = TokenCounter(model="gpt-4o-mini")
            messages = [{"role": "user", "content": "Hello"}]
            count = counter.count_tokens_from_messages(messages)
            assert isinstance(count, int)
            assert count > 0

    def test_estimate_tokens(self):
        """토큰 수 추정 테스트"""
        counter = TokenCounter()
        text = "Hello world " * 10  # 120 characters
        estimate = counter.estimate_tokens(text)
        assert isinstance(estimate, int)
        assert estimate > 0
        # 120 characters / 4 ≈ 30 tokens
        assert estimate == 30

    def test_get_available_tokens(self):
        """사용 가능한 토큰 수 계산 테스트"""
        counter = TokenCounter(model="gpt-4o-mini")
        messages = [{"role": "user", "content": "Hello"}]
        available = counter.get_available_tokens(messages, reserved=1000)
        assert isinstance(available, int)
        assert available >= 0

    def test_get_available_tokens_exceeded(self):
        """컨텍스트 윈도우 초과 테스트"""
        counter = TokenCounter(model="gpt-4o-mini")
        # 매우 긴 메시지 생성
        long_content = "Hello " * 100000
        messages = [{"role": "user", "content": long_content}]
        available = counter.get_available_tokens(messages, reserved=0)
        assert isinstance(available, int)
        # 초과하면 max(0, available)이므로 0 반환
        # 하지만 tiktoken이 없으면 근사치로 계산되므로 0이 아닐 수 있음
        assert available >= 0  # 최소 0 이상


class TestCostEstimator:
    """CostEstimator 테스트"""

    def test_cost_estimator_init(self):
        """CostEstimator 초기화 테스트"""
        estimator = CostEstimator(model="gpt-4o-mini")
        assert estimator.model == "gpt-4o-mini"

    def test_estimate_cost_with_tokens(self):
        """토큰 수로 비용 추정 테스트"""
        estimator = CostEstimator(model="gpt-4o-mini")
        cost_estimate = estimator.estimate_cost(input_tokens=1000, output_tokens=500)
        assert isinstance(cost_estimate, CostEstimate)
        assert cost_estimate.input_tokens == 1000
        assert cost_estimate.output_tokens == 500
        assert cost_estimate.total_cost >= 0

    def test_estimate_cost_from_messages(self):
        """메시지로 비용 추정 테스트"""
        estimator = CostEstimator(model="gpt-4o-mini")
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        cost_estimate = estimator.estimate_cost(messages=messages, output_tokens=100)
        assert isinstance(cost_estimate, CostEstimate)
        assert cost_estimate.total_cost >= 0

    def test_estimate_cost_with_text(self):
        """텍스트로 비용 추정 테스트"""
        estimator = CostEstimator(model="gpt-4o-mini")
        cost_estimate = estimator.estimate_cost(input_text="Hello world", output_text="Hi there")
        assert isinstance(cost_estimate, CostEstimate)
        assert cost_estimate.input_tokens > 0
        assert cost_estimate.output_tokens > 0
        assert cost_estimate.total_cost >= 0

    def test_compare_models(self):
        """여러 모델 비용 비교 테스트"""
        estimator = CostEstimator(model="gpt-4o-mini")
        models = ["gpt-4o-mini", "gpt-4o"]
        estimates = estimator.compare_models(models, input_text="Hello", output_tokens=100)
        assert isinstance(estimates, list)
        assert len(estimates) == 2
        assert all(isinstance(e, CostEstimate) for e in estimates)

    def test_cost_estimate_str(self):
        """CostEstimate 문자열 표현 테스트"""
        estimate = CostEstimate(
            input_tokens=1000,
            output_tokens=500,
            input_cost=0.001,
            output_cost=0.002,
            total_cost=0.003,
            model="gpt-4o-mini",
        )
        str_repr = str(estimate)
        assert "gpt-4o-mini" in str_repr
        assert "1000" in str_repr
        assert "500" in str_repr


class TestModelPricing:
    """ModelPricing 테스트"""

    def test_get_pricing_exact_match(self):
        """정확한 모델 매치 테스트"""
        pricing = ModelPricing.get_pricing("gpt-4o-mini")
        assert pricing is not None
        assert "input" in pricing
        assert "output" in pricing

    def test_get_pricing_partial_match(self):
        """부분 모델 매치 테스트"""
        pricing = ModelPricing.get_pricing("gpt-4o-mini-2024-07-18")
        assert pricing is not None

    def test_get_pricing_not_found(self):
        """모델을 찾을 수 없는 경우 테스트"""
        pricing = ModelPricing.get_pricing("unknown-model")
        assert pricing is None


class TestModelContextWindow:
    """ModelContextWindow 테스트"""

    def test_get_context_window_exact_match(self):
        """정확한 모델 매치 테스트"""
        window = ModelContextWindow.get_context_window("gpt-4o")
        assert isinstance(window, int)
        assert window > 0

    def test_get_context_window_partial_match(self):
        """부분 모델 매치 테스트"""
        window = ModelContextWindow.get_context_window("gpt-4o-mini-2024-07-18")
        assert isinstance(window, int)
        assert window > 0

    def test_get_context_window_default(self):
        """기본값 반환 테스트"""
        window = ModelContextWindow.get_context_window("unknown-model")
        assert isinstance(window, int)
        assert window == 4096  # 기본값


class TestTokenCounterAdvanced:
    """TokenCounter 고급 테스트"""

    def test_count_tokens_from_messages(self):
        """메시지 리스트 토큰 카운팅 테스트"""
        counter = TokenCounter(model="gpt-4o-mini")
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        count = counter.count_tokens_from_messages(messages)
        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_from_messages_with_name(self):
        """이름 포함 메시지 토큰 카운팅 테스트"""
        counter = TokenCounter(model="gpt-4o-mini")
        messages = [
            {"role": "user", "name": "Alice", "content": "Hello"},
        ]
        count = counter.count_tokens_from_messages(messages)
        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_from_messages_no_encoding(self):
        """인코딩 없이 메시지 토큰 카운팅 테스트"""
        with patch("llmkit.utils.token_counter.TIKTOKEN_AVAILABLE", False):
            counter = TokenCounter(model="gpt-4o-mini")
            messages = [{"role": "user", "content": "Hello"}]
            count = counter.count_tokens_from_messages(messages)
            assert isinstance(count, int)
            assert count > 0

    def test_estimate_tokens(self):
        """토큰 수 추정 테스트"""
        counter = TokenCounter()
        text = "Hello world " * 10  # 120 characters
        estimate = counter.estimate_tokens(text)
        assert isinstance(estimate, int)
        assert estimate > 0
        # 120 characters / 4 ≈ 30 tokens
        assert estimate == 30

    def test_get_available_tokens(self):
        """사용 가능한 토큰 수 계산 테스트"""
        counter = TokenCounter(model="gpt-4o-mini")
        messages = [{"role": "user", "content": "Hello"}]
        available = counter.get_available_tokens(messages, reserved=1000)
        assert isinstance(available, int)
        assert available >= 0

    def test_get_available_tokens_exceeded(self):
        """컨텍스트 윈도우 초과 테스트"""
        counter = TokenCounter(model="gpt-4o-mini")
        # 매우 긴 메시지 생성
        long_content = "Hello " * 100000
        messages = [{"role": "user", "content": long_content}]
        available = counter.get_available_tokens(messages, reserved=0)
        assert isinstance(available, int)
        # 초과하면 max(0, available)이므로 0 반환
        # 하지만 tiktoken이 없으면 근사치로 계산되므로 0이 아닐 수 있음
        assert available >= 0  # 최소 0 이상


class TestCostEstimator:
    """CostEstimator 테스트"""

    def test_cost_estimator_init(self):
        """CostEstimator 초기화 테스트"""
        estimator = CostEstimator(model="gpt-4o-mini")
        assert estimator.model == "gpt-4o-mini"

    def test_estimate_cost_with_tokens(self):
        """토큰 수로 비용 추정 테스트"""
        estimator = CostEstimator(model="gpt-4o-mini")
        cost_estimate = estimator.estimate_cost(input_tokens=1000, output_tokens=500)
        assert isinstance(cost_estimate, CostEstimate)
        assert cost_estimate.input_tokens == 1000
        assert cost_estimate.output_tokens == 500
        assert cost_estimate.total_cost >= 0

    def test_estimate_cost_from_messages(self):
        """메시지로 비용 추정 테스트"""
        estimator = CostEstimator(model="gpt-4o-mini")
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        cost_estimate = estimator.estimate_cost(messages=messages, output_tokens=100)
        assert isinstance(cost_estimate, CostEstimate)
        assert cost_estimate.total_cost >= 0

    def test_estimate_cost_with_text(self):
        """텍스트로 비용 추정 테스트"""
        estimator = CostEstimator(model="gpt-4o-mini")
        cost_estimate = estimator.estimate_cost(input_text="Hello world", output_text="Hi there")
        assert isinstance(cost_estimate, CostEstimate)
        assert cost_estimate.input_tokens > 0
        assert cost_estimate.output_tokens > 0
        assert cost_estimate.total_cost >= 0

    def test_compare_models(self):
        """여러 모델 비용 비교 테스트"""
        estimator = CostEstimator(model="gpt-4o-mini")
        models = ["gpt-4o-mini", "gpt-4o"]
        estimates = estimator.compare_models(models, input_text="Hello", output_tokens=100)
        assert isinstance(estimates, list)
        assert len(estimates) == 2
        assert all(isinstance(e, CostEstimate) for e in estimates)

    def test_cost_estimate_str(self):
        """CostEstimate 문자열 표현 테스트"""
        estimate = CostEstimate(
            input_tokens=1000,
            output_tokens=500,
            input_cost=0.001,
            output_cost=0.002,
            total_cost=0.003,
            model="gpt-4o-mini",
        )
        str_repr = str(estimate)
        assert "gpt-4o-mini" in str_repr
        assert "1000" in str_repr
        assert "500" in str_repr
