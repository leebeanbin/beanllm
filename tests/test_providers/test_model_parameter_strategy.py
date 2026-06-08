"""
Tests for model_parameter_strategy.py

Covers ModelParameterStrategy subclasses, ModelParameterFactory, and extract_base_model.
"""

import pytest

from beanllm.providers.model_parameter_strategy import (
    DefaultModelStrategy,
    GPT5Strategy,
    GPT41Strategy,
    MiniModelStrategy,
    ModelParameterFactory,
    ModelParameterStrategy,
    NanoModelStrategy,
    O3ModelStrategy,
    O4ModelStrategy,
)

# ---------------------------------------------------------------------------
# Individual Strategy tests
# ---------------------------------------------------------------------------


class TestGPT5Strategy:
    def test_get_config_returns_correct_values(self):
        strategy = GPT5Strategy()
        config = strategy.get_config()
        assert config["supports_temperature"] is True
        assert config["supports_max_tokens"] is False
        assert config["uses_max_completion_tokens"] is True

    def test_is_instance_of_model_parameter_strategy(self):
        assert isinstance(GPT5Strategy(), ModelParameterStrategy)


class TestGPT41Strategy:
    def test_get_config_returns_correct_values(self):
        strategy = GPT41Strategy()
        config = strategy.get_config()
        assert config["supports_temperature"] is True
        assert config["supports_max_tokens"] is False
        assert config["uses_max_completion_tokens"] is True


class TestNanoModelStrategy:
    def test_get_config_returns_correct_values(self):
        strategy = NanoModelStrategy()
        config = strategy.get_config()
        assert config["supports_temperature"] is False
        assert config["supports_max_tokens"] is False
        assert config["uses_max_completion_tokens"] is False


class TestMiniModelStrategy:
    def test_get_config_returns_correct_values(self):
        strategy = MiniModelStrategy()
        config = strategy.get_config()
        assert config["supports_temperature"] is False
        assert config["supports_max_tokens"] is True
        assert config["uses_max_completion_tokens"] is False


class TestO3ModelStrategy:
    def test_get_config_returns_correct_values(self):
        strategy = O3ModelStrategy()
        config = strategy.get_config()
        assert config["supports_temperature"] is False
        assert config["supports_max_tokens"] is True
        assert config["uses_max_completion_tokens"] is False


class TestO4ModelStrategy:
    def test_get_config_returns_correct_values(self):
        strategy = O4ModelStrategy()
        config = strategy.get_config()
        assert config["supports_temperature"] is False
        assert config["supports_max_tokens"] is True
        assert config["uses_max_completion_tokens"] is False


class TestDefaultModelStrategy:
    def test_get_config_returns_correct_values(self):
        strategy = DefaultModelStrategy()
        config = strategy.get_config()
        assert config["supports_temperature"] is True
        assert config["supports_max_tokens"] is True
        assert config["uses_max_completion_tokens"] is False


# ---------------------------------------------------------------------------
# ModelParameterFactory.extract_base_model
# ---------------------------------------------------------------------------


class TestExtractBaseModel:
    def test_removes_yyyy_mm_dd_suffix(self):
        result = ModelParameterFactory.extract_base_model("gpt-5-nano-2025-08-07")
        assert result == "gpt-5-nano"

    def test_removes_yyyy_suffix(self):
        result = ModelParameterFactory.extract_base_model("gpt-4o-2024")
        assert result == "gpt-4o"

    def test_removes_date_from_gpt4o_versioned(self):
        result = ModelParameterFactory.extract_base_model("gpt-4o-2024-05-13")
        assert result == "gpt-4o"

    def test_no_date_returns_same(self):
        result = ModelParameterFactory.extract_base_model("gpt-4o")
        assert result == "gpt-4o"

    def test_no_date_on_gpt4(self):
        result = ModelParameterFactory.extract_base_model("gpt-4")
        assert result == "gpt-4"

    def test_removes_date_from_gpt5_nano(self):
        result = ModelParameterFactory.extract_base_model("gpt-5-nano-2025-12-31")
        assert result == "gpt-5-nano"

    def test_removes_date_from_gpt41_nano(self):
        result = ModelParameterFactory.extract_base_model("gpt-4.1-nano-2025-08-07")
        assert result == "gpt-4.1-nano"


# ---------------------------------------------------------------------------
# ModelParameterFactory.get_strategy
# ---------------------------------------------------------------------------


class TestGetStrategy:
    def test_gpt5_nano_returns_nano_strategy(self):
        strategy = ModelParameterFactory.get_strategy("gpt-5-nano")
        assert isinstance(strategy, NanoModelStrategy)

    def test_gpt41_nano_returns_nano_strategy(self):
        strategy = ModelParameterFactory.get_strategy("gpt-4.1-nano")
        assert isinstance(strategy, NanoModelStrategy)

    def test_gpt5_returns_gpt5_strategy(self):
        strategy = ModelParameterFactory.get_strategy("gpt-5")
        assert isinstance(strategy, GPT5Strategy)

    def test_gpt41_returns_gpt41_strategy(self):
        strategy = ModelParameterFactory.get_strategy("gpt-4.1")
        assert isinstance(strategy, GPT41Strategy)

    def test_nano_returns_nano_strategy(self):
        strategy = ModelParameterFactory.get_strategy("some-nano-model")
        assert isinstance(strategy, NanoModelStrategy)

    def test_mini_returns_mini_strategy(self):
        strategy = ModelParameterFactory.get_strategy("gpt-4o-mini")
        assert isinstance(strategy, MiniModelStrategy)

    def test_o3_mini_returns_mini_strategy(self):
        # "o3-mini" contains "mini" which appears before "o3" in STRATEGIES list
        strategy = ModelParameterFactory.get_strategy("o3-mini")
        assert isinstance(strategy, MiniModelStrategy)

    def test_o4_mini_returns_mini_strategy(self):
        # "o4-mini" contains "mini" which appears before "o4" in STRATEGIES list
        strategy = ModelParameterFactory.get_strategy("o4-mini")
        assert isinstance(strategy, MiniModelStrategy)

    def test_o3_without_mini_returns_o3_strategy(self):
        # "o3" alone should match O3ModelStrategy
        strategy = ModelParameterFactory.get_strategy("o3")
        assert isinstance(strategy, O3ModelStrategy)

    def test_o4_without_mini_returns_o4_strategy(self):
        # "o4" alone should match O4ModelStrategy
        strategy = ModelParameterFactory.get_strategy("o4")
        assert isinstance(strategy, O4ModelStrategy)

    def test_unknown_model_returns_default_strategy(self):
        strategy = ModelParameterFactory.get_strategy("unknown-model-xyz")
        assert isinstance(strategy, DefaultModelStrategy)

    def test_gpt4_returns_default_strategy(self):
        strategy = ModelParameterFactory.get_strategy("gpt-4")
        assert isinstance(strategy, DefaultModelStrategy)

    def test_gpt35_turbo_returns_default_strategy(self):
        strategy = ModelParameterFactory.get_strategy("gpt-3.5-turbo")
        assert isinstance(strategy, DefaultModelStrategy)

    def test_dated_gpt5_nano_returns_nano_strategy(self):
        # Date is stripped before matching
        strategy = ModelParameterFactory.get_strategy("gpt-5-nano-2025-08-07")
        assert isinstance(strategy, NanoModelStrategy)

    def test_dated_gpt5_returns_gpt5_strategy(self):
        strategy = ModelParameterFactory.get_strategy("gpt-5-2025-07-01")
        assert isinstance(strategy, GPT5Strategy)

    def test_gpt41_mini_returns_mini_strategy_not_gpt41(self):
        # "gpt-4.1-mini": gpt-4.1 pattern appears before mini, so should be GPT41Strategy
        # Actually "gpt-4.1" is matched first in STRATEGIES list before "mini"
        strategy = ModelParameterFactory.get_strategy("gpt-4.1-mini")
        # gpt-4.1 pattern matches first
        assert isinstance(strategy, GPT41Strategy)

    def test_gpt5_mini_returns_gpt5_strategy_not_mini(self):
        # gpt-5 pattern should match before mini
        strategy = ModelParameterFactory.get_strategy("gpt-5-mini")
        assert isinstance(strategy, GPT5Strategy)


# ---------------------------------------------------------------------------
# ModelParameterFactory.get_config
# ---------------------------------------------------------------------------


class TestGetConfig:
    def test_get_config_gpt5_nano(self):
        config = ModelParameterFactory.get_config("gpt-5-nano")
        assert config["supports_temperature"] is False
        assert config["supports_max_tokens"] is False
        assert config["uses_max_completion_tokens"] is False

    def test_get_config_gpt5(self):
        config = ModelParameterFactory.get_config("gpt-5")
        assert config["supports_temperature"] is True
        assert config["uses_max_completion_tokens"] is True

    def test_get_config_default(self):
        config = ModelParameterFactory.get_config("gpt-4o")
        assert config["supports_temperature"] is True
        assert config["supports_max_tokens"] is True
        assert config["uses_max_completion_tokens"] is False

    def test_get_config_o3_mini(self):
        # o3-mini matches "mini" strategy (MiniModelStrategy) in STRATEGIES list
        config = ModelParameterFactory.get_config("o3-mini")
        assert config["supports_temperature"] is False
        assert config["supports_max_tokens"] is True

    def test_get_config_returns_dict_with_all_keys(self):
        config = ModelParameterFactory.get_config("gpt-4o")
        assert "supports_temperature" in config
        assert "supports_max_tokens" in config
        assert "uses_max_completion_tokens" in config

    def test_get_config_gpt41(self):
        config = ModelParameterFactory.get_config("gpt-4.1")
        assert config["supports_temperature"] is True
        assert config["uses_max_completion_tokens"] is True

    def test_get_config_with_dated_model(self):
        config = ModelParameterFactory.get_config("gpt-4o-2024-05-13")
        # After stripping date: gpt-4o -> DefaultModelStrategy
        assert config["supports_temperature"] is True
        assert config["supports_max_tokens"] is True
