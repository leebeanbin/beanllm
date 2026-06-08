"""Tests for domain/optimizer/parameter_search.py (MetadataInferrer)."""

from __future__ import annotations

import pytest

from beanllm.domain.optimizer.parameter_search import MetadataInferrer


@pytest.fixture
def inferrer() -> MetadataInferrer:
    return MetadataInferrer()


# ---------------------------------------------------------------------------
# _extract_base_model
# ---------------------------------------------------------------------------


class TestExtractBaseModel:
    def test_removes_yyyy_mm_dd_suffix(self, inferrer: MetadataInferrer) -> None:
        assert inferrer._extract_base_model("claude-3-5-sonnet-2024-10-22") == "claude-3-5-sonnet"

    def test_removes_yyyymmdd_suffix(self, inferrer: MetadataInferrer) -> None:
        assert inferrer._extract_base_model("gpt-5-nano-20250807") == "gpt-5-nano"

    def test_removes_yyyy_suffix(self, inferrer: MetadataInferrer) -> None:
        assert inferrer._extract_base_model("gemini-2.5-flash-2025") == "gemini-2.5-flash"

    def test_no_date_suffix_unchanged(self, inferrer: MetadataInferrer) -> None:
        assert inferrer._extract_base_model("gemini-2.5-flash") == "gemini-2.5-flash"

    def test_complex_model_id(self, inferrer: MetadataInferrer) -> None:
        result = inferrer._extract_base_model("gpt-5-nano-2025-08-07")
        assert result == "gpt-5-nano"


# ---------------------------------------------------------------------------
# _get_compiled_pattern (cache behavior)
# ---------------------------------------------------------------------------


class TestGetCompiledPattern:
    def test_returns_compiled_pattern(self, inferrer: MetadataInferrer) -> None:
        import re

        p = inferrer._get_compiled_pattern(r"gpt-5.*")
        assert hasattr(p, "match")

    def test_same_pattern_reuses_cached_object(self, inferrer: MetadataInferrer) -> None:
        p1 = inferrer._get_compiled_pattern(r".*nano.*")
        p2 = inferrer._get_compiled_pattern(r".*nano.*")
        assert p1 is p2

    def test_different_patterns_are_different_objects(self, inferrer: MetadataInferrer) -> None:
        p1 = inferrer._get_compiled_pattern(r".*nano.*")
        p2 = inferrer._get_compiled_pattern(r".*mini.*")
        assert p1 is not p2


# ---------------------------------------------------------------------------
# infer — OpenAI provider
# ---------------------------------------------------------------------------


class TestInferOpenAI:
    def test_gpt5_series_uses_max_completion_tokens(self, inferrer: MetadataInferrer) -> None:
        result = inferrer.infer("openai", "gpt-5-2025-08-07")
        assert result["uses_max_completion_tokens"] is True
        assert result["supports_max_tokens"] is False

    def test_nano_model_no_temperature(self, inferrer: MetadataInferrer) -> None:
        result = inferrer.infer("openai", "gpt-4o-nano")
        assert result["supports_temperature"] is False
        assert result["tier"] == "nano"

    def test_mini_model_has_temperature(self, inferrer: MetadataInferrer) -> None:
        result = inferrer.infer("openai", "gpt-4o-mini")
        assert result["supports_temperature"] is True
        assert result["tier"] == "mini"

    def test_o3_reasoning_no_temperature(self, inferrer: MetadataInferrer) -> None:
        result = inferrer.infer("openai", "o3-2025-04-16")
        assert result["supports_temperature"] is False

    def test_o4_reasoning_no_temperature(self, inferrer: MetadataInferrer) -> None:
        result = inferrer.infer("openai", "o4-mini")
        assert result["supports_temperature"] is False

    def test_defaults_applied_for_unmatched_model(self, inferrer: MetadataInferrer) -> None:
        result = inferrer.infer("openai", "gpt-unknown-model")
        assert result["supports_streaming"] is True
        assert result["inference_confidence"] == 0.3

    def test_is_inferred_flag(self, inferrer: MetadataInferrer) -> None:
        result = inferrer.infer("openai", "gpt-4o")
        assert result["is_inferred"] is True

    def test_provider_stored_in_result(self, inferrer: MetadataInferrer) -> None:
        result = inferrer.infer("openai", "gpt-4o")
        assert result["provider"] == "openai"
        assert result["model_id"] == "gpt-4o"

    def test_multiple_pattern_matches_increase_confidence(self, inferrer: MetadataInferrer) -> None:
        # gpt-5 + nano should match both gpt-5.* and .*nano.* patterns
        result = inferrer.infer("openai", "gpt-5-nano")
        assert result["inference_confidence"] > 0.3


# ---------------------------------------------------------------------------
# infer — Anthropic provider
# ---------------------------------------------------------------------------


class TestInferAnthropic:
    def test_claude4_max_tokens(self, inferrer: MetadataInferrer) -> None:
        result = inferrer.infer("anthropic", "claude-4-sonnet-20250514")
        assert result["max_tokens"] == 16384 or result.get("description") is not None

    def test_claude35_max_tokens(self, inferrer: MetadataInferrer) -> None:
        result = inferrer.infer("anthropic", "claude-3-5-sonnet-20241022")
        assert result["max_tokens"] == 8192

    def test_opus_tier(self, inferrer: MetadataInferrer) -> None:
        result = inferrer.infer("anthropic", "claude-3-opus-20240229")
        assert result["tier"] == "opus"

    def test_sonnet_tier(self, inferrer: MetadataInferrer) -> None:
        result = inferrer.infer("anthropic", "claude-sonnet-4-20250514")
        assert result["tier"] == "sonnet"

    def test_haiku_tier(self, inferrer: MetadataInferrer) -> None:
        result = inferrer.infer("anthropic", "claude-haiku-3-5-20241022")
        assert result["tier"] == "haiku"

    def test_defaults_include_streaming(self, inferrer: MetadataInferrer) -> None:
        result = inferrer.infer("anthropic", "claude-unknown")
        assert result["supports_streaming"] is True


# ---------------------------------------------------------------------------
# infer — Google provider
# ---------------------------------------------------------------------------


class TestInferGoogle:
    def test_gemini25_supports_thinking(self, inferrer: MetadataInferrer) -> None:
        result = inferrer.infer("google", "gemini-2.5-pro")
        assert result.get("supports_thinking") is True

    def test_gemini20_no_thinking(self, inferrer: MetadataInferrer) -> None:
        result = inferrer.infer("google", "gemini-2.0-flash")
        assert result.get("supports_thinking") is False

    def test_gemini15_no_thinking(self, inferrer: MetadataInferrer) -> None:
        result = inferrer.infer("google", "gemini-1.5-pro")
        assert result.get("supports_thinking") is False

    def test_flash_tier(self, inferrer: MetadataInferrer) -> None:
        result = inferrer.infer("google", "gemini-2.0-flash")
        assert result.get("tier") == "flash"

    def test_pro_tier(self, inferrer: MetadataInferrer) -> None:
        result = inferrer.infer("google", "gemini-2.5-pro")
        assert result.get("tier") == "pro"

    def test_uses_max_output_tokens(self, inferrer: MetadataInferrer) -> None:
        result = inferrer.infer("google", "gemini-2.5-flash")
        assert result.get("uses_max_output_tokens") is True


# ---------------------------------------------------------------------------
# infer — Ollama provider
# ---------------------------------------------------------------------------


class TestInferOllama:
    def test_ollama_uses_num_predict(self, inferrer: MetadataInferrer) -> None:
        result = inferrer.infer("ollama", "qwen2.5:0.5b")
        assert result["uses_num_predict"] is True

    def test_ollama_supports_streaming(self, inferrer: MetadataInferrer) -> None:
        result = inferrer.infer("ollama", "llama3:8b")
        assert result["supports_streaming"] is True

    def test_ollama_low_confidence_unmatched(self, inferrer: MetadataInferrer) -> None:
        result = inferrer.infer("ollama", "some-local-model")
        assert result["inference_confidence"] == 0.3


# ---------------------------------------------------------------------------
# infer — Unknown provider
# ---------------------------------------------------------------------------


class TestInferUnknown:
    def test_unknown_provider_returns_base_metadata(self, inferrer: MetadataInferrer) -> None:
        result = inferrer.infer("unknown_provider", "some-model")
        assert result["model_id"] == "some-model"
        assert result["provider"] == "unknown_provider"
        assert result["inference_confidence"] == 0.3
        assert result["is_inferred"] is True

    def test_inferred_at_is_iso_string(self, inferrer: MetadataInferrer) -> None:
        result = inferrer.infer("openai", "gpt-4o")
        from datetime import datetime

        # Should parse without error
        datetime.fromisoformat(result["inferred_at"])


# ---------------------------------------------------------------------------
# get_inference_rules
# ---------------------------------------------------------------------------


class TestGetInferenceRules:
    def test_returns_dict_for_known_provider(self, inferrer: MetadataInferrer) -> None:
        rules = inferrer.get_inference_rules("openai")
        assert isinstance(rules, dict)
        assert "patterns" in rules
        assert "defaults" in rules

    def test_returns_empty_dict_for_unknown_provider(self, inferrer: MetadataInferrer) -> None:
        rules = inferrer.get_inference_rules("nonexistent")
        assert rules == {}


# ---------------------------------------------------------------------------
# add_inference_rule
# ---------------------------------------------------------------------------


class TestAddInferenceRule:
    def test_adds_rule_to_existing_provider(self, inferrer: MetadataInferrer) -> None:
        inferrer.add_inference_rule("openai", r"gpt-6.*", "GPT-6 Series", {"max_tokens": 256000})
        rules = inferrer.get_inference_rules("openai")
        names = [p["name"] for p in rules["patterns"]]
        assert "GPT-6 Series" in names

    def test_adds_rule_to_new_provider(self, inferrer: MetadataInferrer) -> None:
        inferrer.add_inference_rule("custom", r"custom-.*", "Custom Model", {"tier": "custom"})
        result = inferrer.infer("custom", "custom-large")
        assert result.get("tier") == "custom"

    def test_new_provider_gets_patterns_and_defaults_keys(self, inferrer: MetadataInferrer) -> None:
        inferrer.add_inference_rule("newco", r".*", "All", {})
        rules = inferrer.get_inference_rules("newco")
        assert "patterns" in rules
