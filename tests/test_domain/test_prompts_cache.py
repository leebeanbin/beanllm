"""Tests for domain/prompts/cache.py — PromptCache and module-level helpers."""

from unittest.mock import MagicMock, patch

import pytest

from beanllm.domain.prompts.cache import (
    PromptCache,
    clear_cache,
    get_cache_stats,
    get_cached_prompt,
)


class TestPromptCache:
    def test_get_returns_none_on_miss(self):
        cache = PromptCache()
        assert cache.get("nonexistent") is None

    def test_set_and_get_roundtrip(self):
        cache = PromptCache()
        cache.set("k1", "hello world")
        assert cache.get("k1") == "hello world"

    def test_get_stats_returns_dict(self):
        cache = PromptCache()
        stats = cache.get_stats()
        assert isinstance(stats, dict)

    def test_clear_removes_all_items(self):
        cache = PromptCache()
        cache.set("a", "1")
        cache.set("b", "2")
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_shutdown_does_not_raise(self):
        cache = PromptCache()
        cache.shutdown()  # Should not raise

    def test_injected_cache_is_used(self):
        mock_cache = MagicMock()
        mock_cache.get.return_value = "injected_result"
        mock_cache.stats.return_value = {"size": 1}

        cache = PromptCache(cache=mock_cache)

        result = cache.get("key")
        assert result == "injected_result"
        mock_cache.get.assert_called_once_with("key")

    def test_injected_cache_set_calls_through(self):
        mock_cache = MagicMock()
        cache = PromptCache(cache=mock_cache)

        cache.set("k", "v")
        mock_cache.set.assert_called_once_with("k", "v")

    def test_injected_cache_get_stats_calls_stats(self):
        mock_cache = MagicMock()
        mock_cache.stats.return_value = {"hits": 5}
        cache = PromptCache(cache=mock_cache)

        stats = cache.get_stats()
        assert stats == {"hits": 5}

    def test_injected_cache_clear_calls_clear(self):
        mock_cache = MagicMock()
        cache = PromptCache(cache=mock_cache)

        cache.clear()
        mock_cache.clear.assert_called_once()

    def test_injected_cache_shutdown_calls_shutdown(self):
        mock_cache = MagicMock()
        cache = PromptCache(cache=mock_cache)

        cache.shutdown()
        mock_cache.shutdown.assert_called_once()

    def test_del_calls_shutdown_silently(self):
        mock_cache = MagicMock()
        mock_cache.shutdown.side_effect = RuntimeError("shutdown error")
        cache = PromptCache(cache=mock_cache)
        # __del__ should not raise even if shutdown fails
        cache.__del__()

    def test_with_ttl_parameter(self):
        cache = PromptCache(ttl=3600)
        cache.set("ttl_key", "ttl_value")
        result = cache.get("ttl_key")
        assert result == "ttl_value"

    def test_with_max_size(self):
        cache = PromptCache(max_size=2)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        # Should not raise
        cache.set("k3", "v3")  # Eviction happens here


class TestGetCachedPrompt:
    def test_formats_template_when_not_cached(self):
        mock_template = MagicMock()
        mock_template.format.return_value = "formatted result"

        result = get_cached_prompt(mock_template, name="Alice")
        assert result == "formatted result"
        mock_template.format.assert_called_once_with(name="Alice")

    def test_returns_cached_value_on_second_call(self):
        mock_template = MagicMock()
        mock_template.format.return_value = "cached output"

        get_cached_prompt(mock_template, text="hello")
        # Second call with same args should use cache
        result = get_cached_prompt(mock_template, text="hello")
        assert result == "cached output"
        # format() called once (second call returns cached value)
        assert mock_template.format.call_count == 1

    def test_bypasses_cache_when_use_cache_false(self):
        mock_template = MagicMock()
        mock_template.format.return_value = "direct result"

        result = get_cached_prompt(mock_template, use_cache=False, text="hi")
        assert result == "direct result"
        mock_template.format.assert_called_once_with(text="hi")

    def test_different_kwargs_get_different_cache_entries(self):
        mock_template = MagicMock()
        mock_template.format.side_effect = lambda **kw: f"result_{kw}"

        r1 = get_cached_prompt(mock_template, text="a")
        r2 = get_cached_prompt(mock_template, text="b")
        assert r1 != r2


class TestModuleLevelHelpers:
    def test_get_cache_stats_returns_dict(self):
        stats = get_cache_stats()
        assert isinstance(stats, dict)

    def test_clear_cache_does_not_raise(self):
        clear_cache()  # Should not raise
