"""Tests for domain/embeddings/utils/cache.py — EmbeddingCache."""

from unittest.mock import MagicMock

import pytest

from beanllm.domain.embeddings.utils.cache import EmbeddingCache

# ---------------------------------------------------------------------------
# EmbeddingCache with default LRUCache
# ---------------------------------------------------------------------------


class TestEmbeddingCacheDefault:
    def setup_method(self):
        self.cache = EmbeddingCache(ttl=3600, max_size=100)

    def test_get_miss_returns_none(self):
        assert self.cache.get("nonexistent text") is None

    def test_set_and_get(self):
        vec = [0.1, 0.2, 0.3]
        self.cache.set("hello world", vec)
        result = self.cache.get("hello world")
        assert result == vec

    def test_different_texts_different_cache(self):
        self.cache.set("text A", [1.0, 0.0])
        self.cache.set("text B", [0.0, 1.0])
        assert self.cache.get("text A") == [1.0, 0.0]
        assert self.cache.get("text B") == [0.0, 1.0]

    def test_overwrite_same_key(self):
        self.cache.set("key", [1.0, 0.0])
        self.cache.set("key", [0.5, 0.5])
        assert self.cache.get("key") == [0.5, 0.5]

    def test_clear_removes_all(self):
        self.cache.set("a", [1.0])
        self.cache.set("b", [2.0])
        self.cache.clear()
        assert self.cache.get("a") is None
        assert self.cache.get("b") is None

    def test_stats_returns_dict(self):
        stats = self.cache.stats()
        assert isinstance(stats, dict)

    def test_ttl_and_max_size_stored(self):
        c = EmbeddingCache(ttl=1800, max_size=500)
        assert c.ttl == 1800
        assert c.max_size == 500

    def test_shutdown_no_crash(self):
        self.cache.shutdown()  # should not raise


# ---------------------------------------------------------------------------
# EmbeddingCache with injected mock cache
# ---------------------------------------------------------------------------


class TestEmbeddingCacheInjected:
    def setup_method(self):
        self.mock_cache = MagicMock()
        self.mock_cache.get.return_value = None
        self.mock_cache.stats.return_value = {"size": 0, "hit_rate": 0.0}
        self.cache = EmbeddingCache(cache=self.mock_cache)

    def test_get_delegates_to_injected_cache(self):
        self.cache.get("hello")
        self.mock_cache.get.assert_called_once_with("hello")

    def test_set_delegates_to_injected_cache(self):
        vec = [0.1, 0.2]
        self.cache.set("text", vec)
        self.mock_cache.set.assert_called_once_with("text", vec)

    def test_clear_delegates_to_injected_cache(self):
        self.cache.clear()
        self.mock_cache.clear.assert_called_once()

    def test_stats_delegates_to_injected_cache(self):
        self.cache.stats()
        self.mock_cache.stats.assert_called_once()

    def test_shutdown_delegates_to_injected_cache(self):
        self.cache.shutdown()
        self.mock_cache.shutdown.assert_called_once()

    def test_injected_get_returns_cached_value(self):
        self.mock_cache.get.return_value = [0.5, 0.6]
        result = self.cache.get("any text")
        assert result == [0.5, 0.6]


# ---------------------------------------------------------------------------
# LRU eviction behavior
# ---------------------------------------------------------------------------


class TestEmbeddingCacheLRUEviction:
    def test_evicts_when_max_size_exceeded(self):
        cache = EmbeddingCache(ttl=9999, max_size=2)
        cache.set("a", [1.0])
        cache.set("b", [2.0])
        cache.set("c", [3.0])  # should evict oldest (a)
        # At least one of the first two should be evicted
        assert cache.get("c") == [3.0]
        # b or a may be evicted; total stored should be ≤ 2
        stored = sum(1 for k in ["a", "b", "c"] if cache.get(k) is not None)
        assert stored <= 2
