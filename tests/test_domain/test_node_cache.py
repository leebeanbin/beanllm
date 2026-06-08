"""Tests for domain/graph/node_cache.py"""

import hashlib
import json
from unittest.mock import MagicMock

import pytest

from beanllm.domain.graph.graph_state import GraphState
from beanllm.domain.graph.node_cache import NodeCache


def _state(**data) -> GraphState:
    return GraphState(data=data)


class TestNodeCacheInit:
    def test_default_init(self):
        cache = NodeCache()
        assert cache.max_size == 1000
        assert cache.ttl is None

    def test_custom_max_size_and_ttl(self):
        cache = NodeCache(max_size=50, ttl=300)
        assert cache.max_size == 50
        assert cache.ttl == 300

    def test_injected_cache_is_used(self):
        mock_cache = MagicMock()
        cache = NodeCache(cache=mock_cache)
        assert cache._cache is mock_cache

    def test_injected_cache_not_replaced_by_lru(self):
        mock_cache = MagicMock()
        cache = NodeCache(max_size=200, ttl=60, cache=mock_cache)
        # injected cache takes precedence over max_size/ttl
        assert cache._cache is mock_cache


class TestNodeCacheGetKey:
    def test_returns_string(self):
        cache = NodeCache()
        key = cache.get_key("node1", _state(x=1))
        assert isinstance(key, str)

    def test_includes_node_name(self):
        cache = NodeCache()
        key = cache.get_key("my_node", _state(x=1))
        assert key.startswith("my_node:")

    def test_deterministic(self):
        cache = NodeCache()
        state = _state(x=1, y=2)
        key1 = cache.get_key("n", state)
        key2 = cache.get_key("n", state)
        assert key1 == key2

    def test_same_state_different_node_different_key(self):
        cache = NodeCache()
        state = _state(x=1)
        key1 = cache.get_key("node_a", state)
        key2 = cache.get_key("node_b", state)
        assert key1 != key2

    def test_different_state_same_node_different_key(self):
        cache = NodeCache()
        key1 = cache.get_key("node", _state(x=1))
        key2 = cache.get_key("node", _state(x=2))
        assert key1 != key2

    def test_key_format_is_name_colon_md5(self):
        cache = NodeCache()
        state = _state(a=1)
        key = cache.get_key("step", state)
        expected_hash = hashlib.md5(json.dumps(state.data, sort_keys=True).encode()).hexdigest()
        assert key == f"step:{expected_hash}"

    def test_sort_keys_makes_key_order_independent(self):
        cache = NodeCache()
        # Build states with same items in different insertion orders
        state_a = GraphState(data={"x": 1, "y": 2})
        state_b = GraphState(data={"y": 2, "x": 1})
        assert cache.get_key("n", state_a) == cache.get_key("n", state_b)


class TestNodeCacheGetSet:
    def test_get_returns_none_on_miss(self):
        cache = NodeCache()
        result = cache.get("node", _state(x=1))
        assert result is None

    def test_set_then_get_returns_value(self):
        cache = NodeCache()
        state = _state(x=10)
        cache.set("process", state, {"output": "result"})
        retrieved = cache.get("process", state)
        assert retrieved == {"output": "result"}

    def test_different_node_same_state_miss(self):
        cache = NodeCache()
        state = _state(x=10)
        cache.set("node_a", state, "value_a")
        assert cache.get("node_b", state) is None

    def test_same_node_different_state_miss(self):
        cache = NodeCache()
        cache.set("node", _state(x=1), "val1")
        assert cache.get("node", _state(x=2)) is None

    def test_set_overwrites_existing_value(self):
        cache = NodeCache()
        state = _state(q="hello")
        cache.set("n", state, "old")
        cache.set("n", state, "new")
        assert cache.get("n", state) == "new"

    def test_set_stores_none_value(self):
        # None stored explicitly should still be findable via cache.get
        # but our get() returns None for both miss and explicit None
        # verify set is at least called on underlying cache
        mock_cache = MagicMock()
        mock_cache.get.return_value = "stored_none"
        nc = NodeCache(cache=mock_cache)
        state = _state()
        nc.set("n", state, None)
        mock_cache.set.assert_called_once()

    def test_multiple_nodes_and_states(self):
        cache = NodeCache()
        s1, s2 = _state(k=1), _state(k=2)
        cache.set("a", s1, "a1")
        cache.set("a", s2, "a2")
        cache.set("b", s1, "b1")
        assert cache.get("a", s1) == "a1"
        assert cache.get("a", s2) == "a2"
        assert cache.get("b", s1) == "b1"
        assert cache.get("b", s2) is None


class TestNodeCacheClear:
    def test_clear_removes_all_entries(self):
        cache = NodeCache()
        state = _state(x=1)
        cache.set("n1", state, "v1")
        cache.set("n2", state, "v2")
        cache.clear()
        assert cache.get("n1", state) is None
        assert cache.get("n2", state) is None

    def test_clear_delegates_to_underlying_cache(self):
        mock_cache = MagicMock()
        nc = NodeCache(cache=mock_cache)
        nc.clear()
        mock_cache.clear.assert_called_once()


class TestNodeCacheStats:
    def test_get_stats_delegates_to_cache(self):
        mock_cache = MagicMock()
        mock_cache.stats.return_value = {"size": 5, "hits": 3, "misses": 2}
        nc = NodeCache(cache=mock_cache)
        stats = nc.get_stats()
        assert stats["size"] == 5
        mock_cache.stats.assert_called_once()

    def test_get_stats_returns_dict(self):
        cache = NodeCache()
        stats = cache.get_stats()
        assert isinstance(stats, dict)


class TestNodeCacheShutdown:
    def test_shutdown_delegates_to_cache(self):
        mock_cache = MagicMock()
        nc = NodeCache(cache=mock_cache)
        nc.shutdown()
        mock_cache.shutdown.assert_called_once()

    def test_double_shutdown_safe(self):
        cache = NodeCache()
        cache.shutdown()
        cache.shutdown()  # second call should not raise

    def test_del_calls_shutdown_safely(self):
        mock_cache = MagicMock()
        nc = NodeCache(cache=mock_cache)
        nc.__del__()
        mock_cache.shutdown.assert_called_once()

    def test_del_suppresses_shutdown_error(self):
        mock_cache = MagicMock()
        mock_cache.shutdown.side_effect = RuntimeError("already closed")
        nc = NodeCache(cache=mock_cache)
        nc.__del__()  # should not propagate the error
