"""Tests for utils/core/cache.py (LRUCache TTL, eviction callbacks, cleanup)."""

import threading
import time
from unittest.mock import MagicMock

import pytest

from beanllm.utils.core.cache import LRUCache

# ---------------------------------------------------------------------------
# _start_cleanup_thread idempotency (line 121)
# ---------------------------------------------------------------------------


class TestStartCleanupThreadIdempotent:
    def test_second_call_reuses_existing_thread(self):
        cache = LRUCache(max_size=10, ttl=3600)
        first_thread = cache._cleanup_thread
        assert first_thread is not None
        cache._start_cleanup_thread()  # Must return early (line 121)
        assert cache._cleanup_thread is first_thread
        cache.shutdown()


# ---------------------------------------------------------------------------
# _cleanup_expired (lines 136-157)
# ---------------------------------------------------------------------------


class TestCleanupExpired:
    def test_cleanup_expired_when_ttl_none_returns_early(self):
        cache = LRUCache(max_size=10, ttl=None)
        cache._cache["key"] = ("val", time.time() - 1000)
        cache._cleanup_expired()  # ttl is None → returns immediately (line 136-137)
        assert "key" in cache._cache

    def test_cleanup_expired_removes_stale_entries(self):
        cache = LRUCache(max_size=100, ttl=3600)
        with cache._lock:
            cache._cache["stale"] = ("old_val", time.time() - 5000)
        cache._cleanup_expired()
        assert "stale" not in cache._cache
        cache.shutdown()

    def test_cleanup_expired_keeps_fresh_entries(self):
        cache = LRUCache(max_size=100, ttl=3600)
        with cache._lock:
            cache._cache["fresh"] = ("val", time.time())
        cache._cleanup_expired()
        assert "fresh" in cache._cache
        cache.shutdown()

    def test_cleanup_expired_calls_on_evict_callback(self):
        evicted = []

        def on_evict(k, v):
            evicted.append((k, v))

        cache = LRUCache(max_size=100, ttl=3600, on_evict=on_evict)
        with cache._lock:
            cache._cache["old"] = ("old_val", time.time() - 5000)
        cache._cleanup_expired()
        assert ("old", "old_val") in evicted
        cache.shutdown()

    def test_cleanup_expired_handles_evict_callback_exception(self):
        def bad_evict(k, v):
            raise RuntimeError("eviction exploded")

        cache = LRUCache(max_size=100, ttl=3600, on_evict=bad_evict)
        with cache._lock:
            cache._cache["old"] = ("val", time.time() - 5000)
        cache._cleanup_expired()  # Should not propagate the exception
        cache.shutdown()

    def test_cleanup_expired_increments_expirations_counter(self):
        cache = LRUCache(max_size=100, ttl=3600)
        with cache._lock:
            cache._cache["exp1"] = ("v1", time.time() - 5000)
            cache._cache["exp2"] = ("v2", time.time() - 5000)
        cache._cleanup_expired()
        assert cache._expirations >= 2
        cache.shutdown()


# ---------------------------------------------------------------------------
# get() with TTL expired and on_evict callback (lines 190-193)
# ---------------------------------------------------------------------------


class TestGetWithEvictionCallback:
    def test_get_calls_on_evict_when_expired(self):
        evicted = []

        def on_evict(k, v):
            evicted.append((k, v))

        cache = LRUCache(max_size=100, ttl=3600, on_evict=on_evict)
        with cache._lock:
            cache._cache["key"] = ("value", time.time() - 5000)
        result = cache.get("key")
        assert result is None
        assert ("key", "value") in evicted
        cache.shutdown()

    def test_get_handles_on_evict_exception_gracefully(self):
        def bad_evict(k, v):
            raise RuntimeError("evict failed")

        cache = LRUCache(max_size=100, ttl=3600, on_evict=bad_evict)
        with cache._lock:
            cache._cache["key"] = ("value", time.time() - 5000)
        result = cache.get("key")  # Should not raise
        assert result is None
        cache.shutdown()


# ---------------------------------------------------------------------------
# set() with LRU eviction and on_evict callback (lines 227-228)
# ---------------------------------------------------------------------------


class TestSetWithEvictionCallback:
    def test_set_calls_on_evict_on_lru_eviction(self):
        evicted = []

        def on_evict(k, v):
            evicted.append((k, v))

        cache = LRUCache(max_size=2, ttl=None, on_evict=on_evict)
        cache.set("a", "va")
        cache.set("b", "vb")
        cache.set("c", "vc")  # Evicts "a" (LRU)
        assert len(evicted) == 1
        assert evicted[0] == ("a", "va")

    def test_set_handles_on_evict_exception_gracefully(self):
        def bad_evict(k, v):
            raise RuntimeError("set eviction exploded")

        cache = LRUCache(max_size=1, ttl=None, on_evict=bad_evict)
        cache.set("first", "v1")
        cache.set("second", "v2")  # Evicts "first" → callback raises → no propagation


# ---------------------------------------------------------------------------
# delete() with on_evict callback (lines 254-257)
# ---------------------------------------------------------------------------


class TestDeleteWithEvictionCallback:
    def test_delete_calls_on_evict(self):
        evicted = []

        def on_evict(k, v):
            evicted.append((k, v))

        cache = LRUCache(max_size=100, ttl=None, on_evict=on_evict)
        cache.set("key", "val")
        result = cache.delete("key")
        assert result is True
        assert ("key", "val") in evicted

    def test_delete_handles_on_evict_exception_gracefully(self):
        def bad_evict(k, v):
            raise RuntimeError("delete eviction exploded")

        cache = LRUCache(max_size=100, ttl=None, on_evict=bad_evict)
        cache.set("key", "val")
        cache.delete("key")  # Should not raise


# ---------------------------------------------------------------------------
# clear() with on_evict callback (lines 272-273)
# ---------------------------------------------------------------------------


class TestClearWithEvictionCallback:
    def test_clear_calls_on_evict_for_each_entry(self):
        evicted = []

        def on_evict(k, v):
            evicted.append(k)

        cache = LRUCache(max_size=100, ttl=None, on_evict=on_evict)
        cache.set("x", 1)
        cache.set("y", 2)
        cache.clear()
        assert "x" in evicted
        assert "y" in evicted

    def test_clear_handles_on_evict_exception_gracefully(self):
        def bad_evict(k, v):
            raise RuntimeError("clear eviction exploded")

        cache = LRUCache(max_size=100, ttl=None, on_evict=bad_evict)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()  # Should not raise


# ---------------------------------------------------------------------------
# shutdown() with cleanup thread (lines 334-335: __del__ exception path)
# ---------------------------------------------------------------------------


class TestShutdown:
    def test_shutdown_joins_cleanup_thread(self):
        cache = LRUCache(max_size=10, ttl=3600)
        assert cache._cleanup_thread is not None
        cache.shutdown()
        assert cache._cleanup_thread is None

    def test_shutdown_when_no_cleanup_thread(self):
        cache = LRUCache(max_size=10, ttl=None)
        assert cache._cleanup_thread is None
        cache.shutdown()  # Should not raise

    def test_del_handles_shutdown_exception(self):
        cache = LRUCache(max_size=10, ttl=None)
        original_shutdown = cache.shutdown
        cache.shutdown = MagicMock(side_effect=RuntimeError("shutdown in del failed"))
        del cache  # Triggers __del__ which calls the mock shutdown → exception caught


# ---------------------------------------------------------------------------
# __contains__ with expired entry and on_evict (lines 358-361)
# ---------------------------------------------------------------------------


class TestContainsWithExpiredEntry:
    def test_contains_returns_false_for_expired_entry(self):
        cache = LRUCache(max_size=100, ttl=3600)
        with cache._lock:
            cache._cache["expired"] = ("val", time.time() - 5000)
        result = "expired" in cache  # Should remove and return False
        assert result is False
        cache.shutdown()

    def test_contains_calls_on_evict_for_expired_entry(self):
        evicted = []

        def on_evict(k, v):
            evicted.append(k)

        cache = LRUCache(max_size=100, ttl=3600, on_evict=on_evict)
        with cache._lock:
            cache._cache["exp_key"] = ("val", time.time() - 5000)
        "exp_key" in cache
        assert "exp_key" in evicted
        cache.shutdown()

    def test_contains_handles_on_evict_exception(self):
        def bad_evict(k, v):
            raise RuntimeError("contains eviction exploded")

        cache = LRUCache(max_size=100, ttl=3600, on_evict=bad_evict)
        with cache._lock:
            cache._cache["exp_key"] = ("val", time.time() - 5000)
        result = "exp_key" in cache  # Should not raise
        assert result is False
        cache.shutdown()
