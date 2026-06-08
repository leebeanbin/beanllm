"""Tests for infrastructure/distributed/cache_wrapper.py (SyncCacheWrapper)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from beanllm.infrastructure.distributed.cache_wrapper import SyncCacheWrapper, get_distributed_cache


def _make_wrapper(max_size: int = 50, ttl: int = 120):
    """Create a SyncCacheWrapper with a mocked async cache."""
    mock_cache = AsyncMock()
    mock_cache.get = AsyncMock(return_value=None)
    mock_cache.set = AsyncMock(return_value=None)
    mock_cache.delete = AsyncMock(return_value=None)
    mock_cache.clear = AsyncMock(return_value=None)
    with patch(
        "beanllm.infrastructure.distributed.cache_wrapper.get_cache", return_value=mock_cache
    ):
        wrapper = SyncCacheWrapper(max_size=max_size, ttl=ttl)
    return wrapper, mock_cache


def _make_non_running_loop(return_value=None):
    mock_loop = MagicMock()
    mock_loop.is_running.return_value = False
    mock_loop.run_until_complete.return_value = return_value
    return mock_loop


# ---------------------------------------------------------------------------
# _get_loop RuntimeError fallback (lines 39-41)
# ---------------------------------------------------------------------------


class TestGetLoopFallback:
    def test_creates_new_loop_when_get_event_loop_raises(self):
        wrapper, _ = _make_wrapper()
        wrapper._loop = None
        mock_new_loop = MagicMock()

        with (
            patch("asyncio.get_event_loop", side_effect=RuntimeError("no running event loop")),
            patch("asyncio.new_event_loop", return_value=mock_new_loop) as mock_new,
            patch("asyncio.set_event_loop") as mock_set,
        ):
            loop = wrapper._get_loop()

        mock_new.assert_called_once()
        mock_set.assert_called_once_with(mock_new_loop)
        assert loop is mock_new_loop

    def test_get_loop_cached_after_first_call(self):
        wrapper, _ = _make_wrapper()
        wrapper._loop = None
        mock_loop = MagicMock()

        with patch("asyncio.get_event_loop", return_value=mock_loop):
            loop1 = wrapper._get_loop()
            loop2 = wrapper._get_loop()

        assert loop1 is mock_loop
        assert loop2 is mock_loop


# ---------------------------------------------------------------------------
# get() (line 51: running loop returns None; line 53: non-running returns value)
# ---------------------------------------------------------------------------


class TestGet:
    async def test_get_returns_none_when_loop_is_running(self):
        wrapper, mock_cache = _make_wrapper()
        # In async test context, loop.is_running() is True → returns None (line 51)
        result = wrapper.get("key1")
        assert result is None

    def test_get_returns_value_when_loop_not_running(self):
        wrapper, mock_cache = _make_wrapper()
        mock_loop = _make_non_running_loop(return_value="cached_value")
        wrapper._loop = mock_loop

        result = wrapper.get("key1")

        assert result == "cached_value"
        mock_loop.run_until_complete.assert_called_once()


# ---------------------------------------------------------------------------
# set() (line 61: running loop skips; lines 64-71: not running; RuntimeError)
# ---------------------------------------------------------------------------


class TestSet:
    async def test_set_returns_when_loop_running(self):
        wrapper, mock_cache = _make_wrapper()
        # In async context, loop.is_running() True → returns without run_until_complete (line 61)
        wrapper.set("key", "value")
        # Verify mock_cache.set was not called (since we just returned)
        mock_cache.set.assert_not_called()

    def test_set_calls_run_until_complete_when_loop_not_running(self):
        wrapper, mock_cache = _make_wrapper()
        mock_loop = _make_non_running_loop()

        with patch("asyncio.get_event_loop", return_value=mock_loop):
            wrapper.set("key", "value", ttl=60)

        mock_loop.run_until_complete.assert_called_once()

    def test_set_creates_new_loop_on_runtime_error(self):
        wrapper, _ = _make_wrapper()
        mock_new_loop = MagicMock()
        mock_new_loop.run_until_complete.return_value = None

        with (
            patch("asyncio.get_event_loop", side_effect=RuntimeError("no loop")),
            patch("asyncio.new_event_loop", return_value=mock_new_loop),
            patch("asyncio.set_event_loop"),
        ):
            wrapper.set("key", "value")

        mock_new_loop.run_until_complete.assert_called_once()
        mock_new_loop.close.assert_called_once()


# ---------------------------------------------------------------------------
# delete() (lines 75-87)
# ---------------------------------------------------------------------------


class TestDelete:
    async def test_delete_returns_when_loop_running(self):
        wrapper, mock_cache = _make_wrapper()
        # In async context, loop.is_running() True → returns (line 78)
        wrapper.delete("key")
        mock_cache.delete.assert_not_called()

    def test_delete_calls_run_until_complete_when_loop_not_running(self):
        wrapper, _ = _make_wrapper()
        mock_loop = _make_non_running_loop()

        with patch("asyncio.get_event_loop", return_value=mock_loop):
            wrapper.delete("key")

        mock_loop.run_until_complete.assert_called_once()

    def test_delete_creates_new_loop_on_runtime_error(self):
        wrapper, _ = _make_wrapper()
        mock_new_loop = MagicMock()
        mock_new_loop.run_until_complete.return_value = None

        with (
            patch("asyncio.get_event_loop", side_effect=RuntimeError("no loop")),
            patch("asyncio.new_event_loop", return_value=mock_new_loop),
            patch("asyncio.set_event_loop"),
        ):
            wrapper.delete("key")

        mock_new_loop.run_until_complete.assert_called_once()
        mock_new_loop.close.assert_called_once()


# ---------------------------------------------------------------------------
# clear() (lines 91-103)
# ---------------------------------------------------------------------------


class TestClear:
    async def test_clear_returns_when_loop_running(self):
        wrapper, mock_cache = _make_wrapper()
        wrapper.clear()
        mock_cache.clear.assert_not_called()

    def test_clear_calls_run_until_complete_when_loop_not_running(self):
        wrapper, _ = _make_wrapper()
        mock_loop = _make_non_running_loop()

        with patch("asyncio.get_event_loop", return_value=mock_loop):
            wrapper.clear()

        mock_loop.run_until_complete.assert_called_once()

    def test_clear_creates_new_loop_on_runtime_error(self):
        wrapper, _ = _make_wrapper()
        mock_new_loop = MagicMock()
        mock_new_loop.run_until_complete.return_value = None

        with (
            patch("asyncio.get_event_loop", side_effect=RuntimeError("no loop")),
            patch("asyncio.new_event_loop", return_value=mock_new_loop),
            patch("asyncio.set_event_loop"),
        ):
            wrapper.clear()

        mock_new_loop.run_until_complete.assert_called_once()
        mock_new_loop.close.assert_called_once()


# ---------------------------------------------------------------------------
# stats() (line 108)
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_returns_dict_with_expected_keys(self):
        wrapper, _ = _make_wrapper()
        stats = wrapper.stats()
        assert isinstance(stats, dict)
        assert "size" in stats
        assert "hit_rate" in stats
        assert "ttl" in stats
        assert stats["hit_rate"] == 0.0
        assert stats["size"] == 0

    def test_stats_hits_always_zero(self):
        wrapper, _ = _make_wrapper()
        stats = wrapper.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0


# ---------------------------------------------------------------------------
# shutdown() (line 122)
# ---------------------------------------------------------------------------


class TestShutdown:
    def test_shutdown_does_not_raise(self):
        wrapper, _ = _make_wrapper()
        wrapper.shutdown()  # Just passes (line 122)

    def test_shutdown_can_be_called_multiple_times(self):
        wrapper, _ = _make_wrapper()
        wrapper.shutdown()
        wrapper.shutdown()


# ---------------------------------------------------------------------------
# get_distributed_cache() (line 139)
# ---------------------------------------------------------------------------


class TestGetDistributedCache:
    def test_returns_sync_cache_wrapper(self):
        with patch("beanllm.infrastructure.distributed.cache_wrapper.get_cache") as mock_gc:
            mock_gc.return_value = AsyncMock()
            result = get_distributed_cache(max_size=200, ttl=300)
        assert isinstance(result, SyncCacheWrapper)

    def test_accepts_default_arguments(self):
        with patch("beanllm.infrastructure.distributed.cache_wrapper.get_cache") as mock_gc:
            mock_gc.return_value = AsyncMock()
            result = get_distributed_cache()
        assert isinstance(result, SyncCacheWrapper)
