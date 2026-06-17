"""Tests for infrastructure/distributed/cache_wrapper.py (SyncCacheWrapper)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from beanllm.infrastructure.distributed.cache_wrapper import (
    SyncCacheWrapper,
    _run_coro,
    get_distributed_cache,
)


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


# ---------------------------------------------------------------------------
# _run_coro: running loop → None; no loop → asyncio.run result
# ---------------------------------------------------------------------------


class TestRunCoro:
    async def test_returns_none_when_loop_is_running(self):
        """In async context get_running_loop() succeeds → coro is closed, None returned."""
        mock_coro = AsyncMock(return_value="value")()
        result = _run_coro(mock_coro)
        assert result is None

    def test_runs_coro_when_no_loop(self):
        """Outside async context asyncio.run() executes the coroutine."""

        async def _coro():
            return "hello"

        result = _run_coro(_coro())
        assert result == "hello"


# ---------------------------------------------------------------------------
# get() — running loop returns None; no loop executes
# ---------------------------------------------------------------------------


class TestGet:
    async def test_get_returns_none_when_loop_is_running(self):
        """get() returns None when called from inside an async context."""
        wrapper, _ = _make_wrapper()
        result = wrapper.get("key1")
        assert result is None

    def test_get_returns_value_when_no_loop(self):
        """get() executes the coroutine when there is no running loop."""
        wrapper, mock_cache = _make_wrapper()
        mock_cache.get.return_value = "cached_value"

        with patch(
            "beanllm.infrastructure.distributed.cache_wrapper._run_coro",
            return_value="cached_value",
        ):
            result = wrapper.get("key1")

        assert result == "cached_value"


# ---------------------------------------------------------------------------
# set() — running loop is a no-op; no loop executes
# ---------------------------------------------------------------------------


class TestSet:
    async def test_set_no_op_when_loop_running(self):
        """set() closes coro without awaiting when called from async context."""
        wrapper, mock_cache = _make_wrapper()
        wrapper.set("key", "value")
        # coro was created but closed (not awaited) — get_running_loop() succeeds
        mock_cache.set.assert_not_awaited()

    def test_set_executes_when_no_loop(self):
        """set() calls _run_coro when there is no running loop."""
        wrapper, _ = _make_wrapper()
        with patch(
            "beanllm.infrastructure.distributed.cache_wrapper._run_coro", return_value=None
        ) as mock_run:
            wrapper.set("key", "value", ttl=60)
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


class TestDelete:
    async def test_delete_no_op_when_loop_running(self):
        """delete() closes coro without awaiting when called from async context."""
        wrapper, mock_cache = _make_wrapper()
        wrapper.delete("key")
        mock_cache.delete.assert_not_awaited()

    def test_delete_executes_when_no_loop(self):
        """delete() calls _run_coro when there is no running loop."""
        wrapper, _ = _make_wrapper()
        with patch(
            "beanllm.infrastructure.distributed.cache_wrapper._run_coro", return_value=None
        ) as mock_run:
            wrapper.delete("key")
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------


class TestClear:
    async def test_clear_no_op_when_loop_running(self):
        """clear() closes coro without awaiting when called from async context."""
        wrapper, mock_cache = _make_wrapper()
        wrapper.clear()
        mock_cache.clear.assert_not_awaited()

    def test_clear_executes_when_no_loop(self):
        """clear() calls _run_coro when there is no running loop."""
        wrapper, _ = _make_wrapper()
        with patch(
            "beanllm.infrastructure.distributed.cache_wrapper._run_coro", return_value=None
        ) as mock_run:
            wrapper.clear()
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# stats()
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
# shutdown()
# ---------------------------------------------------------------------------


class TestShutdown:
    def test_shutdown_does_not_raise(self):
        wrapper, _ = _make_wrapper()
        wrapper.shutdown()

    def test_shutdown_can_be_called_multiple_times(self):
        wrapper, _ = _make_wrapper()
        wrapper.shutdown()
        wrapper.shutdown()


# ---------------------------------------------------------------------------
# get_distributed_cache()
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
