"""Tests for infrastructure/distributed/lock_integration.py."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from beanllm.infrastructure.distributed.lock_integration import (
    LockManager,
    get_lock_manager,
    with_distributed_lock,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_lock():
    """Return a mock distributed lock whose acquire() context manager succeeds."""
    mock_lock = MagicMock()

    @asynccontextmanager
    async def _acquire(key, timeout=30.0):
        yield

    mock_lock.acquire = _acquire
    return mock_lock


def _mock_lock_raises():
    """Return a mock distributed lock whose acquire() raises."""
    mock_lock = MagicMock()

    @asynccontextmanager
    async def _acquire(key, timeout=30.0):
        raise RuntimeError("lock unavailable")
        yield  # noqa: unreachable — needed for asynccontextmanager

    mock_lock.acquire = _acquire
    return mock_lock


# ---------------------------------------------------------------------------
# with_distributed_lock – async function (lines 45-58)
# ---------------------------------------------------------------------------


class TestWithDistributedLockAsync:
    async def test_async_function_runs_with_lock(self):
        call_count = [0]

        with patch(
            "beanllm.infrastructure.distributed.lock_integration.get_distributed_lock",
            return_value=_mock_lock(),
        ):

            @with_distributed_lock("test:lock:key")
            async def fn():
                call_count[0] += 1
                return "result"

            result = await fn()

        assert result == "result"
        assert call_count[0] == 1

    async def test_async_function_propagates_exception(self):
        with patch(
            "beanllm.infrastructure.distributed.lock_integration.get_distributed_lock",
            return_value=_mock_lock(),
        ):

            @with_distributed_lock("test:lock:err")
            async def fn():
                raise ValueError("from function")

            with pytest.raises(ValueError, match="from function"):
                await fn()

    async def test_async_function_raises_when_lock_fails(self):
        with patch(
            "beanllm.infrastructure.distributed.lock_integration.get_distributed_lock",
            return_value=_mock_lock_raises(),
        ):

            @with_distributed_lock("test:lock:fail")
            async def fn():
                return "should not reach"

            with pytest.raises(RuntimeError, match="lock unavailable"):
                await fn()


# ---------------------------------------------------------------------------
# with_distributed_lock – sync function (lines 60-80)
# ---------------------------------------------------------------------------


class TestWithDistributedLockSync:
    async def test_sync_function_skips_lock_when_loop_running(self):
        call_count = [0]

        with patch(
            "beanllm.infrastructure.distributed.lock_integration.get_distributed_lock",
            return_value=_mock_lock(),
        ):

            @with_distributed_lock("sync:test:key")
            def fn(x):
                call_count[0] += 1
                return x * 2

            result = fn(5)  # In async context, loop is running → lock skipped (line 71)

        assert result == 10
        assert call_count[0] == 1

    def test_sync_function_runs_with_loop_not_running(self):
        mock_loop = MagicMock()
        mock_loop.is_running.return_value = False
        mock_loop.run_until_complete.return_value = "ok"

        with (
            patch(
                "beanllm.infrastructure.distributed.lock_integration.get_distributed_lock",
                return_value=_mock_lock(),
            ),
            patch("asyncio.get_event_loop", return_value=mock_loop),
        ):

            @with_distributed_lock("sync:no:loop")
            def fn():
                return "ok"

            fn()

        mock_loop.run_until_complete.assert_called_once()

    def test_sync_function_uses_asyncio_run_on_runtime_error(self):
        with (
            patch(
                "beanllm.infrastructure.distributed.lock_integration.get_distributed_lock",
                return_value=_mock_lock(),
            ),
            patch("asyncio.get_event_loop", side_effect=RuntimeError("no loop")),
            patch("asyncio.run", return_value="runtime_ok") as mock_run,
        ):

            @with_distributed_lock("sync:runtime:error")
            def fn():
                return "runtime_ok"

            fn()

        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# LockManager (lines 117-151)
# ---------------------------------------------------------------------------


class TestLockManager:
    async def test_acquire_resource_lock_succeeds(self):
        manager = LockManager()
        manager.lock = _mock_lock()

        async with manager.acquire_resource_lock("vector_store", "123"):
            pass  # Should not raise

    async def test_acquire_resource_lock_falls_back_on_exception(self):
        manager = LockManager()
        manager.lock = _mock_lock_raises()

        # Should not raise — falls back (yields even after exception)
        async with manager.acquire_resource_lock("vector_store", "broken"):
            pass  # fallback yield reached

    async def test_with_vector_store_lock_returns_context_manager(self):
        manager = LockManager()
        manager.lock = _mock_lock()

        ctx = await manager.with_vector_store_lock("store_xyz")
        async with ctx:
            pass

    def test_with_model_lock_returns_context_manager(self):
        manager = LockManager()
        manager.lock = _mock_lock()

        ctx = manager.with_model_lock("gpt-4o")
        assert ctx is not None

    def test_with_file_lock_returns_context_manager(self):
        manager = LockManager()
        manager.lock = _mock_lock()

        ctx = manager.with_file_lock("/tmp/test.txt")
        assert ctx is not None


# ---------------------------------------------------------------------------
# get_lock_manager (line 151)
# ---------------------------------------------------------------------------


class TestGetLockManager:
    def test_returns_lock_manager_instance(self):
        manager = get_lock_manager()
        assert isinstance(manager, LockManager)

    def test_returns_same_global_instance(self):
        m1 = get_lock_manager()
        m2 = get_lock_manager()
        assert m1 is m2
