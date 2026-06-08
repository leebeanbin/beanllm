"""Tests for utils/async_helpers.py — run_async_in_sync, AsyncHelperMixin, standalone functions."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from beanllm.utils.async_helpers import (
    AsyncHelperMixin,
    get_cached_sync,
    log_event_sync,
    run_async_in_sync,
    set_cache_sync,
)

# ---------------------------------------------------------------------------
# run_async_in_sync
# ---------------------------------------------------------------------------


class TestRunAsyncInSync:
    def test_runs_coroutine_and_returns_result(self):
        async def coro():
            return 42

        result = run_async_in_sync(coro())
        assert result == 42

    def test_returns_none_for_none_result(self):
        async def coro():
            return None

        result = run_async_in_sync(coro())
        assert result is None

    def test_runs_string_coroutine(self):
        async def coro():
            return "hello"

        result = run_async_in_sync(coro())
        assert result == "hello"

    async def test_running_loop_fire_and_forget(self):
        completed = []

        async def coro():
            completed.append(True)

        # When called inside a running loop, returns None (fire-and-forget)
        result = run_async_in_sync(coro())
        # In a running loop context, returns None immediately
        assert result is None


# ---------------------------------------------------------------------------
# log_event_sync
# ---------------------------------------------------------------------------


class TestLogEventSync:
    def test_none_logger_does_nothing(self):
        log_event_sync(None, "test.event", {"key": "value"})  # no crash

    def test_calls_log_event_on_logger(self):
        async def fake_log(event, data):
            pass

        mock_logger = MagicMock()
        mock_logger.log_event = MagicMock(side_effect=lambda e, d: fake_log(e, d))
        log_event_sync(mock_logger, "test.event", {"key": "value"})
        mock_logger.log_event.assert_called_once_with("test.event", {"key": "value"})


# ---------------------------------------------------------------------------
# get_cached_sync
# ---------------------------------------------------------------------------


class TestGetCachedSync:
    def test_none_cache_returns_default(self):
        result = get_cached_sync(None, "key", default="fallback")
        assert result == "fallback"

    def test_none_cache_returns_none_by_default(self):
        result = get_cached_sync(None, "key")
        assert result is None

    def test_returns_cached_value(self):
        async def get_fn(key):
            return "cached_value"

        mock_cache = MagicMock()
        mock_cache.get = MagicMock(side_effect=lambda key: get_fn(key))
        result = get_cached_sync(mock_cache, "key")
        assert result == "cached_value"

    def test_returns_default_on_cache_exception(self):
        mock_cache = MagicMock()
        mock_cache.get = MagicMock(side_effect=Exception("cache error"))
        result = get_cached_sync(mock_cache, "key", default="default_val")
        assert result == "default_val"

    def test_returns_default_when_cached_value_is_none(self):
        async def get_fn(key):
            return None

        mock_cache = MagicMock()
        mock_cache.get = MagicMock(side_effect=lambda key: get_fn(key))
        result = get_cached_sync(mock_cache, "key", default="fallback")
        assert result == "fallback"


# ---------------------------------------------------------------------------
# set_cache_sync
# ---------------------------------------------------------------------------


class TestSetCacheSync:
    def test_none_cache_does_nothing(self):
        set_cache_sync(None, "key", "value")  # no crash

    def test_calls_set_on_cache(self):
        async def set_fn(key, value):
            pass

        mock_cache = MagicMock()
        mock_cache.set = MagicMock(side_effect=lambda k, v: set_fn(k, v))
        set_cache_sync(mock_cache, "key", "value")
        mock_cache.set.assert_called_once_with("key", "value")

    def test_calls_set_with_ttl(self):
        async def set_fn(key, value, ttl=None):
            pass

        mock_cache = MagicMock()
        mock_cache.set = MagicMock(side_effect=lambda k, v, ttl=None: set_fn(k, v, ttl=ttl))
        set_cache_sync(mock_cache, "key", "value", ttl=3600)
        mock_cache.set.assert_called_once_with("key", "value", ttl=3600)

    def test_handles_cache_exception_silently(self):
        # The try/except in set_cache_sync wraps run_async_in_sync, not cache.set itself.
        # Make cache.set return a coroutine that raises to test the catch.
        async def failing_set(key, value):
            raise RuntimeError("async set failed")

        mock_cache = MagicMock()
        mock_cache.set = MagicMock(side_effect=lambda k, v: failing_set(k, v))
        set_cache_sync(mock_cache, "key", "value")  # should not raise


# ---------------------------------------------------------------------------
# AsyncHelperMixin
# ---------------------------------------------------------------------------


class ConcreteHelper(AsyncHelperMixin):
    def __init__(self, event_logger=None, cache=None):
        self._event_logger = event_logger
        self._cache = cache


class TestAsyncHelperMixin:
    def test_log_event_no_logger_does_nothing(self):
        helper = ConcreteHelper(event_logger=None)
        helper._log_event("test.event", {"data": "value"})  # no crash

    def test_log_event_calls_logger(self):
        async def log_fn(event, data):
            pass

        mock_logger = MagicMock()
        mock_logger.log_event = MagicMock(side_effect=lambda e, d: log_fn(e, d))
        helper = ConcreteHelper(event_logger=mock_logger)
        helper._log_event("test.event", {"data": "value"})
        mock_logger.log_event.assert_called_once()

    def test_log_event_fire_and_forget_false(self):
        async def log_fn(event, data):
            pass

        mock_logger = MagicMock()
        mock_logger.log_event = MagicMock(side_effect=lambda e, d: log_fn(e, d))
        helper = ConcreteHelper(event_logger=mock_logger)
        helper._log_event("test.event", {"data": "value"}, fire_and_forget=False)
        mock_logger.log_event.assert_called_once()

    def test_get_cached_no_cache_returns_default(self):
        helper = ConcreteHelper(cache=None)
        result = helper._get_cached("key", default="fallback")
        assert result == "fallback"

    def test_get_cached_with_cache(self):
        async def get_fn(key):
            return "cached"

        mock_cache = MagicMock()
        mock_cache.get = MagicMock(side_effect=lambda k: get_fn(k))
        helper = ConcreteHelper(cache=mock_cache)
        result = helper._get_cached("key")
        assert result == "cached"

    def test_get_cached_exception_returns_default(self):
        mock_cache = MagicMock()
        mock_cache.get = MagicMock(side_effect=Exception("cache error"))
        helper = ConcreteHelper(cache=mock_cache)
        result = helper._get_cached("key", default="default")
        assert result == "default"

    def test_set_cache_no_cache_does_nothing(self):
        helper = ConcreteHelper(cache=None)
        helper._set_cache("key", "value")  # no crash

    def test_set_cache_with_cache(self):
        async def set_fn(key, value):
            pass

        mock_cache = MagicMock()
        mock_cache.set = MagicMock(side_effect=lambda k, v: set_fn(k, v))
        helper = ConcreteHelper(cache=mock_cache)
        helper._set_cache("key", "value", fire_and_forget=False)
        mock_cache.set.assert_called_once_with("key", "value")

    def test_set_cache_with_ttl(self):
        async def set_fn(key, value, ttl=None):
            pass

        mock_cache = MagicMock()
        mock_cache.set = MagicMock(side_effect=lambda k, v, ttl=None: set_fn(k, v, ttl=ttl))
        helper = ConcreteHelper(cache=mock_cache)
        helper._set_cache("key", "value", ttl=3600, fire_and_forget=False)
        mock_cache.set.assert_called_once_with("key", "value", ttl=3600)

    def test_run_async(self):
        async def coro():
            return "result"

        helper = ConcreteHelper()
        result = helper._run_async(coro())
        assert result == "result"

    # ------------------------------------------------------------------
    # _log_event: loop.is_running() True/False branches (lines 123-126)
    # ------------------------------------------------------------------

    async def test_log_event_fire_and_forget_creates_task_in_running_loop(self):
        """In an async context, loop.is_running() is True → create_task."""
        import asyncio as _asyncio

        async def log_fn(event, data):
            pass

        mock_logger = MagicMock()
        mock_logger.log_event = MagicMock(side_effect=lambda e, d: log_fn(e, d))
        helper = ConcreteHelper(event_logger=mock_logger)

        created = []
        original_create_task = _asyncio.create_task

        def capture_create_task(coro, **kw):
            t = original_create_task(coro, **kw)
            created.append(t)
            return t

        with MagicMock() as _:
            import unittest.mock as _mock

            with _mock.patch("asyncio.create_task", side_effect=capture_create_task):
                helper._log_event("test.event", {"data": "value"})

        # create_task was called (loop is running in async context)
        assert len(created) == 1

    def test_log_event_fire_and_forget_loop_not_running(self):
        """loop.is_running() False → loop.run_until_complete."""

        async def log_fn(event, data):
            pass

        mock_logger = MagicMock()
        mock_logger.log_event = MagicMock(side_effect=lambda e, d: log_fn(e, d))
        helper = ConcreteHelper(event_logger=mock_logger)

        mock_loop = MagicMock()
        mock_loop.is_running.return_value = False
        mock_loop.run_until_complete.return_value = None

        with MagicMock() as _:
            import unittest.mock as _mock

            with _mock.patch("asyncio.get_event_loop", return_value=mock_loop):
                helper._log_event("test.event", {})

        mock_loop.run_until_complete.assert_called_once()

    # ------------------------------------------------------------------
    # _set_cache: fire_and_forget=True branches (lines 194-210)
    # ------------------------------------------------------------------

    async def test_set_cache_fire_and_forget_creates_task_in_running_loop(self):
        """In async context, loop.is_running() True → create_task."""
        import asyncio as _asyncio
        import unittest.mock as _mock

        async def set_fn(key, value):
            pass

        mock_cache = MagicMock()
        mock_cache.set = MagicMock(side_effect=lambda k, v: set_fn(k, v))
        helper = ConcreteHelper(cache=mock_cache)

        created = []
        original_create_task = _asyncio.create_task

        def capture(coro, **kw):
            t = original_create_task(coro, **kw)
            created.append(t)
            return t

        with _mock.patch("asyncio.create_task", side_effect=capture):
            helper._set_cache("key", "value", fire_and_forget=True)

        assert len(created) == 1

    def test_set_cache_fire_and_forget_loop_not_running(self):
        """loop.is_running() False → loop.run_until_complete."""
        import unittest.mock as _mock

        async def set_fn(key, value):
            pass

        mock_cache = MagicMock()
        mock_cache.set = MagicMock(side_effect=lambda k, v: set_fn(k, v))
        helper = ConcreteHelper(cache=mock_cache)

        mock_loop = MagicMock()
        mock_loop.is_running.return_value = False
        mock_loop.run_until_complete.return_value = None

        with _mock.patch("asyncio.get_event_loop", return_value=mock_loop):
            helper._set_cache("key", "value", fire_and_forget=True)

        mock_loop.run_until_complete.assert_called_once()

    def test_set_cache_fire_and_forget_runtime_error_fallback(self):
        """RuntimeError from get_event_loop → asyncio.run."""
        import unittest.mock as _mock

        async def set_fn(key, value):
            pass

        mock_cache = MagicMock()
        mock_cache.set = MagicMock(side_effect=lambda k, v: set_fn(k, v))
        helper = ConcreteHelper(cache=mock_cache)

        with (
            _mock.patch("asyncio.get_event_loop", side_effect=RuntimeError("no loop")),
            _mock.patch("asyncio.run", return_value=None) as mock_run,
        ):
            helper._set_cache("key", "value", fire_and_forget=True)

        mock_run.assert_called_once()

    def test_set_cache_fire_and_forget_asyncio_run_exception_silenced(self):
        """asyncio.run raises → exception silently caught (line 203-204)."""
        import unittest.mock as _mock

        async def set_fn(key, value):
            pass

        mock_cache = MagicMock()
        mock_cache.set = MagicMock(side_effect=lambda k, v: set_fn(k, v))
        helper = ConcreteHelper(cache=mock_cache)

        with (
            _mock.patch("asyncio.get_event_loop", side_effect=RuntimeError("no loop")),
            _mock.patch("asyncio.run", side_effect=RuntimeError("asyncio.run failed")),
        ):
            helper._set_cache("key", "value", fire_and_forget=True)  # Must not raise

    def test_set_cache_wait_exception_silenced(self):
        """_run_async raises in fire_and_forget=False path (lines 209-210)."""
        import unittest.mock as _mock

        async def set_fn(key, value):
            raise RuntimeError("set failed in run_async")

        mock_cache = MagicMock()
        mock_cache.set = MagicMock(side_effect=lambda k, v: set_fn(k, v))
        helper = ConcreteHelper(cache=mock_cache)

        with _mock.patch("asyncio.get_event_loop", side_effect=RuntimeError("no loop")):
            with _mock.patch("asyncio.run", side_effect=RuntimeError("run failed")):
                helper._set_cache("key", "value", fire_and_forget=False)  # Must not raise


# ---------------------------------------------------------------------------
# run_async_in_sync: RuntimeError fallback branch (line 49)
# ---------------------------------------------------------------------------


class TestRunAsyncInSyncRuntimeError:
    def test_runtime_error_falls_back_to_asyncio_run(self):
        """When get_event_loop raises RuntimeError, falls back to asyncio.run."""
        import unittest.mock as _mock

        async def coro():
            return "fallback_result"

        with (
            _mock.patch("asyncio.get_event_loop", side_effect=RuntimeError("no loop")),
            _mock.patch("asyncio.run", return_value="fallback_result") as mock_run,
        ):
            result = run_async_in_sync(coro())

        mock_run.assert_called_once()
        assert result == "fallback_result"
