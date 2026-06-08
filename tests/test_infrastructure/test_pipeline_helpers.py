"""Tests for infrastructure/distributed/pipeline_helpers.py."""

import asyncio
import hashlib
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from beanllm.infrastructure.distributed.pipeline_helpers import (
    _file_hash,
    execute_with_features,
    execute_with_features_async,
    generate_cache_key,
    generate_lock_key,
    publish_event,
    publish_event_async,
)

# ---------------------------------------------------------------------------
# _file_hash
# ---------------------------------------------------------------------------


class TestFileHash:
    def test_returns_hex_digest(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"hello world")
            path = f.name
        try:
            result = _file_hash(path)
            assert len(result) == 64  # SHA256 hex digest
            assert all(c in "0123456789abcdef" for c in result)
        finally:
            os.unlink(path)

    def test_same_content_same_hash(self):
        content = b"test content"
        with tempfile.NamedTemporaryFile(delete=False) as f1:
            f1.write(content)
            path1 = f1.name
        with tempfile.NamedTemporaryFile(delete=False) as f2:
            f2.write(content)
            path2 = f2.name
        try:
            assert _file_hash(path1) == _file_hash(path2)
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_different_content_different_hash(self):
        with tempfile.NamedTemporaryFile(delete=False) as f1:
            f1.write(b"content a")
            path1 = f1.name
        with tempfile.NamedTemporaryFile(delete=False) as f2:
            f2.write(b"content b")
            path2 = f2.name
        try:
            assert _file_hash(path1) != _file_hash(path2)
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_matches_standard_sha256(self):
        content = b"hello"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(content)
            path = f.name
        try:
            expected = hashlib.sha256(content).hexdigest()
            assert _file_hash(path) == expected
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# generate_cache_key
# ---------------------------------------------------------------------------


class TestGenerateCacheKey:
    def test_returns_prefixed_string(self):
        key = generate_cache_key("myprefix", (), {})
        assert key.startswith("myprefix:")

    def test_same_args_same_key(self):
        key1 = generate_cache_key("prefix", ("arg1",), {"k": "v"})
        key2 = generate_cache_key("prefix", ("arg1",), {"k": "v"})
        assert key1 == key2

    def test_different_args_different_key(self):
        key1 = generate_cache_key("prefix", ("arg1",), {})
        key2 = generate_cache_key("prefix", ("arg2",), {})
        assert key1 != key2

    def test_non_path_string_arg(self):
        key = generate_cache_key("prefix", ("not_a_file_path_xyz",), {})
        assert key.startswith("prefix:")

    def test_file_path_arg_uses_hash(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"content")
            path = f.name
        try:
            key = generate_cache_key("prefix", (path,), {})
            assert "file:" in key or key.startswith("prefix:")
        finally:
            os.unlink(path)

    def test_path_arg_that_is_not_file(self):
        key = generate_cache_key("prefix", ("/definitely/does/not/exist",), {})
        assert key.startswith("prefix:")

    def test_numeric_arg(self):
        key = generate_cache_key("prefix", (42,), {})
        assert key.startswith("prefix:")

    def test_path_object_arg(self):
        key = generate_cache_key("prefix", (Path("/tmp/nonexistent"),), {})
        assert key.startswith("prefix:")

    def test_array_arg_with_tobytes(self):
        mock_arr = MagicMock()
        mock_arr.tobytes.return_value = b"\x01\x02\x03"
        key = generate_cache_key("prefix", (mock_arr,), {})
        assert "array:" in key or key.startswith("prefix:")

    def test_key_hash_is_16_chars(self):
        key = generate_cache_key("p", (), {})
        hash_part = key.split(":", 1)[1]
        assert len(hash_part) == 16

    def test_kwargs_affect_key(self):
        key1 = generate_cache_key("prefix", (), {"a": 1})
        key2 = generate_cache_key("prefix", (), {"a": 2})
        assert key1 != key2


# ---------------------------------------------------------------------------
# generate_lock_key
# ---------------------------------------------------------------------------


class TestGenerateLockKey:
    def test_returns_none_without_args(self):
        result = generate_lock_key("prefix", (), {})
        assert result is None

    def test_returns_none_for_non_path_arg(self):
        result = generate_lock_key("prefix", (42,), {})
        assert result is None

    def test_returns_lock_key_for_string_path(self):
        result = generate_lock_key("prefix", ("/some/file/path",), {})
        assert result is not None
        assert "prefix" in result
        assert "lock" in result

    def test_returns_lock_key_for_path_object(self):
        result = generate_lock_key("prefix", (Path("/some/path"),), {})
        assert result is not None
        assert "prefix" in result

    def test_same_path_same_lock_key(self):
        key1 = generate_lock_key("prefix", ("/same/path",), {})
        key2 = generate_lock_key("prefix", ("/same/path",), {})
        assert key1 == key2

    def test_different_path_different_lock_key(self):
        key1 = generate_lock_key("prefix", ("/path/a",), {})
        key2 = generate_lock_key("prefix", ("/path/b",), {})
        assert key1 != key2


# ---------------------------------------------------------------------------
# publish_event (async)
# ---------------------------------------------------------------------------


class TestPublishEvent:
    async def test_calls_event_logger(self):
        mock_logger = MagicMock()
        mock_logger.log_event = AsyncMock()

        with patch(
            "beanllm.infrastructure.distributed.pipeline_helpers.get_event_logger",
            return_value=mock_logger,
        ):
            await publish_event("test.event", {"key": "value"})

        mock_logger.log_event.assert_awaited_once_with("test.event", {"key": "value"})

    async def test_handles_logger_exception_silently(self):
        mock_logger = MagicMock()
        mock_logger.log_event = AsyncMock(side_effect=RuntimeError("logger error"))

        with patch(
            "beanllm.infrastructure.distributed.pipeline_helpers.get_event_logger",
            return_value=mock_logger,
        ):
            await publish_event("test.event", {})  # should not raise


# ---------------------------------------------------------------------------
# publish_event_async (sync wrapper)
# ---------------------------------------------------------------------------


class TestPublishEventAsync:
    def test_runs_without_error_in_sync_context(self):
        mock_logger = MagicMock()
        mock_logger.log_event = AsyncMock()

        with patch(
            "beanllm.infrastructure.distributed.pipeline_helpers.get_event_logger",
            return_value=mock_logger,
        ):
            publish_event_async("test.event", {"k": "v"})  # should not raise

    def test_handles_exception_silently(self):
        with patch(
            "beanllm.infrastructure.distributed.pipeline_helpers.get_event_logger",
            side_effect=RuntimeError("no logger"),
        ):
            publish_event_async("test.event", {})  # should not raise


# ---------------------------------------------------------------------------
# execute_with_features (sync)
# ---------------------------------------------------------------------------


def _make_config(enable_cache=False, enable_rate_limiting=False, enable_events=False):
    cfg = MagicMock()
    cfg.enable_cache = enable_cache
    cfg.enable_rate_limiting = enable_rate_limiting
    cfg.enable_event_streaming = enable_events
    return cfg


class TestExecuteWithFeatures:
    def test_calls_func_and_returns_result(self):
        func = MagicMock(return_value="result_value")
        config = _make_config()
        result = execute_with_features(func, "self", (), {}, config, None, "key", "prefix")
        assert result == "result_value"
        func.assert_called_once_with("self")

    def test_passes_args_and_kwargs_to_func(self):
        func = MagicMock(return_value="ok")
        config = _make_config()
        execute_with_features(func, "obj", ("a", "b"), {"x": 1}, config, None, "key", "prefix")
        func.assert_called_once_with("obj", "a", "b", x=1)

    def test_publishes_events_when_enabled(self):
        func = MagicMock(return_value="ok")
        config = _make_config(enable_events=True)

        with patch(
            "beanllm.infrastructure.distributed.pipeline_helpers.publish_event_async"
        ) as mock_pub:
            execute_with_features(func, "obj", (), {}, config, None, "key", "prefix")

        assert mock_pub.call_count >= 2  # started + completed

    def test_publishes_error_event_on_exception(self):
        func = MagicMock(side_effect=ValueError("oh no"))
        config = _make_config(enable_events=True)

        with patch(
            "beanllm.infrastructure.distributed.pipeline_helpers.publish_event_async"
        ) as mock_pub:
            with pytest.raises(ValueError):
                execute_with_features(func, "obj", (), {}, config, None, "key", "prefix")

        event_types = [call.args[0] for call in mock_pub.call_args_list]
        assert any("failed" in ev for ev in event_types)

    def test_rate_limiting_with_string_key(self):
        func = MagicMock(return_value="ok")
        config = _make_config(enable_rate_limiting=True)

        mock_rate_limiter = MagicMock()
        mock_rate_limiter.acquire = AsyncMock()

        with patch(
            "beanllm.infrastructure.distributed.pipeline_helpers.get_rate_limiter",
            return_value=mock_rate_limiter,
        ):
            result = execute_with_features(func, "obj", (), {}, config, None, "mykey", "prefix")
        assert result == "ok"

    def test_callable_rate_key_is_invoked(self):
        func = MagicMock(return_value="ok")
        config = _make_config(enable_rate_limiting=True)
        rate_key_fn = MagicMock(return_value="computed_key")

        mock_rate_limiter = MagicMock()
        mock_rate_limiter.acquire = AsyncMock()

        with patch(
            "beanllm.infrastructure.distributed.pipeline_helpers.get_rate_limiter",
            return_value=mock_rate_limiter,
        ):
            execute_with_features(func, "obj", (), {}, config, None, rate_key_fn, "prefix")
        rate_key_fn.assert_called_once()

    def test_callable_rate_key_exception_uses_default(self):
        func = MagicMock(return_value="ok")
        config = _make_config(enable_rate_limiting=True)
        rate_key_fn = MagicMock(side_effect=RuntimeError("key error"))

        with patch(
            "beanllm.infrastructure.distributed.pipeline_helpers.get_rate_limiter",
        ) as mock_rl:
            mock_rl.return_value.acquire = AsyncMock()
            execute_with_features(func, "obj", (), {}, config, None, rate_key_fn, "prefix")
        # Should not raise despite key function error


# ---------------------------------------------------------------------------
# execute_with_features_async
# ---------------------------------------------------------------------------


class TestExecuteWithFeaturesAsync:
    async def test_calls_async_func_and_returns_result(self):
        async def func(self, x):
            return f"async:{x}"

        config = _make_config()
        result = await execute_with_features_async(
            func, "self", ("hello",), {}, config, None, "key", "prefix"
        )
        assert result == "async:hello"

    async def test_publishes_events_when_enabled(self):
        async def func(self):
            return "ok"

        config = _make_config(enable_events=True)
        mock_logger = MagicMock()
        mock_logger.log_event = AsyncMock()

        with patch(
            "beanllm.infrastructure.distributed.pipeline_helpers.get_event_logger",
            return_value=mock_logger,
        ):
            await execute_with_features_async(func, "obj", (), {}, config, None, "key", "prefix")

        assert mock_logger.log_event.await_count >= 2

    async def test_publishes_error_event_on_exception(self):
        async def func(self):
            raise ValueError("async fail")

        config = _make_config(enable_events=True)
        mock_logger = MagicMock()
        mock_logger.log_event = AsyncMock()

        with patch(
            "beanllm.infrastructure.distributed.pipeline_helpers.get_event_logger",
            return_value=mock_logger,
        ):
            with pytest.raises(ValueError):
                await execute_with_features_async(
                    func, "obj", (), {}, config, None, "key", "prefix"
                )

        event_names = [call.args[0] for call in mock_logger.log_event.call_args_list]
        assert any("failed" in ev for ev in event_names)

    async def test_rate_limiting_with_string_key(self):
        async def func(self):
            return "result"

        config = _make_config(enable_rate_limiting=True)
        mock_rate_limiter = MagicMock()
        mock_rate_limiter.acquire = AsyncMock()

        with patch(
            "beanllm.infrastructure.distributed.pipeline_helpers.get_rate_limiter",
            return_value=mock_rate_limiter,
        ):
            result = await execute_with_features_async(
                func, "obj", (), {}, config, None, "mykey", "prefix"
            )
        assert result == "result"
        mock_rate_limiter.acquire.assert_awaited_once_with(key="mykey", cost=1.0)

    async def test_callable_rate_key_resolved(self):
        async def func(self):
            return "ok"

        config = _make_config(enable_rate_limiting=True)
        rate_key_fn = MagicMock(return_value="computed")
        mock_rate_limiter = MagicMock()
        mock_rate_limiter.acquire = AsyncMock()

        with patch(
            "beanllm.infrastructure.distributed.pipeline_helpers.get_rate_limiter",
            return_value=mock_rate_limiter,
        ):
            await execute_with_features_async(
                func, "obj", (), {}, config, None, rate_key_fn, "prefix"
            )
        rate_key_fn.assert_called_once()

    async def test_callable_rate_key_returns_none_skips_limiting(self):
        async def func(self):
            return "ok"

        config = _make_config(enable_rate_limiting=True)
        rate_key_fn = MagicMock(return_value=None)
        mock_rate_limiter = MagicMock()
        mock_rate_limiter.acquire = AsyncMock()

        with patch(
            "beanllm.infrastructure.distributed.pipeline_helpers.get_rate_limiter",
            return_value=mock_rate_limiter,
        ):
            await execute_with_features_async(
                func, "obj", (), {}, config, None, rate_key_fn, "prefix"
            )
        mock_rate_limiter.acquire.assert_not_awaited()

    async def test_rate_limit_condition_false_skips_limiting(self):
        async def func(self):
            return "ok"

        config = _make_config(enable_rate_limiting=True)
        condition_fn = MagicMock(return_value=False)
        mock_rate_limiter = MagicMock()
        mock_rate_limiter.acquire = AsyncMock()

        with patch(
            "beanllm.infrastructure.distributed.pipeline_helpers.get_rate_limiter",
            return_value=mock_rate_limiter,
        ):
            await execute_with_features_async(
                func,
                "obj",
                (),
                {},
                config,
                None,
                "key",
                "prefix",
                rate_limit_condition=condition_fn,
            )
        mock_rate_limiter.acquire.assert_not_awaited()

    async def test_rate_limit_condition_exception_defaults_to_limit(self):
        async def func(self):
            return "ok"

        config = _make_config(enable_rate_limiting=True)
        condition_fn = MagicMock(side_effect=RuntimeError("cond error"))
        mock_rate_limiter = MagicMock()
        mock_rate_limiter.acquire = AsyncMock()

        with patch(
            "beanllm.infrastructure.distributed.pipeline_helpers.get_rate_limiter",
            return_value=mock_rate_limiter,
        ):
            result = await execute_with_features_async(
                func,
                "obj",
                (),
                {},
                config,
                None,
                "key",
                "prefix",
                rate_limit_condition=condition_fn,
            )
        assert result == "ok"

    async def test_rate_limit_exception_logs_and_continues(self):
        async def func(self):
            return "still ok"

        config = _make_config(enable_rate_limiting=True)
        mock_rate_limiter = MagicMock()
        mock_rate_limiter.acquire = AsyncMock(side_effect=RuntimeError("rate limit failed"))

        with patch(
            "beanllm.infrastructure.distributed.pipeline_helpers.get_rate_limiter",
            return_value=mock_rate_limiter,
        ):
            result = await execute_with_features_async(
                func, "obj", (), {}, config, None, "mykey", "prefix"
            )
        assert result == "still ok"
