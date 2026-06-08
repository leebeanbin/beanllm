"""Tests for infrastructure/distributed/pipeline_decorators.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from beanllm.infrastructure.distributed.pipeline_decorators import with_distributed_features

# ---------------------------------------------------------------------------
# Helpers — simple classes/functions for decorator tests
# ---------------------------------------------------------------------------


class _SyncService:
    def __init__(self):
        self.call_count = 0

    @with_distributed_features(pipeline_type="default")
    def compute(self, x: int, y: int = 0) -> int:
        self.call_count += 1
        return x + y


class _AsyncService:
    def __init__(self):
        self.call_count = 0

    @with_distributed_features(pipeline_type="default")
    async def compute_async(self, x: int, y: int = 0) -> int:
        self.call_count += 1
        return x + y


async def _async_exec_side_effect(func, self_obj, args, kwargs, *rest):
    """Async side effect that awaits coroutine functions."""
    result = func(self_obj, *args, **kwargs)
    import asyncio

    if asyncio.iscoroutine(result):
        return await result
    return result


def _mock_helpers():
    """Patch all helper/infrastructure calls so no real connections needed."""
    return [
        patch(
            "beanllm.infrastructure.distributed.pipeline_decorators._execute_with_features",
            side_effect=lambda func, self_obj, args, kwargs, *rest: func(self_obj, *args, **kwargs),
        ),
        patch(
            "beanllm.infrastructure.distributed.pipeline_decorators._execute_with_features_async",
            new=AsyncMock(side_effect=_async_exec_side_effect),
        ),
        patch("beanllm.infrastructure.distributed.pipeline_decorators.get_cache"),
        patch("beanllm.infrastructure.distributed.pipeline_decorators.get_lock_manager"),
        patch(
            "beanllm.infrastructure.distributed.pipeline_decorators._generate_cache_key",
            return_value="test-key",
        ),
        patch(
            "beanllm.infrastructure.distributed.pipeline_decorators._generate_lock_key",
            return_value="lock-key",
        ),
        patch(
            "beanllm.infrastructure.distributed.pipeline_decorators._publish_event", new=AsyncMock()
        ),
        patch("beanllm.infrastructure.distributed.pipeline_decorators._publish_event_async"),
    ]


# ---------------------------------------------------------------------------
# Sync wrapper
# ---------------------------------------------------------------------------


class TestWithDistributedFeaturesSyncWrapper:
    def test_decorated_sync_function_is_callable(self):
        svc = _SyncService()
        ctx = _mock_helpers()
        for c in ctx:
            c.start()
        try:
            result = svc.compute(3, y=4)
            assert result == 7
        finally:
            for c in reversed(ctx):
                c.stop()

    def test_sync_function_call_count_incremented(self):
        svc = _SyncService()
        ctx = _mock_helpers()
        for c in ctx:
            c.start()
        try:
            svc.compute(1)
            svc.compute(2)
            assert svc.call_count == 2
        finally:
            for c in reversed(ctx):
                c.stop()

    def test_all_overrides_applied(self):
        @with_distributed_features(
            pipeline_type="default",
            enable_cache=True,
            enable_rate_limiting=True,
            enable_event_streaming=True,
            enable_distributed_lock=True,
        )
        def fn(self):
            return "ok"

        assert callable(fn)

    def test_cache_key_prefix_used(self):
        @with_distributed_features(
            pipeline_type="default",
            enable_cache=True,
            cache_key_prefix="my:prefix",
        )
        def fn(self, x):
            return x

        class S:
            method = fn

        svc = S()
        ctx = _mock_helpers()
        for c in ctx:
            c.start()
        try:
            result = svc.method(99)
            assert result == 99
        finally:
            for c in reversed(ctx):
                c.stop()


# ---------------------------------------------------------------------------
# Async wrapper
# ---------------------------------------------------------------------------


class TestWithDistributedFeaturesAsyncWrapper:
    async def test_decorated_async_function_called(self):
        svc = _AsyncService()
        ctx = _mock_helpers()
        for c in ctx:
            c.start()
        try:
            result = await svc.compute_async(5, y=10)
            assert result == 15
        finally:
            for c in reversed(ctx):
                c.stop()

    async def test_async_call_count(self):
        svc = _AsyncService()
        ctx = _mock_helpers()
        for c in ctx:
            c.start()
        try:
            await svc.compute_async(1)
            await svc.compute_async(2)
            assert svc.call_count == 2
        finally:
            for c in reversed(ctx):
                c.stop()


# ---------------------------------------------------------------------------
# Decorator configuration
# ---------------------------------------------------------------------------


class TestDecoratorConfig:
    def test_unknown_pipeline_type_uses_defaults(self):
        @with_distributed_features(pipeline_type="completely_unknown_xyz")
        def fn(self):
            return "done"

        class S:
            method = fn

        svc = S()
        ctx = _mock_helpers()
        for c in ctx:
            c.start()
        try:
            result = svc.method()
            assert result == "done"
        finally:
            for c in reversed(ctx):
                c.stop()

    def test_known_pipeline_type_reads_config(self):
        # "ocr" has enable_cache, enable_rate_limiting etc.
        @with_distributed_features(pipeline_type="ocr")
        def fn(self, x=0):
            return x

        class S:
            method = fn

        svc = S()
        ctx = _mock_helpers()
        for c in ctx:
            c.start()
        try:
            result = svc.method(x=42)
            assert result == 42
        finally:
            for c in reversed(ctx):
                c.stop()

    def test_callable_rate_key_resolved(self):
        @with_distributed_features(
            pipeline_type="default",
            enable_rate_limiting=True,
            rate_limit_key=lambda self, args, kwargs: f"custom:{kwargs.get('model', 'gpt')}",
        )
        def fn(self, model="gpt"):
            return model

        class S:
            method = fn

        svc = S()
        ctx = _mock_helpers()
        for c in ctx:
            c.start()
        try:
            result = svc.method(model="claude")
            assert result == "claude"
        finally:
            for c in reversed(ctx):
                c.stop()

    def test_callable_rate_key_exception_falls_back(self):
        @with_distributed_features(
            pipeline_type="default",
            enable_rate_limiting=True,
            rate_limit_key=lambda self, args, kwargs: (_ for _ in ()).throw(RuntimeError("fail")),
        )
        def fn(self):
            return "result"

        class S:
            method = fn

        svc = S()
        ctx = _mock_helpers()
        for c in ctx:
            c.start()
        try:
            result = svc.method()
            assert result == "result"
        finally:
            for c in reversed(ctx):
                c.stop()

    async def test_async_callable_rate_key_resolved(self):
        @with_distributed_features(
            pipeline_type="default",
            enable_rate_limiting=True,
            rate_limit_key=lambda self, args, kwargs: "dynamic-key",
        )
        async def fn(self):
            return "async-result"

        class S:
            method = fn

        svc = S()
        ctx = _mock_helpers()
        for c in ctx:
            c.start()
        try:
            result = await svc.method()
            assert result == "async-result"
        finally:
            for c in reversed(ctx):
                c.stop()

    def test_preserves_function_name(self):
        @with_distributed_features(pipeline_type="default")
        def my_special_method(self):
            pass

        assert my_special_method.__name__ == "my_special_method"


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------


class TestCacheBehavior:
    def test_sync_cache_hit_returns_cached_result(self):
        @with_distributed_features(
            pipeline_type="default",
            enable_cache=True,
        )
        def fn(self, x):
            return x * 2

        class S:
            method = fn

        svc = S()
        mock_sync_cache = MagicMock()
        mock_sync_cache.get.return_value = "cached-value"

        with (
            patch(
                "beanllm.infrastructure.distributed.pipeline_decorators.SyncCacheWrapper",
                return_value=mock_sync_cache,
            ),
            patch(
                "beanllm.infrastructure.distributed.pipeline_decorators._generate_cache_key",
                return_value="key",
            ),
        ):
            result = svc.method(5)

        assert result == "cached-value"

    async def test_async_cache_hit_returns_cached_result(self):
        @with_distributed_features(
            pipeline_type="default",
            enable_cache=True,
        )
        async def fn(self, x):
            return x * 2

        class S:
            method = fn

        svc = S()
        mock_async_cache = AsyncMock()
        mock_async_cache.get = AsyncMock(return_value="async-cached")

        with (
            patch(
                "beanllm.infrastructure.distributed.pipeline_decorators.get_cache",
                return_value=mock_async_cache,
            ),
            patch(
                "beanllm.infrastructure.distributed.pipeline_decorators._generate_cache_key",
                return_value="akey",
            ),
        ):
            result = await svc.method(3)

        assert result == "async-cached"

    def test_sync_cache_miss_runs_function(self):
        call_count = [0]

        @with_distributed_features(
            pipeline_type="default",
            enable_cache=True,
        )
        def fn(self, x):
            call_count[0] += 1
            return x + 10

        class S:
            method = fn

        svc = S()
        mock_sync_cache = MagicMock()
        mock_sync_cache.get.return_value = None  # cache miss

        with (
            patch(
                "beanllm.infrastructure.distributed.pipeline_decorators.SyncCacheWrapper",
                return_value=mock_sync_cache,
            ),
            patch(
                "beanllm.infrastructure.distributed.pipeline_decorators._execute_with_features",
                side_effect=lambda func, self, args, kwargs, *rest: func(self, *args, **kwargs),
            ),
            patch(
                "beanllm.infrastructure.distributed.pipeline_decorators._generate_cache_key",
                return_value="k",
            ),
        ):
            result = svc.method(5)

        assert result == 15
        assert call_count[0] == 1
