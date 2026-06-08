"""Tests for infrastructure/distributed/utils.py."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from beanllm.infrastructure.distributed.utils import (
    check_kafka_health,
    check_redis_health,
    with_fallback,
)

# ---------------------------------------------------------------------------
# check_redis_health (lines 35-46)
# ---------------------------------------------------------------------------


class TestCheckRedisHealth:
    async def test_returns_true_when_ping_succeeds(self):
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)

        result = await check_redis_health(redis_client=mock_redis)
        assert result is True

    async def test_returns_false_when_ping_returns_false(self):
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=False)

        result = await check_redis_health(redis_client=mock_redis)
        assert result is False

    async def test_returns_false_when_ping_raises(self):
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(side_effect=ConnectionError("Redis down"))

        result = await check_redis_health(redis_client=mock_redis)
        assert result is False

    async def test_returns_false_when_timeout(self):
        async def slow_ping():
            await asyncio.sleep(10)
            return True

        mock_redis = MagicMock()
        mock_redis.ping = slow_ping

        # REDIS_TIMEOUT is small; patch wait_for to raise TimeoutError
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            result = await check_redis_health(redis_client=mock_redis)

        assert result is False

    async def test_auto_creates_client_when_none(self):
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)

        with patch(
            "beanllm.infrastructure.distributed.redis.client.get_redis_client",
            return_value=mock_redis,
        ):
            result = await check_redis_health(redis_client=None)

        assert result is True

    async def test_auto_create_client_import_error_returns_false(self):
        with patch(
            "beanllm.infrastructure.distributed.utils.check_redis_health",
            wraps=check_redis_health,
        ):
            # Simulate missing redis client module
            with patch(
                "beanllm.infrastructure.distributed.redis.client.get_redis_client",
                side_effect=ImportError("redis not installed"),
            ):
                result = await check_redis_health(redis_client=None)

        assert result is False


# ---------------------------------------------------------------------------
# check_kafka_health (lines 59-72)
# ---------------------------------------------------------------------------


class TestCheckKafkaHealth:
    async def test_returns_true_when_producer_and_consumer_not_none(self):
        mock_producer = MagicMock()
        mock_consumer = MagicMock()

        result = await check_kafka_health(kafka_client=(mock_producer, mock_consumer))
        assert result is True

    async def test_returns_false_when_producer_is_none(self):
        mock_consumer = MagicMock()

        result = await check_kafka_health(kafka_client=(None, mock_consumer))
        assert result is False

    async def test_returns_false_when_consumer_is_none(self):
        mock_producer = MagicMock()

        result = await check_kafka_health(kafka_client=(mock_producer, None))
        assert result is False

    async def test_returns_false_when_both_none(self):
        result = await check_kafka_health(kafka_client=(None, None))
        assert result is False

    async def test_returns_false_when_exception_raised(self):
        # Pass a non-iterable to trigger TypeError when unpacking
        result = await check_kafka_health(kafka_client="not-a-tuple")
        assert result is False

    async def test_unpacking_exception_returns_false(self):
        # Passing something that raises when unpacked
        result = await check_kafka_health(kafka_client=object())
        assert result is False


# ---------------------------------------------------------------------------
# with_fallback decorator (lines 93-123)
# ---------------------------------------------------------------------------


class TestWithFallback:
    async def test_success_returns_original_result(self):
        @with_fallback()
        async def my_func():
            return "original_result"

        result = await my_func()
        assert result == "original_result"

    async def test_failure_calls_fallback_func(self):
        async def fallback():
            return "fallback_result"

        @with_fallback(fallback_func=fallback)
        async def my_func():
            raise RuntimeError("primary failed")

        result = await my_func()
        assert result == "fallback_result"

    async def test_failure_passes_args_to_fallback_func(self):
        received_args = []

        async def fallback(*args, **kwargs):
            received_args.extend(args)
            return "fallback"

        @with_fallback(fallback_func=fallback)
        async def my_func(x, y):
            raise RuntimeError("failed")

        await my_func(1, 2)
        assert 1 in received_args
        assert 2 in received_args

    async def test_failure_without_fallback_reraises_for_unknown_func(self):
        @with_fallback()
        async def unknown_component():
            raise RuntimeError("unknown component failed")

        with pytest.raises(RuntimeError, match="unknown component failed"):
            await unknown_component()

    async def test_log_error_false_suppresses_warning(self):
        @with_fallback(log_error=False)
        async def my_func():
            raise RuntimeError("no log")

        with pytest.raises(RuntimeError):
            await my_func()

    async def test_failure_no_fallback_redis_in_name_returns_in_memory_cache(self):
        from beanllm.infrastructure.distributed.in_memory.cache import InMemoryCache

        @with_fallback()
        async def get_RedisCache():
            raise RuntimeError("redis not available")

        result = await get_RedisCache()
        assert isinstance(result, InMemoryCache)

    async def test_failure_no_fallback_kafka_in_name_returns_in_memory_queue(self):
        from beanllm.infrastructure.distributed.in_memory.queue import InMemoryTaskQueue

        @with_fallback()
        async def get_KafkaQueue():
            raise RuntimeError("kafka not available")

        result = await get_KafkaQueue()
        assert isinstance(result, InMemoryTaskQueue)

    async def test_wraps_preserves_function_name(self):
        @with_fallback()
        async def my_named_func():
            return "ok"

        assert my_named_func.__name__ == "my_named_func"

    async def test_success_with_args(self):
        @with_fallback()
        async def add(a, b):
            return a + b

        result = await add(3, 4)
        assert result == 7

    async def test_fallback_func_exception_propagates(self):
        async def bad_fallback():
            raise ValueError("fallback also failed")

        @with_fallback(fallback_func=bad_fallback)
        async def my_func():
            raise RuntimeError("primary failed")

        with pytest.raises(ValueError, match="fallback also failed"):
            await my_func()
