"""Tests for infrastructure/distributed/redis/cache.py — RedisCache."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_cache(redis_client=None, key_prefix="test", ttl=None):
    """Create RedisCache with mocked Redis client."""
    mock_redis = redis_client or MagicMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock()
    mock_redis.setex = AsyncMock()
    mock_redis.delete = AsyncMock()
    mock_redis.scan = AsyncMock(return_value=(0, []))

    with patch(
        "beanllm.infrastructure.distributed.redis.cache.get_redis_client",
        return_value=mock_redis,
    ):
        from beanllm.infrastructure.distributed.redis.cache import RedisCache

        cache = RedisCache(redis_client=mock_redis, key_prefix=key_prefix, ttl=ttl)
    return cache, mock_redis


def _healthy_health_check():
    return patch(
        "beanllm.infrastructure.distributed.redis.cache.check_redis_health",
        return_value=True,
    )


def _unhealthy_health_check():
    return patch(
        "beanllm.infrastructure.distributed.redis.cache.check_redis_health",
        return_value=False,
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestRedisCache:
    async def test_init_with_custom_prefix(self):
        cache, redis = _make_cache(key_prefix="myapp")
        assert cache.key_prefix == "myapp"

    async def test_init_with_default_ttl(self):
        cache, redis = _make_cache(ttl=3600)
        assert cache.default_ttl == 3600

    async def test_make_key_string(self):
        cache, _ = _make_cache(key_prefix="cache")
        key = cache._make_key("my_key")
        assert key == "cache:my_key"

    async def test_make_key_dict(self):
        cache, _ = _make_cache(key_prefix="cache")
        key = cache._make_key({"a": 1, "b": 2})
        assert key.startswith("cache:")
        assert "a" in key

    async def test_make_key_list(self):
        cache, _ = _make_cache(key_prefix="cache")
        key = cache._make_key([1, 2, 3])
        assert key.startswith("cache:")


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------


class TestRedisCacheGet:
    async def test_get_returns_none_when_unhealthy(self):
        cache, redis = _make_cache()
        with _unhealthy_health_check():
            result = await cache.get("key")
        assert result is None
        redis.get.assert_not_awaited()

    async def test_get_returns_none_when_key_missing(self):
        cache, redis = _make_cache()
        redis.get = AsyncMock(return_value=None)
        with _healthy_health_check():
            result = await cache.get("missing_key")
        assert result is None

    async def test_get_returns_deserialized_value(self):
        cache, redis = _make_cache()
        redis.get = AsyncMock(return_value=json.dumps({"name": "Alice"}).encode("utf-8"))
        with _healthy_health_check():
            result = await cache.get("user_key")
        assert result == {"name": "Alice"}

    async def test_get_handles_string_value(self):
        cache, redis = _make_cache()
        redis.get = AsyncMock(return_value=json.dumps("hello"))
        with _healthy_health_check():
            result = await cache.get("str_key")
        assert result == "hello"

    async def test_get_returns_none_on_decode_error(self):
        cache, redis = _make_cache()
        redis.get = AsyncMock(return_value=b"not valid json {{{")
        with _healthy_health_check():
            result = await cache.get("bad_key")
        assert result is None

    async def test_get_returns_none_on_timeout(self):
        cache, redis = _make_cache()
        redis.get = AsyncMock(side_effect=asyncio.TimeoutError())
        with _healthy_health_check():
            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                result = await cache.get("timeout_key")
        assert result is None

    async def test_get_returns_none_on_exception(self):
        cache, redis = _make_cache()
        with _healthy_health_check():
            with patch("asyncio.wait_for", side_effect=RuntimeError("redis down")):
                result = await cache.get("error_key")
        assert result is None

    async def test_get_uses_correct_key(self):
        cache, redis = _make_cache(key_prefix="myns")
        redis.get = AsyncMock(return_value=None)
        with _healthy_health_check():
            await cache.get("some_key")
        call_args = redis.get.call_args.args[0]
        assert call_args == "myns:some_key"


# ---------------------------------------------------------------------------
# set()
# ---------------------------------------------------------------------------


class TestRedisCacheSet:
    async def test_set_skips_when_unhealthy(self):
        cache, redis = _make_cache()
        with _unhealthy_health_check():
            await cache.set("key", "value")
        redis.set.assert_not_awaited()
        redis.setex.assert_not_awaited()

    async def test_set_without_ttl(self):
        cache, redis = _make_cache(ttl=None)
        with _healthy_health_check():
            with patch("asyncio.wait_for", new_callable=AsyncMock):
                await cache.set("key", {"data": 42})

    async def test_set_with_explicit_ttl_uses_setex(self):
        cache, redis = _make_cache()
        captured = {}

        async def fake_wait_for(coro, timeout):
            captured["called"] = True
            return None

        with _healthy_health_check():
            with patch("asyncio.wait_for", side_effect=fake_wait_for):
                await cache.set("key", "value", ttl=300)
        # setex was called (via wait_for)
        assert captured.get("called")

    async def test_set_with_default_ttl(self):
        cache, redis = _make_cache(ttl=60)
        captured = {}

        async def fake_wait_for(coro, timeout):
            captured["called"] = True
            return None

        with _healthy_health_check():
            with patch("asyncio.wait_for", side_effect=fake_wait_for):
                await cache.set("key", "value")
        assert captured.get("called")

    async def test_set_swallows_timeout(self):
        cache, redis = _make_cache()
        with _healthy_health_check():
            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                await cache.set("key", "value")  # Should not raise

    async def test_set_swallows_exception(self):
        cache, redis = _make_cache()
        with _healthy_health_check():
            with patch("asyncio.wait_for", side_effect=RuntimeError("redis crash")):
                await cache.set("key", "value")  # Should not raise


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


class TestRedisCacheDelete:
    async def test_delete_skips_when_unhealthy(self):
        cache, redis = _make_cache()
        with _unhealthy_health_check():
            await cache.delete("key")
        redis.delete.assert_not_awaited()

    async def test_delete_calls_redis_delete(self):
        cache, redis = _make_cache(key_prefix="ns")
        captured = {}

        async def fake_wait_for(coro, timeout):
            captured["called"] = True
            return None

        with _healthy_health_check():
            with patch("asyncio.wait_for", side_effect=fake_wait_for):
                await cache.delete("my_key")
        assert captured.get("called")

    async def test_delete_swallows_timeout(self):
        cache, redis = _make_cache()
        with _healthy_health_check():
            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                await cache.delete("key")  # Should not raise

    async def test_delete_swallows_exception(self):
        cache, redis = _make_cache()
        with _healthy_health_check():
            with patch("asyncio.wait_for", side_effect=RuntimeError("crash")):
                await cache.delete("key")  # Should not raise


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------


class TestRedisCacheClear:
    async def test_clear_skips_when_unhealthy(self):
        cache, redis = _make_cache()
        with _unhealthy_health_check():
            await cache.clear()
        redis.scan.assert_not_awaited()

    async def test_clear_deletes_keys_with_prefix(self):
        cache, redis = _make_cache(key_prefix="ns")
        scan_calls = [0]

        async def fake_wait_for(coro, timeout):
            # First call: scan returns keys, cursor=0 (done)
            scan_calls[0] += 1
            if scan_calls[0] == 1:
                return (0, [b"ns:key1", b"ns:key2"])
            return (0, [])

        with _healthy_health_check():
            with patch("asyncio.wait_for", side_effect=fake_wait_for):
                await cache.clear()
        # Scan was invoked
        assert scan_calls[0] >= 1

    async def test_clear_multiple_scan_pages(self):
        cache, redis = _make_cache(key_prefix="ns")
        call_count = [0]

        async def fake_wait_for(coro, timeout):
            call_count[0] += 1
            if call_count[0] == 1:
                # First scan returns cursor != 0 → continue
                return (42, [b"ns:key1"])
            elif call_count[0] == 2:
                # Delete call
                return None
            elif call_count[0] == 3:
                # Second scan returns cursor=0 → stop
                return (0, [b"ns:key2"])
            return None

        with _healthy_health_check():
            with patch("asyncio.wait_for", side_effect=fake_wait_for):
                await cache.clear()
        assert call_count[0] >= 2

    async def test_clear_swallows_timeout(self):
        cache, redis = _make_cache()
        with _healthy_health_check():
            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                await cache.clear()  # Should not raise

    async def test_clear_swallows_exception(self):
        cache, redis = _make_cache()
        with _healthy_health_check():
            with patch("asyncio.wait_for", side_effect=RuntimeError("crash")):
                await cache.clear()  # Should not raise
