"""Tests for infrastructure/distributed/in_memory/*.py."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from beanllm.infrastructure.distributed.in_memory.cache import InMemoryCache
from beanllm.infrastructure.distributed.in_memory.events import InMemoryEventBus
from beanllm.infrastructure.distributed.in_memory.lock import InMemoryLock
from beanllm.infrastructure.distributed.in_memory.queue import InMemoryTaskQueue
from beanllm.infrastructure.distributed.in_memory.rate_limiter import InMemoryRateLimiter

# ---------------------------------------------------------------------------
# InMemoryCache (lines 71-87)
# ---------------------------------------------------------------------------


class TestInMemoryCache:
    async def test_set_and_get_value(self):
        cache = InMemoryCache()
        await cache.set("key1", "value1")
        result = await cache.get("key1")
        assert result == "value1"

    async def test_get_missing_key_returns_none(self):
        cache = InMemoryCache()
        result = await cache.get("nonexistent")
        assert result is None

    async def test_delete_removes_key(self):
        cache = InMemoryCache()
        await cache.set("key1", "value1")
        await cache.delete("key1")
        result = await cache.get("key1")
        assert result is None

    async def test_delete_nonexistent_key_does_not_raise(self):
        cache = InMemoryCache()
        await cache.delete("never_set")  # Should not raise

    async def test_clear_removes_all_keys(self):
        cache = InMemoryCache()
        await cache.set("a", 1)
        await cache.set("b", 2)
        await cache.clear()
        assert await cache.get("a") is None
        assert await cache.get("b") is None

    async def test_set_with_max_size(self):
        cache = InMemoryCache(max_size=2)
        await cache.set("k1", "v1")
        await cache.set("k2", "v2")
        await cache.set("k3", "v3")  # Evicts k1
        # k3 should be retrievable
        assert await cache.get("k3") == "v3"

    async def test_set_with_ttl_parameter(self):
        cache = InMemoryCache(ttl=60)
        await cache.set("key", "val", ttl=60)
        result = await cache.get("key")
        assert result == "val"

    async def test_get_returns_various_types(self):
        cache = InMemoryCache()
        await cache.set("int", 42)
        await cache.set("dict", {"a": 1})
        await cache.set("list", [1, 2, 3])
        assert await cache.get("int") == 42
        assert await cache.get("dict") == {"a": 1}
        assert await cache.get("list") == [1, 2, 3]


# ---------------------------------------------------------------------------
# InMemoryEventBus (lines 26-85)
# ---------------------------------------------------------------------------


class TestInMemoryEventBus:
    async def test_publish_without_subscribers(self):
        bus = InMemoryEventBus()
        # Should not raise
        await bus.publish("test.topic", {"key": "value"})

    async def test_publish_calls_sync_subscriber(self):
        bus = InMemoryEventBus()
        received = []

        def handler(event):
            received.append(event)

        bus._subscribers["my.topic"] = [handler]
        await bus.publish("my.topic", {"data": "hello"})
        assert received == [{"data": "hello"}]

    async def test_publish_calls_async_subscriber(self):
        bus = InMemoryEventBus()
        received = []

        async def handler(event):
            received.append(event)

        bus._subscribers["my.topic"] = [handler]
        await bus.publish("my.topic", {"data": "world"})
        assert received == [{"data": "world"}]

    async def test_publish_catches_handler_exception(self):
        bus = InMemoryEventBus()

        def bad_handler(event):
            raise RuntimeError("handler error")

        bus._subscribers["err.topic"] = [bad_handler]
        # Should not raise
        await bus.publish("err.topic", {"key": "val"})

    async def test_publish_stores_event_in_history(self):
        bus = InMemoryEventBus()
        await bus.publish("a.topic", {"x": 1})
        assert len(bus._events) == 1
        assert bus._events[0]["topic"] == "a.topic"
        assert bus._events[0]["event"] == {"x": 1}

    async def test_publish_prunes_history_over_1000(self):
        bus = InMemoryEventBus()
        # Pre-fill to 1000
        bus._events = [{"topic": "t", "event": {}, "timestamp": 0.0}] * 1000
        await bus.publish("new.topic", {"n": 1})
        # Should stay at 1000 (prune oldest)
        assert len(bus._events) == 1000
        assert bus._events[-1]["topic"] == "new.topic"

    async def test_subscribe_registers_handler(self):
        bus = InMemoryEventBus()
        received = []

        async def reader():
            async for event in bus.subscribe("msg.topic", lambda e: None):
                received.append(event)
                break  # Stop after first event

        # Publish after subscribe starts
        async def publisher():
            await asyncio.sleep(0.01)
            await bus.publish("msg.topic", {"text": "hello"})

        await asyncio.gather(reader(), publisher())
        assert len(received) == 1
        assert received[0] == {"text": "hello"}

    async def test_subscribe_cleans_up_handler_on_exit(self):
        bus = InMemoryEventBus()
        received = []

        async def reader():
            async for event in bus.subscribe("cleanup.topic", lambda e: None):
                received.append(event)
                break

        async def publisher():
            await asyncio.sleep(0.01)
            await bus.publish("cleanup.topic", {"msg": "done"})

        await asyncio.gather(reader(), publisher())

        # After the generator exits, _handler should be removed
        # (the outer handler added by subscribe() call is still there, but _handler should be removed)
        async_handlers = [
            h for h in bus._subscribers.get("cleanup.topic", []) if asyncio.iscoroutinefunction(h)
        ]
        # The queue-bound _handler should have been removed
        assert len(async_handlers) == 0


# ---------------------------------------------------------------------------
# InMemoryLock (lines 28-42)
# ---------------------------------------------------------------------------


class TestInMemoryLock:
    async def test_acquire_new_key_creates_lock(self):
        lock = InMemoryLock()
        async with lock.acquire("resource:1"):
            assert "resource:1" in lock._locks

    async def test_acquire_same_key_twice_sequentially(self):
        lock = InMemoryLock()
        async with lock.acquire("key"):
            pass
        async with lock.acquire("key"):
            pass  # Should work fine

    async def test_acquire_different_keys_simultaneously(self):
        lock = InMemoryLock()
        results = []

        async def task(key):
            async with lock.acquire(key):
                results.append(key)
                await asyncio.sleep(0.01)

        await asyncio.gather(task("a"), task("b"))
        assert sorted(results) == ["a", "b"]

    async def test_acquire_timeout_raises_timeout_error(self):
        lock = InMemoryLock()
        # Acquire and hold the lock to trigger timeout
        held_lock = asyncio.Lock()
        await held_lock.acquire()
        lock._locks["blocking_key"] = held_lock

        with pytest.raises(TimeoutError, match="Failed to acquire lock"):
            async with lock.acquire("blocking_key", timeout=0.01):
                pass

    async def test_acquire_reuses_existing_lock(self):
        lock = InMemoryLock()
        async with lock.acquire("shared"):
            pass
        first_lock = lock._locks["shared"]
        async with lock.acquire("shared"):
            pass
        assert lock._locks["shared"] is first_lock

    async def test_acquire_releases_on_exception(self):
        lock = InMemoryLock()
        with pytest.raises(ValueError):
            async with lock.acquire("err_key"):
                raise ValueError("inside lock")

        # Lock should be released — can be acquired again
        async with lock.acquire("err_key"):
            pass


# ---------------------------------------------------------------------------
# InMemoryTaskQueue (lines 27-82)
# ---------------------------------------------------------------------------


class TestInMemoryTaskQueue:
    async def test_enqueue_returns_task_id(self):
        queue = InMemoryTaskQueue()
        task_id = await queue.enqueue("test.task", {"key": "val"})
        assert isinstance(task_id, str)
        assert len(task_id) > 0

    async def test_dequeue_returns_enqueued_task(self):
        queue = InMemoryTaskQueue()
        await queue.enqueue("my.task", {"x": 42})
        task = await queue.dequeue("my.task")
        assert task is not None
        assert task["task_type"] == "my.task"
        assert task["data"] == {"x": 42}

    async def test_dequeue_empty_queue_with_timeout_returns_none(self):
        queue = InMemoryTaskQueue()
        task = await queue.dequeue("empty.task", timeout=0.05)
        assert task is None

    async def test_dequeue_without_timeout_gets_item(self):
        queue = InMemoryTaskQueue()
        await queue.enqueue("sync.task", {"v": 1})
        task = await queue.dequeue("sync.task", timeout=None)
        assert task is not None
        assert task["data"] == {"v": 1}

    async def test_enqueue_sets_status_to_pending(self):
        queue = InMemoryTaskQueue()
        task_id = await queue.enqueue("t", {})
        status = await queue.get_task_status(task_id)
        assert status is not None
        assert status["status"] == "pending"

    async def test_dequeue_updates_status_to_processing(self):
        queue = InMemoryTaskQueue()
        task_id = await queue.enqueue("t", {})
        task = await queue.dequeue("t")
        assert task is not None
        status = await queue.get_task_status(task_id)
        assert status is not None
        assert status["status"] == "processing"

    async def test_get_task_status_unknown_id_returns_none(self):
        queue = InMemoryTaskQueue()
        result = await queue.get_task_status("non-existent-id")
        assert result is None

    async def test_enqueue_with_priority_parameter(self):
        """Enqueue accepts priority parameter without error."""
        queue = InMemoryTaskQueue()
        tid1 = await queue.enqueue("prio.task", {"order": "low"}, priority=0)
        tid2 = await queue.enqueue("prio.task", {"order": "high"}, priority=10)
        # Both task IDs are non-empty strings
        assert isinstance(tid1, str) and len(tid1) > 0
        assert isinstance(tid2, str) and len(tid2) > 0

    async def test_get_queue_creates_new_queue_for_new_type(self):
        queue = InMemoryTaskQueue()
        q = queue._get_queue("new.type")
        assert isinstance(q, asyncio.Queue)

    async def test_get_queue_reuses_existing(self):
        queue = InMemoryTaskQueue()
        q1 = queue._get_queue("t")
        q2 = queue._get_queue("t")
        assert q1 is q2


# ---------------------------------------------------------------------------
# InMemoryRateLimiter (lines 82-99)
# ---------------------------------------------------------------------------


class TestInMemoryRateLimiter:
    async def test_acquire_returns_true_when_tokens_available(self):
        limiter = InMemoryRateLimiter(default_rate=100.0, default_capacity=100.0)
        result = await limiter.acquire("api:key")
        assert result is True

    async def test_acquire_creates_bucket_on_first_call(self):
        limiter = InMemoryRateLimiter()
        await limiter.acquire("new:key")
        assert "new:key" in limiter._buckets

    async def test_acquire_reuses_existing_bucket(self):
        limiter = InMemoryRateLimiter()
        await limiter.acquire("shared:key")
        first_bucket = limiter._buckets["shared:key"]
        await limiter.acquire("shared:key")
        assert limiter._buckets["shared:key"] is first_bucket

    async def test_wait_completes_when_tokens_available(self):
        limiter = InMemoryRateLimiter(default_rate=100.0, default_capacity=100.0)
        await limiter.wait("wait:key")  # Should complete quickly

    async def test_get_status_returns_key_and_bucket_info(self):
        limiter = InMemoryRateLimiter(default_rate=5.0, default_capacity=10.0)
        status = limiter.get_status("status:key")
        assert status["key"] == "status:key"
        assert "tokens" in status
        assert "rate" in status
        assert "capacity" in status

    async def test_get_status_different_keys_independent(self):
        limiter = InMemoryRateLimiter()
        s1 = limiter.get_status("key:1")
        s2 = limiter.get_status("key:2")
        assert s1["key"] == "key:1"
        assert s2["key"] == "key:2"

    async def test_acquire_returns_false_when_capacity_exhausted(self):
        # Very small capacity
        limiter = InMemoryRateLimiter(default_rate=0.001, default_capacity=1.0)
        # Use up the token
        await limiter.acquire("cap:key", cost=1.0)
        # Next acquire should fail (no tokens left, slow refill)
        result = await limiter.acquire("cap:key", cost=1.0)
        assert result is False
