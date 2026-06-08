"""
Tests for infrastructure/distributed/messaging.py

Mocks Redis, Kafka, and the factory module to test MessageProducer,
ConcurrencyController, DistributedErrorHandler, and RequestMonitor.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helper: build mocked factory components
# ---------------------------------------------------------------------------


def make_mock_event_bus():
    """Return (producer, consumer) mocks."""
    producer = MagicMock()
    producer.publish = AsyncMock()
    consumer = MagicMock()
    consumer.subscribe = MagicMock()
    return producer, consumer


def make_mock_rate_limiter():
    rl = MagicMock()
    rl.wait = AsyncMock()
    return rl


def make_mock_lock():
    lock = MagicMock()
    return lock


def make_mock_redis():
    r = MagicMock()
    r.setex = AsyncMock()
    r.lpush = AsyncMock()
    r.get = AsyncMock(return_value=None)
    r.incr = AsyncMock(return_value=1)
    r.decr = AsyncMock(return_value=0)
    r.expire = AsyncMock()
    return r


# ---------------------------------------------------------------------------
# MessageProducer tests
# ---------------------------------------------------------------------------


@pytest.fixture
def message_producer():
    """MessageProducer with fully mocked dependencies."""
    producer_mock, consumer_mock = make_mock_event_bus()
    redis_mock = make_mock_redis()

    with (
        patch(
            "beanllm.infrastructure.distributed.messaging.get_event_bus",
            return_value=(producer_mock, consumer_mock),
        ),
        patch(
            "beanllm.infrastructure.distributed.messaging.get_rate_limiter",
            return_value=make_mock_rate_limiter(),
        ),
        patch(
            "beanllm.infrastructure.distributed.messaging.get_distributed_lock",
            return_value=make_mock_lock(),
        ),
    ):
        from beanllm.infrastructure.distributed.messaging import MessageProducer

        mp = MessageProducer.__new__(MessageProducer)
        mp.producer = producer_mock
        mp.redis = redis_mock
        yield mp, producer_mock, redis_mock


class TestMessageProducerPublishRequest:
    async def test_publish_request_returns_uuid_string(self, message_producer):
        mp, producer, _ = message_producer
        request_id = await mp.publish_request("ocr.recognize", {"file": "image.png"})
        assert isinstance(request_id, str)
        assert len(request_id) == 36  # UUID format

    async def test_publish_request_calls_producer_publish(self, message_producer):
        mp, producer, _ = message_producer
        await mp.publish_request("audio.transcribe", {"audio": "audio.wav"})
        producer.publish.assert_called_once()
        call_args = producer.publish.call_args
        assert call_args[0][0] == "llm.requests"

    async def test_publish_request_message_structure(self, message_producer):
        mp, producer, _ = message_producer
        await mp.publish_request("rag.query", {"query": "What is AI?"})
        message = producer.publish.call_args[0][1]
        assert "request_id" in message
        assert "request_type" in message
        assert message["request_type"] == "rag.query"
        assert "timestamp" in message
        assert "data" in message
        assert message["status"] == "pending"
        assert "metadata" in message

    async def test_publish_request_saves_status_to_redis(self, message_producer):
        mp, _, redis = message_producer
        await mp.publish_request("ocr.recognize", {"file": "img.png"})
        redis.setex.assert_called()

    async def test_publish_request_pushes_to_redis_queue(self, message_producer):
        mp, _, redis = message_producer
        await mp.publish_request("ocr.recognize", {"file": "img.png"})
        redis.lpush.assert_called()

    async def test_publish_request_without_redis(self, message_producer):
        mp, producer, _ = message_producer
        mp.redis = None
        request_id = await mp.publish_request("test.op", {"data": "val"})
        # Should still return a UUID even without Redis
        assert isinstance(request_id, str)
        producer.publish.assert_called_once()

    async def test_publish_request_handles_producer_failure(self, message_producer):
        mp, producer, _ = message_producer
        producer.publish.side_effect = Exception("Kafka down")
        # Should not raise (error is logged internally)
        request_id = await mp.publish_request("rag.query", {"q": "test"})
        assert isinstance(request_id, str)

    async def test_publish_request_handles_redis_setex_failure(self, message_producer):
        mp, producer, redis = message_producer
        redis.setex.side_effect = Exception("Redis down")
        # Should not raise
        request_id = await mp.publish_request("ocr.recognize", {"f": "x"})
        assert isinstance(request_id, str)

    async def test_publish_request_handles_redis_lpush_failure(self, message_producer):
        mp, producer, redis = message_producer
        redis.lpush.side_effect = Exception("Redis queue error")
        # Should not raise
        request_id = await mp.publish_request("rag.query", {"q": "x"})
        assert isinstance(request_id, str)


class TestMessageProducerPublishEvent:
    async def test_publish_event_calls_producer(self, message_producer):
        mp, producer, _ = message_producer
        await mp.publish_event("ocr.request.started", {"request_id": "123"})
        producer.publish.assert_called_once()
        call_args = producer.publish.call_args
        assert call_args[0][0] == "llm.events"

    async def test_publish_event_message_structure(self, message_producer):
        mp, producer, _ = message_producer
        await mp.publish_event("document.processed", {"doc_id": "abc"})
        event = producer.publish.call_args[0][1]
        assert "event_id" in event
        assert "event_type" in event
        assert event["event_type"] == "document.processed"
        assert "timestamp" in event
        assert "data" in event

    async def test_publish_event_handles_producer_failure(self, message_producer):
        mp, producer, _ = message_producer
        producer.publish.side_effect = Exception("Kafka error")
        # Should not raise
        await mp.publish_event("some.event", {"key": "val"})


# ---------------------------------------------------------------------------
# ConcurrencyController tests
# ---------------------------------------------------------------------------


@pytest.fixture
def concurrency_controller():
    """ConcurrencyController with mocked dependencies."""
    rate_limiter = make_mock_rate_limiter()
    lock = make_mock_lock()
    redis = make_mock_redis()

    with (
        patch(
            "beanllm.infrastructure.distributed.messaging.get_rate_limiter",
            return_value=rate_limiter,
        ),
        patch(
            "beanllm.infrastructure.distributed.messaging.get_distributed_lock",
            return_value=lock,
        ),
        patch(
            "beanllm.infrastructure.distributed.messaging.get_event_bus",
            return_value=make_mock_event_bus(),
        ),
    ):
        from beanllm.infrastructure.distributed.messaging import ConcurrencyController

        cc = ConcurrencyController.__new__(ConcurrencyController)
        cc.rate_limiter = rate_limiter
        cc.lock = lock
        cc.redis = redis
        yield cc, rate_limiter, redis


class TestConcurrencyControllerAcquireSlot:
    async def test_acquire_slot_returns_true_without_redis(self, concurrency_controller):
        cc, _, _ = concurrency_controller
        cc.redis = None
        result = await cc.acquire_slot("ocr", max_concurrent=5)
        assert result is True

    async def test_acquire_slot_returns_false_at_max(self, concurrency_controller):
        cc, _, redis = concurrency_controller
        redis.get.return_value = b"10"
        redis.incr.return_value = 11
        result = await cc.acquire_slot("ocr", max_concurrent=10)
        assert result is False

    async def test_acquire_slot_returns_true_under_limit(self, concurrency_controller):
        cc, _, redis = concurrency_controller
        redis.get.return_value = b"3"
        redis.incr.return_value = 4
        result = await cc.acquire_slot("ocr", max_concurrent=10)
        assert result is True
        redis.incr.assert_called_once()
        redis.expire.assert_called_once()

    async def test_acquire_slot_returns_true_on_redis_error(self, concurrency_controller):
        cc, _, redis = concurrency_controller
        redis.get.side_effect = Exception("Redis error")
        # Fallback: returns True on error
        result = await cc.acquire_slot("ocr", max_concurrent=5)
        assert result is True

    async def test_acquire_slot_returns_true_when_no_current(self, concurrency_controller):
        cc, _, redis = concurrency_controller
        redis.get.return_value = None  # No existing count
        redis.incr.return_value = 1
        result = await cc.acquire_slot("ocr", max_concurrent=5)
        assert result is True


class TestConcurrencyControllerReleaseSlot:
    async def test_release_slot_decrements_redis(self, concurrency_controller):
        cc, _, redis = concurrency_controller
        await cc.release_slot("ocr")
        redis.decr.assert_called_once()

    async def test_release_slot_without_redis_does_not_raise(self, concurrency_controller):
        cc, _, _ = concurrency_controller
        cc.redis = None
        await cc.release_slot("ocr")  # Should not raise

    async def test_release_slot_handles_redis_error(self, concurrency_controller):
        cc, _, redis = concurrency_controller
        redis.decr.side_effect = Exception("Redis error")
        await cc.release_slot("ocr")  # Should not raise


class TestConcurrencyControllerWithConcurrencyControl:
    async def test_context_manager_acquires_and_releases(self, concurrency_controller):
        cc, _, redis = concurrency_controller
        redis.get.return_value = b"0"
        redis.incr.return_value = 1

        ctx = await cc.with_concurrency_control("ocr", max_concurrent=5)
        async with ctx:
            pass

        redis.incr.assert_called_once()
        redis.decr.assert_called_once()

    async def test_context_manager_raises_on_max_concurrent(self, concurrency_controller):
        cc, _, redis = concurrency_controller
        redis.get.return_value = b"5"
        redis.incr.return_value = 6

        from beanllm.infrastructure.distributed.utils import DistributedError

        ctx = await cc.with_concurrency_control("ocr", max_concurrent=5)
        with pytest.raises(DistributedError, match="Max concurrent"):
            async with ctx:
                pass

    async def test_context_manager_with_rate_limit_key(self, concurrency_controller):
        cc, rate_limiter, redis = concurrency_controller
        redis.get.return_value = b"0"
        redis.incr.return_value = 1

        ctx = await cc.with_concurrency_control("ocr", max_concurrent=5, rate_limit_key="ocr:key")
        async with ctx:
            pass

        rate_limiter.wait.assert_called_once_with("ocr:key")

    async def test_context_manager_releases_on_exception(self, concurrency_controller):
        cc, _, redis = concurrency_controller
        redis.get.return_value = b"0"
        redis.incr.return_value = 1

        ctx = await cc.with_concurrency_control("ocr", max_concurrent=5)
        with pytest.raises(ValueError):
            async with ctx:
                raise ValueError("test error")

        # release should still be called
        redis.decr.assert_called_once()


# ---------------------------------------------------------------------------
# DistributedErrorHandler tests
# ---------------------------------------------------------------------------


@pytest.fixture
def error_handler():
    """DistributedErrorHandler with mocked dependencies."""
    producer_mock, consumer_mock = make_mock_event_bus()
    redis_mock = make_mock_redis()

    with (
        patch(
            "beanllm.infrastructure.distributed.messaging.get_event_bus",
            return_value=(producer_mock, consumer_mock),
        ),
        patch(
            "beanllm.infrastructure.distributed.messaging.get_rate_limiter",
            return_value=make_mock_rate_limiter(),
        ),
        patch(
            "beanllm.infrastructure.distributed.messaging.get_distributed_lock",
            return_value=make_mock_lock(),
        ),
    ):
        from beanllm.infrastructure.distributed.messaging import DistributedErrorHandler

        handler = DistributedErrorHandler.__new__(DistributedErrorHandler)
        # Build MessageProducer with mocked data
        from beanllm.infrastructure.distributed.messaging import MessageProducer

        mp = MessageProducer.__new__(MessageProducer)
        mp.producer = producer_mock
        mp.redis = redis_mock
        handler.message_producer = mp
        handler.redis = redis_mock
        yield handler, producer_mock, redis_mock


class TestDistributedErrorHandler:
    async def test_handle_error_publishes_event(self, error_handler):
        handler, producer, _ = error_handler
        err = ValueError("test error")
        await handler.handle_error("req-1", err, "ocr.process")
        producer.publish.assert_called()

    async def test_handle_error_saves_to_redis(self, error_handler):
        handler, _, redis = error_handler
        err = RuntimeError("failure")
        await handler.handle_error("req-2", err, "rag.query")
        redis.setex.assert_called()

    async def test_handle_error_without_redis(self, error_handler):
        handler, producer, _ = error_handler
        handler.redis = None
        err = Exception("no redis")
        await handler.handle_error("req-3", err, "audio.transcribe")
        # Should not raise; event should still be published
        producer.publish.assert_called()

    async def test_handle_error_updates_request_status(self, error_handler):
        handler, _, redis = error_handler
        err = Exception("something failed")
        await handler.handle_error("req-4", err, "embedding.compute", context={"info": "x"})
        # setex should be called multiple times (error log + status update + error count)
        assert redis.setex.call_count >= 1

    async def test_handle_error_increments_error_count(self, error_handler):
        handler, _, redis = error_handler
        err = Exception("error")
        await handler.handle_error("req-5", err, "ocr.process")
        redis.incr.assert_called()

    async def test_get_error_log_returns_none_without_redis(self, error_handler):
        handler, _, _ = error_handler
        handler.redis = None
        result = await handler.get_error_log("req-x")
        assert result is None

    async def test_get_error_log_returns_none_when_not_found(self, error_handler):
        handler, _, redis = error_handler
        redis.get.return_value = None
        result = await handler.get_error_log("req-missing")
        assert result is None

    async def test_get_error_log_returns_dict_when_found(self, error_handler):
        handler, _, redis = error_handler
        error_data = {"error_type": "ValueError", "error_message": "test"}
        redis.get.return_value = json.dumps(error_data).encode("utf-8")
        result = await handler.get_error_log("req-found")
        assert result == error_data

    async def test_get_error_log_decodes_bytes(self, error_handler):
        handler, _, redis = error_handler
        error_data = {"error_message": "bytes test"}
        redis.get.return_value = json.dumps(error_data).encode("utf-8")
        result = await handler.get_error_log("req-bytes")
        assert result["error_message"] == "bytes test"

    async def test_get_error_stats_without_redis(self, error_handler):
        handler, _, _ = error_handler
        handler.redis = None
        result = await handler.get_error_stats("ocr.process")
        assert result["error_count"] == 0
        assert result["operation"] == "ocr.process"

    async def test_get_error_stats_with_redis(self, error_handler):
        handler, _, redis = error_handler
        redis.get.return_value = b"5"
        result = await handler.get_error_stats("rag.query")
        assert result["error_count"] == 5

    async def test_get_error_stats_handles_redis_error(self, error_handler):
        handler, _, redis = error_handler
        redis.get.side_effect = Exception("Redis error")
        result = await handler.get_error_stats("failing.op")
        assert result["error_count"] == 0


# ---------------------------------------------------------------------------
# RequestMonitor tests
# ---------------------------------------------------------------------------


@pytest.fixture
def request_monitor():
    """RequestMonitor with mocked dependencies."""
    redis_mock = make_mock_redis()
    producer_mock, consumer_mock = make_mock_event_bus()

    with (
        patch(
            "beanllm.infrastructure.distributed.messaging.get_event_bus",
            return_value=(producer_mock, consumer_mock),
        ),
        patch(
            "beanllm.infrastructure.distributed.messaging.get_rate_limiter",
            return_value=make_mock_rate_limiter(),
        ),
        patch(
            "beanllm.infrastructure.distributed.messaging.get_distributed_lock",
            return_value=make_mock_lock(),
        ),
    ):
        from beanllm.infrastructure.distributed.messaging import RequestMonitor

        monitor = RequestMonitor.__new__(RequestMonitor)
        monitor.redis = redis_mock
        monitor.consumer = consumer_mock
        yield monitor, redis_mock, consumer_mock


class TestRequestMonitor:
    async def test_get_request_status_returns_none_without_redis(self, request_monitor):
        monitor, _, _ = request_monitor
        monitor.redis = None
        result = await monitor.get_request_status("req-1")
        assert result is None

    async def test_get_request_status_returns_none_when_not_found(self, request_monitor):
        monitor, redis, _ = request_monitor
        redis.get.return_value = None
        result = await monitor.get_request_status("req-missing")
        assert result is None

    async def test_get_request_status_returns_dict(self, request_monitor):
        monitor, redis, _ = request_monitor
        status_data = {"status": "pending", "request_type": "ocr.recognize"}
        redis.get.return_value = json.dumps(status_data).encode("utf-8")
        result = await monitor.get_request_status("req-1")
        assert result == status_data

    async def test_get_request_status_handles_redis_error(self, request_monitor):
        monitor, redis, _ = request_monitor
        redis.get.side_effect = Exception("Redis unavailable")
        result = await monitor.get_request_status("req-err")
        assert result is None

    async def test_get_error_log_returns_none_without_redis(self, request_monitor):
        monitor, _, _ = request_monitor
        monitor.redis = None
        result = await monitor.get_error_log("req-1")
        assert result is None

    async def test_get_error_log_returns_none_when_not_found(self, request_monitor):
        monitor, redis, _ = request_monitor
        redis.get.return_value = None
        result = await monitor.get_error_log("req-missing")
        assert result is None

    async def test_get_error_log_returns_dict_when_found(self, request_monitor):
        monitor, redis, _ = request_monitor
        error_data = {"error_type": "RuntimeError", "error_message": "crash"}
        redis.get.return_value = json.dumps(error_data).encode("utf-8")
        result = await monitor.get_error_log("req-found")
        assert result == error_data

    async def test_get_error_log_handles_redis_error(self, request_monitor):
        monitor, redis, _ = request_monitor
        redis.get.side_effect = Exception("Redis crash")
        result = await monitor.get_error_log("req-err")
        assert result is None

    async def test_get_request_logs_without_consumer_returns_empty(self, request_monitor):
        monitor, _, _ = request_monitor
        monitor.consumer = None
        # Should complete without yielding anything
        results = []
        async for item in monitor.get_request_logs("req-1"):
            results.append(item)
        assert results == []
