"""Tests for infrastructure/distributed/kafka/ (client, events, queue).

The kafka source files contain a broken import:
    `from beanllm.utils import check_kafka_health`
but check_kafka_health lives in beanllm.infrastructure.distributed.utils.
We inject it into beanllm.utils at module load time so that all kafka
imports work in the test environment.
"""

import asyncio
import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---- Patch beanllm.utils to expose check_kafka_health ----
import beanllm.utils as _bu

if not hasattr(_bu, "check_kafka_health"):
    _check_kafka_stub = AsyncMock(return_value=True)
    _bu.check_kafka_health = _check_kafka_stub
    # Also register in sys.modules so `from beanllm.utils import check_kafka_health` works
    setattr(sys.modules["beanllm.utils"], "check_kafka_health", _check_kafka_stub)
# ---- End patch ----


def _make_kafka_mocks():
    """Return mock (KafkaProducer class, KafkaConsumer class) pair."""
    mock_prod_cls = MagicMock()
    mock_cons_cls = MagicMock()
    mock_prod_cls.return_value = MagicMock()
    mock_cons_cls.return_value = MagicMock()
    return mock_prod_cls, mock_cons_cls


# ---------------------------------------------------------------------------
# kafka/client.py
# ---------------------------------------------------------------------------


class TestGetKafkaClientImportError:
    def test_raises_import_error_when_kafka_not_installed(self):
        import beanllm.infrastructure.distributed.kafka.client as mod

        original = mod.KafkaProducer
        try:
            mod.KafkaProducer = None
            mod._kafka_producer = None
            mod._kafka_consumer = None
            with pytest.raises(ImportError, match="kafka-python"):
                mod.get_kafka_client()
        finally:
            mod.KafkaProducer = original
            mod._kafka_producer = None
            mod._kafka_consumer = None


class TestGetKafkaClientSuccess:
    def setup_method(self):
        import beanllm.infrastructure.distributed.kafka.client as mod

        mod._kafka_producer = None
        mod._kafka_consumer = None

    def teardown_method(self):
        import beanllm.infrastructure.distributed.kafka.client as mod

        mod._kafka_producer = None
        mod._kafka_consumer = None

    def test_returns_producer_consumer_tuple(self):
        mock_prod_cls, mock_cons_cls = _make_kafka_mocks()
        import beanllm.infrastructure.distributed.kafka.client as mod

        orig_p, orig_c = mod.KafkaProducer, mod.KafkaConsumer
        try:
            mod.KafkaProducer = mock_prod_cls
            mod.KafkaConsumer = mock_cons_cls
            producer, consumer = mod.get_kafka_client()
            assert producer is mock_prod_cls.return_value
            assert consumer is mock_cons_cls.return_value
        finally:
            mod.KafkaProducer, mod.KafkaConsumer = orig_p, orig_c

    def test_cached_client_returned_on_second_call(self):
        mock_prod_cls, mock_cons_cls = _make_kafka_mocks()
        import beanllm.infrastructure.distributed.kafka.client as mod

        orig_p, orig_c = mod.KafkaProducer, mod.KafkaConsumer
        try:
            mod.KafkaProducer = mock_prod_cls
            mod.KafkaConsumer = mock_cons_cls
            p1, _ = mod.get_kafka_client()
            p2, _ = mod.get_kafka_client()
            assert p1 is p2
            assert mock_prod_cls.call_count == 1
        finally:
            mod.KafkaProducer, mod.KafkaConsumer = orig_p, orig_c

    def test_fallback_when_first_connect_raises(self):
        mock_prod_cls, mock_cons_cls = _make_kafka_mocks()
        import beanllm.infrastructure.distributed.kafka.client as mod

        orig_p, orig_c = mod.KafkaProducer, mod.KafkaConsumer
        try:
            fallback_prod = MagicMock()
            # Producer raises on first call (in try block), succeeds on second (in except/fallback)
            # Consumer is only called in the fallback path so its first call must succeed
            mock_prod_cls.side_effect = [RuntimeError("connect fail"), fallback_prod]
            mock_cons_cls.return_value = MagicMock()  # always succeeds
            mod.KafkaProducer = mock_prod_cls
            mod.KafkaConsumer = mock_cons_cls
            producer, _ = mod.get_kafka_client()
            assert producer is fallback_prod
        finally:
            mod.KafkaProducer, mod.KafkaConsumer = orig_p, orig_c

    def test_uses_env_vars(self, monkeypatch):
        mock_prod_cls, mock_cons_cls = _make_kafka_mocks()
        import beanllm.infrastructure.distributed.kafka.client as mod

        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "broker1:9092,broker2:9092")
        monkeypatch.setenv("KAFKA_CLIENT_ID", "test-client")
        orig_p, orig_c = mod.KafkaProducer, mod.KafkaConsumer
        try:
            mod.KafkaProducer = mock_prod_cls
            mod.KafkaConsumer = mock_cons_cls
            mod.get_kafka_client()
            call_kwargs = mock_prod_cls.call_args[1]
            assert call_kwargs["bootstrap_servers"] == ["broker1:9092", "broker2:9092"]
            assert call_kwargs["client_id"] == "test-client"
        finally:
            mod.KafkaProducer, mod.KafkaConsumer = orig_p, orig_c


class TestCloseKafkaClient:
    def test_close_clears_globals(self):
        import beanllm.infrastructure.distributed.kafka.client as mod

        mock_prod, mock_cons = MagicMock(), MagicMock()
        mod._kafka_producer = mock_prod
        mod._kafka_consumer = mock_cons
        mod.close_kafka_client()
        assert mod._kafka_producer is None
        assert mod._kafka_consumer is None
        mock_prod.close.assert_called_once()
        mock_cons.close.assert_called_once()

    def test_close_when_already_none_is_safe(self):
        import beanllm.infrastructure.distributed.kafka.client as mod

        mod._kafka_producer = None
        mod._kafka_consumer = None
        mod.close_kafka_client()  # should not raise


# ---------------------------------------------------------------------------
# kafka/events.py
# ---------------------------------------------------------------------------


class TestKafkaEventProducer:
    def _make_ep(self):
        from beanllm.infrastructure.distributed.kafka.events import KafkaEventProducer

        mock_prod, mock_cons = MagicMock(), MagicMock()
        ep = KafkaEventProducer(kafka_client=(mock_prod, mock_cons))
        return ep, mock_prod

    def test_init_stores_producer(self):
        ep, mock_prod = self._make_ep()
        assert ep.producer is mock_prod

    async def test_publish_skips_when_kafka_unhealthy(self):
        ep, mock_prod = self._make_ep()
        with patch(
            "beanllm.infrastructure.distributed.kafka.events.check_kafka_health",
            new=AsyncMock(return_value=False),
        ):
            await ep.publish("test.topic", {"key": "val"})
        mock_prod.send.assert_not_called()

    async def test_publish_sends_when_kafka_healthy(self):
        ep, mock_prod = self._make_ep()
        with (
            patch(
                "beanllm.infrastructure.distributed.kafka.events.check_kafka_health",
                new=AsyncMock(return_value=True),
            ),
            patch("asyncio.wait_for", new=AsyncMock(return_value=None)),
        ):
            await ep.publish("chat.events", {"action": "start"})
        mock_prod.flush.assert_called_once()

    async def test_publish_handles_timeout(self):
        ep, _ = self._make_ep()
        with (
            patch(
                "beanllm.infrastructure.distributed.kafka.events.check_kafka_health",
                new=AsyncMock(return_value=True),
            ),
            patch("asyncio.wait_for", new=AsyncMock(side_effect=asyncio.TimeoutError())),
        ):
            await ep.publish("chat.events", {"action": "start"})

    async def test_publish_handles_generic_exception(self):
        ep, _ = self._make_ep()
        with (
            patch(
                "beanllm.infrastructure.distributed.kafka.events.check_kafka_health",
                new=AsyncMock(return_value=True),
            ),
            patch("asyncio.wait_for", new=AsyncMock(side_effect=RuntimeError("fail"))),
        ):
            await ep.publish("topic", {"x": 1})

    async def test_publish_includes_event_metadata(self):
        ep, mock_prod = self._make_ep()
        captured = {}

        async def fake_wait_for(coro, timeout):
            return None

        with (
            patch(
                "beanllm.infrastructure.distributed.kafka.events.check_kafka_health",
                new=AsyncMock(return_value=True),
            ),
            patch("asyncio.wait_for", side_effect=fake_wait_for),
        ):
            await ep.publish("my.topic", {"user_id": "123"})
        mock_prod.flush.assert_called()


class TestKafkaEventConsumer:
    def test_init_stores_consumer(self):
        from beanllm.infrastructure.distributed.kafka.events import KafkaEventConsumer

        mock_prod, mock_cons = MagicMock(), MagicMock()
        ec = KafkaEventConsumer(kafka_client=(mock_prod, mock_cons))
        assert ec.consumer is mock_cons

    async def test_subscribe_yields_events(self):
        from beanllm.infrastructure.distributed.kafka.events import KafkaEventConsumer

        mock_prod, mock_cons = MagicMock(), MagicMock()
        ec = KafkaEventConsumer(kafka_client=(mock_prod, mock_cons))

        event_data = {"event_id": "abc", "data": {"msg": "hello"}}
        message = MagicMock()
        message.value = json.dumps(event_data)
        mock_cons.subscribe = MagicMock()

        call_count = 0

        async def fake_to_thread(fn, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"tp": [message]}
            return {}

        collected = []
        handler = MagicMock()

        with patch("asyncio.to_thread", side_effect=fake_to_thread):
            async for event in ec.subscribe("topic", handler):
                collected.append(event)
                break

        assert len(collected) == 1
        assert collected[0] == event_data
        handler.assert_called_once_with(event_data)

    async def test_subscribe_calls_async_handler(self):
        from beanllm.infrastructure.distributed.kafka.events import KafkaEventConsumer

        mock_prod, mock_cons = MagicMock(), MagicMock()
        ec = KafkaEventConsumer(kafka_client=(mock_prod, mock_cons))

        event_data = {"event_id": "xyz"}
        message = MagicMock()
        message.value = json.dumps(event_data)
        mock_cons.subscribe = MagicMock()

        call_count = 0

        async def fake_to_thread(fn, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return {"tp": [message]} if call_count == 1 else {}

        async_handler = AsyncMock()

        with patch("asyncio.to_thread", side_effect=fake_to_thread):
            async for _ in ec.subscribe("topic", async_handler):
                break

        async_handler.assert_called_once_with(event_data)


# ---------------------------------------------------------------------------
# kafka/queue.py
# ---------------------------------------------------------------------------


class TestKafkaTaskQueue:
    def _make_queue(self, topic="llm.tasks"):
        from beanllm.infrastructure.distributed.kafka.queue import KafkaTaskQueue

        mock_prod, mock_cons = MagicMock(), MagicMock()
        q = KafkaTaskQueue(kafka_client=(mock_prod, mock_cons), topic=topic)
        return q, mock_prod, mock_cons

    def test_stores_topic(self):
        q, _, _ = self._make_queue(topic="my.topic")
        assert q.topic == "my.topic"

    async def test_enqueue_fallback_when_kafka_unhealthy(self):
        q, _, _ = self._make_queue()
        with patch(
            "beanllm.infrastructure.distributed.kafka.queue.check_kafka_health",
            new=AsyncMock(return_value=False),
        ):
            task_id = await q.enqueue("llm_chat", {"prompt": "hi"})
        assert isinstance(task_id, str)
        assert task_id in q._task_status
        assert q._task_status[task_id]["status"] == "pending"

    async def test_enqueue_healthy_kafka_flushes(self):
        q, mock_prod, _ = self._make_queue()
        mock_prod.flush = MagicMock()
        with (
            patch(
                "beanllm.infrastructure.distributed.kafka.queue.check_kafka_health",
                new=AsyncMock(return_value=True),
            ),
            patch("asyncio.wait_for", new=AsyncMock(return_value=None)),
        ):
            task_id = await q.enqueue("rag", {"query": "test"})
        assert task_id in q._task_status
        mock_prod.flush.assert_called_once()

    async def test_enqueue_fallback_on_timeout(self):
        q, _, _ = self._make_queue()
        with (
            patch(
                "beanllm.infrastructure.distributed.kafka.queue.check_kafka_health",
                new=AsyncMock(return_value=True),
            ),
            patch("asyncio.wait_for", new=AsyncMock(side_effect=asyncio.TimeoutError())),
        ):
            task_id = await q.enqueue("slow_task", {"data": "x"})
        assert task_id in q._task_status

    async def test_enqueue_fallback_on_send_error(self):
        q, _, _ = self._make_queue()
        with (
            patch(
                "beanllm.infrastructure.distributed.kafka.queue.check_kafka_health",
                new=AsyncMock(return_value=True),
            ),
            patch("asyncio.wait_for", new=AsyncMock(side_effect=RuntimeError("send failed"))),
        ):
            task_id = await q.enqueue("failing_task", {"x": 1})
        assert task_id in q._task_status

    async def test_enqueue_with_priority(self):
        q, _, _ = self._make_queue()
        with patch(
            "beanllm.infrastructure.distributed.kafka.queue.check_kafka_health",
            new=AsyncMock(return_value=False),
        ):
            task_id = await q.enqueue("task", {"data": "x"}, priority=5)
        assert task_id in q._task_status

    async def test_get_task_status_none_for_unknown_id(self):
        q, _, _ = self._make_queue()
        result = await q.get_task_status("nonexistent-uuid")
        assert result is None

    async def test_get_task_status_after_enqueue(self):
        q, _, _ = self._make_queue()
        with patch(
            "beanllm.infrastructure.distributed.kafka.queue.check_kafka_health",
            new=AsyncMock(return_value=False),
        ):
            task_id = await q.enqueue("mytask", {"val": 42})
        status = await q.get_task_status(task_id)
        assert status is not None
        assert status["status"] == "pending"

    async def test_dequeue_returns_matching_task(self):
        q, _, mock_cons = self._make_queue()

        task_payload = {
            "task_id": "test-id-123",
            "task_type": "my_task",
            "data": {"val": 99},
            "priority": 0,
            "created_at": 1000.0,
            "status": "pending",
        }
        message = MagicMock()
        message.value = json.dumps(task_payload)
        mock_cons.subscribe = MagicMock()

        call_count_holder = [0]

        async def fake_to_thread(fn, *args, **kwargs):
            idx = call_count_holder[0]
            call_count_holder[0] += 1
            return {"tp": [message]} if idx == 0 else {}

        with patch("asyncio.to_thread", side_effect=fake_to_thread):
            result = await q.dequeue("my_task", timeout=10)

        assert result is not None
        assert result["task_type"] == "my_task"
        assert result["task_id"] == "test-id-123"

    async def test_dequeue_returns_none_on_timeout_no_match(self):
        q, _, mock_cons = self._make_queue()
        mock_cons.subscribe = MagicMock()

        async def fake_to_thread(fn, *args, **kwargs):
            return {}

        with patch("asyncio.to_thread", side_effect=fake_to_thread):
            result = await q.dequeue("missing_task", timeout=0.05)

        assert result is None

    async def test_dequeue_updates_task_status_to_processing(self):
        q, _, mock_cons = self._make_queue()

        task_id = "known-id"
        q._task_status[task_id] = {"status": "pending", "created_at": 1000.0}

        task_payload = {
            "task_id": task_id,
            "task_type": "proc_task",
            "data": {},
            "priority": 0,
            "created_at": 1000.0,
            "status": "pending",
        }
        message = MagicMock()
        message.value = json.dumps(task_payload)
        mock_cons.subscribe = MagicMock()

        call_count_holder = [0]

        async def fake_to_thread(fn, *args, **kwargs):
            idx = call_count_holder[0]
            call_count_holder[0] += 1
            return {"tp": [message]} if idx == 0 else {}

        with patch("asyncio.to_thread", side_effect=fake_to_thread):
            await q.dequeue("proc_task", timeout=5)

        assert q._task_status[task_id]["status"] == "processing"
        assert "started_at" in q._task_status[task_id]
