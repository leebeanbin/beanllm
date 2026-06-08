"""Tests for infrastructure/distributed/task_processor.py — TaskProcessor, BatchProcessor."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _mock_task_queue():
    q = MagicMock()
    q.enqueue = AsyncMock(return_value="task-123")
    q.dequeue = AsyncMock(return_value=None)
    q.get_task_status = AsyncMock(return_value={"status": "pending"})
    return q


def _mock_message_producer():
    mp = MagicMock()
    mp.publish_request = AsyncMock()
    mp.publish_event = AsyncMock()
    return mp


def _mock_error_handler():
    eh = MagicMock()
    eh.handle_error = AsyncMock()
    return eh


def _make_task_processor():
    """Create TaskProcessor with mocked dependencies."""
    mock_queue = _mock_task_queue()
    mock_producer = _mock_message_producer()
    mock_error_handler = _mock_error_handler()

    with (
        patch(
            "beanllm.infrastructure.distributed.task_processor.get_task_queue",
            return_value=mock_queue,
        ),
        patch(
            "beanllm.infrastructure.distributed.task_processor.MessageProducer",
            return_value=mock_producer,
        ),
        patch(
            "beanllm.infrastructure.distributed.task_processor.DistributedErrorHandler",
            return_value=mock_error_handler,
        ),
    ):
        from beanllm.infrastructure.distributed.task_processor import TaskProcessor

        processor = TaskProcessor("test.tasks")

    processor.task_queue = mock_queue
    processor.message_producer = mock_producer
    processor.error_handler = mock_error_handler
    return processor, mock_queue, mock_producer, mock_error_handler


# ---------------------------------------------------------------------------
# TaskProcessor
# ---------------------------------------------------------------------------


class TestTaskProcessorInit:
    def test_creates_with_defaults(self):
        mock_queue = _mock_task_queue()
        with (
            patch(
                "beanllm.infrastructure.distributed.task_processor.get_task_queue",
                return_value=mock_queue,
            ),
            patch(
                "beanllm.infrastructure.distributed.task_processor.MessageProducer",
                return_value=_mock_message_producer(),
            ),
            patch(
                "beanllm.infrastructure.distributed.task_processor.DistributedErrorHandler",
                return_value=_mock_error_handler(),
            ),
        ):
            from beanllm.infrastructure.distributed.task_processor import TaskProcessor

            proc = TaskProcessor()
        assert proc.task_type == "llm.tasks"

    def test_custom_task_type(self):
        mock_queue = _mock_task_queue()
        with (
            patch(
                "beanllm.infrastructure.distributed.task_processor.get_task_queue",
                return_value=mock_queue,
            ),
            patch(
                "beanllm.infrastructure.distributed.task_processor.MessageProducer",
                return_value=_mock_message_producer(),
            ),
            patch(
                "beanllm.infrastructure.distributed.task_processor.DistributedErrorHandler",
                return_value=_mock_error_handler(),
            ),
        ):
            from beanllm.infrastructure.distributed.task_processor import TaskProcessor

            proc = TaskProcessor("ocr.tasks")
        assert proc.task_type == "ocr.tasks"


class TestEnqueueTask:
    async def test_enqueue_returns_task_id(self):
        processor, q, mp, _ = _make_task_processor()
        task_id = await processor.enqueue_task("recognize", {"file": "test.pdf"})
        assert task_id == "task-123"

    async def test_enqueue_calls_task_queue(self):
        processor, q, mp, _ = _make_task_processor()
        await processor.enqueue_task("recognize", {"file": "test.pdf"}, priority=5)
        q.enqueue.assert_awaited_once_with(
            task_type="test.tasks:recognize",
            data={"file": "test.pdf"},
            priority=5,
        )

    async def test_enqueue_publishes_request(self):
        processor, q, mp, _ = _make_task_processor()
        await processor.enqueue_task("recognize", {"file": "doc.pdf"})
        mp.publish_request.assert_awaited_once()

    async def test_enqueue_raises_on_queue_failure(self):
        processor, q, mp, _ = _make_task_processor()
        q.enqueue = AsyncMock(side_effect=RuntimeError("queue down"))
        with pytest.raises(RuntimeError):
            await processor.enqueue_task("recognize", {})


class TestProcessTask:
    async def test_returns_none_when_no_task(self):
        processor, q, mp, _ = _make_task_processor()
        q.dequeue = AsyncMock(return_value=None)
        result = await processor.process_task("recognize", handler=AsyncMock())
        assert result is None

    async def test_processes_async_handler(self):
        processor, q, mp, eh = _make_task_processor()
        q.dequeue = AsyncMock(
            return_value={
                "task_id": "t1",
                "data": {"file": "test.pdf"},
            }
        )

        async def handler(data):
            return {"text": "extracted"}

        result = await processor.process_task("recognize", handler=handler)
        assert result is not None
        assert result["result"] == {"text": "extracted"}

    async def test_processes_sync_handler(self):
        processor, q, mp, eh = _make_task_processor()
        q.dequeue = AsyncMock(
            return_value={
                "task_id": "t1",
                "data": {"value": 42},
            }
        )

        def sync_handler(data):
            return {"doubled": data["value"] * 2}

        result = await processor.process_task("compute", handler=sync_handler)
        assert result["result"]["doubled"] == 84

    async def test_returns_none_on_exception(self):
        processor, q, mp, eh = _make_task_processor()
        q.dequeue = AsyncMock(side_effect=RuntimeError("dequeue failed"))
        result = await processor.process_task("recognize", handler=AsyncMock())
        assert result is None

    async def test_handler_exception_is_handled(self):
        processor, q, mp, eh = _make_task_processor()
        q.dequeue = AsyncMock(
            return_value={
                "task_id": "t1",
                "data": {},
            }
        )

        async def failing_handler(data):
            raise ValueError("processing failed")

        result = await processor.process_task("recognize", handler=failing_handler)
        # Error handled internally, returns None
        assert result is None
        eh.handle_error.assert_awaited()

    async def test_publishes_started_event(self):
        processor, q, mp, eh = _make_task_processor()
        q.dequeue = AsyncMock(return_value={"task_id": "t1", "data": {}})
        await processor.process_task("recognize", handler=AsyncMock(return_value={}))
        # publish_event should have been called for started
        assert mp.publish_event.await_count >= 1

    async def test_publishes_completed_event(self):
        processor, q, mp, eh = _make_task_processor()
        q.dequeue = AsyncMock(return_value={"task_id": "t1", "data": {}})
        await processor.process_task("recognize", handler=AsyncMock(return_value={"ok": True}))
        event_names = [call.args[0] for call in mp.publish_event.call_args_list]
        assert any("completed" in ev for ev in event_names)

    async def test_with_timeout(self):
        processor, q, mp, eh = _make_task_processor()
        q.dequeue = AsyncMock(return_value=None)
        result = await processor.process_task("recognize", handler=AsyncMock(), timeout=0.1)
        assert result is None


class TestBatchEnqueue:
    async def test_enqueues_all_tasks(self):
        processor, q, mp, eh = _make_task_processor()
        q.enqueue = AsyncMock(side_effect=["id-1", "id-2", "id-3"])
        task_ids = await processor.batch_enqueue(
            "recognize",
            [{"f": "a.pdf"}, {"f": "b.pdf"}, {"f": "c.pdf"}],
        )
        assert len(task_ids) == 3
        assert q.enqueue.await_count == 3

    async def test_returns_task_ids(self):
        processor, q, mp, eh = _make_task_processor()
        q.enqueue = AsyncMock(side_effect=["id-1", "id-2"])
        task_ids = await processor.batch_enqueue("recognize", [{"x": 1}, {"x": 2}])
        assert task_ids == ["id-1", "id-2"]

    async def test_empty_list_returns_empty(self):
        processor, q, mp, eh = _make_task_processor()
        task_ids = await processor.batch_enqueue("recognize", [])
        assert task_ids == []


class TestGetTaskStatus:
    async def test_returns_status(self):
        processor, q, mp, eh = _make_task_processor()
        q.get_task_status = AsyncMock(return_value={"status": "completed"})
        result = await processor.get_task_status("task-123")
        assert result == {"status": "completed"}
        q.get_task_status.assert_awaited_once_with("task-123")

    async def test_returns_none_for_unknown_task(self):
        processor, q, mp, eh = _make_task_processor()
        q.get_task_status = AsyncMock(return_value=None)
        result = await processor.get_task_status("unknown-id")
        assert result is None


# ---------------------------------------------------------------------------
# BatchProcessor
# ---------------------------------------------------------------------------


def _make_batch_processor():
    mock_queue = _mock_task_queue()
    mock_producer = _mock_message_producer()
    mock_error_handler = _mock_error_handler()

    with (
        patch(
            "beanllm.infrastructure.distributed.task_processor.get_task_queue",
            return_value=mock_queue,
        ),
        patch(
            "beanllm.infrastructure.distributed.task_processor.MessageProducer",
            return_value=mock_producer,
        ),
        patch(
            "beanllm.infrastructure.distributed.task_processor.DistributedErrorHandler",
            return_value=mock_error_handler,
        ),
    ):
        from beanllm.infrastructure.distributed.task_processor import BatchProcessor

        processor = BatchProcessor("test.tasks", max_concurrent=5)

    # Patch internals
    processor.task_processor.task_queue = mock_queue
    processor.task_processor.message_producer = mock_producer
    processor.task_processor.error_handler = mock_error_handler
    processor.concurrency_controller = None  # No distributed controller
    return processor, mock_queue, mock_error_handler


class TestBatchProcessorInit:
    def test_creates_with_task_processor(self):
        mock_queue = _mock_task_queue()
        with (
            patch(
                "beanllm.infrastructure.distributed.task_processor.get_task_queue",
                return_value=mock_queue,
            ),
            patch(
                "beanllm.infrastructure.distributed.task_processor.MessageProducer",
                return_value=_mock_message_producer(),
            ),
            patch(
                "beanllm.infrastructure.distributed.task_processor.DistributedErrorHandler",
                return_value=_mock_error_handler(),
            ),
        ):
            from beanllm.infrastructure.distributed.task_processor import BatchProcessor

            bp = BatchProcessor("llm.tasks", max_concurrent=8)
        assert bp.max_concurrent == 8


class TestBatchProcessorProcessItems:
    async def test_processes_async_handler(self):
        bp, q, eh = _make_batch_processor()

        async def handler(item):
            return item * 2

        results = await bp.process_items([1, 2, 3], handler=handler)
        assert results == [2, 4, 6]

    async def test_processes_sync_handler(self):
        bp, q, eh = _make_batch_processor()

        def sync_handler(item):
            return item + 10

        results = await bp.process_items([1, 2, 3], handler=sync_handler)
        assert results == [11, 12, 13]

    async def test_error_returns_none_for_failed_item(self):
        bp, q, eh = _make_batch_processor()

        async def failing_handler(item):
            if item == 2:
                raise ValueError("bad item")
            return item

        results = await bp.process_items([1, 2, 3], handler=failing_handler)
        assert results[0] == 1
        assert results[1] is None  # failed item
        assert results[2] == 3

    async def test_empty_items_returns_empty(self):
        bp, q, eh = _make_batch_processor()
        results = await bp.process_items([], handler=AsyncMock())
        assert results == []

    async def test_respects_max_concurrent(self):
        bp, q, eh = _make_batch_processor()
        active = []
        max_active = [0]

        async def handler(item):
            active.append(item)
            max_active[0] = max(max_active[0], len(active))
            await asyncio.sleep(0)
            active.remove(item)
            return item

        await bp.process_items(list(range(10)), handler=handler)
        assert max_active[0] <= bp.max_concurrent

    async def test_custom_max_concurrent_override(self):
        bp, q, eh = _make_batch_processor()

        async def handler(item):
            return item

        results = await bp.process_items([1, 2, 3], handler=handler, max_concurrent=2)
        assert results == [1, 2, 3]
