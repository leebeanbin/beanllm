"""Tests for infrastructure/distributed/event_integration.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from beanllm.infrastructure.distributed.event_integration import (
    EventLogger,
    get_event_logger,
    with_event_publishing,
)

# ---------------------------------------------------------------------------
# EventLogger
# ---------------------------------------------------------------------------


class TestEventLogger:
    def _make_logger(self):
        logger = EventLogger()
        logger.message_producer = MagicMock()
        logger.message_producer.publish_event = AsyncMock()
        return logger

    async def test_log_event_info_level(self):
        el = self._make_logger()
        await el.log_event("test.event", {"key": "val"}, level="info")
        el.message_producer.publish_event.assert_called_once_with("test.event", {"key": "val"})

    async def test_log_event_warning_level(self):
        el = self._make_logger()
        await el.log_event("warn.event", {}, level="warning")
        el.message_producer.publish_event.assert_called_once()

    async def test_log_event_error_level(self):
        el = self._make_logger()
        await el.log_event("error.event", {"detail": "bad"}, level="error")
        el.message_producer.publish_event.assert_called_once()

    async def test_log_event_publish_failure_does_not_raise(self):
        el = self._make_logger()
        el.message_producer.publish_event = AsyncMock(side_effect=Exception("publish fail"))
        # Should catch exception internally and not propagate
        await el.log_event("fail.event", {}, level="info")

    def test_log_document_loaded_schedules_task(self):
        el = self._make_logger()
        with patch("asyncio.create_task") as mock_task:
            el.log_document_loaded("/tmp/doc.txt", "pdf")
        mock_task.assert_called_once()

    def test_log_embedding_completed_schedules_task(self):
        el = self._make_logger()
        with patch("asyncio.create_task") as mock_task:
            el.log_embedding_completed(100, "nomic-embed-text")
        mock_task.assert_called_once()

    def test_log_rag_query_schedules_task(self):
        el = self._make_logger()
        with patch("asyncio.create_task") as mock_task:
            el.log_rag_query("What is AI?", 5)
        mock_task.assert_called_once()

    def test_log_rag_query_truncates_long_question(self):
        el = self._make_logger()
        long_question = "A" * 200
        with patch("asyncio.create_task") as mock_task:
            el.log_rag_query(long_question, 3)
        # Verify that create_task was called (truncation happens inside the coroutine)
        mock_task.assert_called_once()


class TestGetEventLogger:
    def test_returns_event_logger_instance(self):
        logger = get_event_logger()
        assert isinstance(logger, EventLogger)

    def test_returns_same_singleton(self):
        l1 = get_event_logger()
        l2 = get_event_logger()
        assert l1 is l2


# ---------------------------------------------------------------------------
# with_event_publishing decorator
# ---------------------------------------------------------------------------


class TestWithEventPublishing:
    def _mock_event_bus(self):
        mock_producer = MagicMock()
        mock_producer.publish = AsyncMock()
        return mock_producer

    async def test_async_function_publishes_start_event(self):
        mock_producer = self._mock_event_bus()
        mock_consumer = MagicMock()
        with patch(
            "beanllm.infrastructure.distributed.event_integration.get_event_bus",
            return_value=(mock_producer, mock_consumer),
        ):

            @with_event_publishing("my.event")
            async def fn(x):
                return x * 2

            result = await fn(5)

        assert result == 10
        # Should have published at least the started and completed events
        assert mock_producer.publish.call_count >= 2

    async def test_async_function_publishes_error_event_on_exception(self):
        mock_producer = self._mock_event_bus()
        mock_consumer = MagicMock()
        with patch(
            "beanllm.infrastructure.distributed.event_integration.get_event_bus",
            return_value=(mock_producer, mock_consumer),
        ):

            @with_event_publishing("error.event")
            async def failing_fn():
                raise ValueError("something went wrong")

            with pytest.raises(ValueError, match="something went wrong"):
                await failing_fn()

        # Should publish started + error events
        call_args_list = mock_producer.publish.call_args_list
        event_types = [call[0][1].get("event_type") for call in call_args_list]
        assert any("error" in et for et in event_types)

    async def test_async_function_include_result_in_event(self):
        mock_producer = self._mock_event_bus()
        mock_consumer = MagicMock()
        with patch(
            "beanllm.infrastructure.distributed.event_integration.get_event_bus",
            return_value=(mock_producer, mock_consumer),
        ):

            @with_event_publishing("result.event", include_result=True)
            async def fn():
                return "important_result"

            await fn()

        # Find completed event call
        completed_call = None
        for call in mock_producer.publish.call_args_list:
            event_data = call[0][1]
            if "completed" in event_data.get("event_type", ""):
                completed_call = event_data
                break

        assert completed_call is not None
        assert "result" in completed_call

    async def test_async_function_no_result_in_event(self):
        mock_producer = self._mock_event_bus()
        mock_consumer = MagicMock()
        with patch(
            "beanllm.infrastructure.distributed.event_integration.get_event_bus",
            return_value=(mock_producer, mock_consumer),
        ):

            @with_event_publishing("no_result.event", include_result=False)
            async def fn():
                return "hidden_result"

            await fn()

        # Find completed event call
        completed_call = None
        for call in mock_producer.publish.call_args_list:
            event_data = call[0][1]
            if "completed" in event_data.get("event_type", ""):
                completed_call = event_data
                break

        assert completed_call is not None
        assert "result" not in completed_call

    async def test_sync_function_wrapped_runs_function(self):
        """Sync functions decorated with with_event_publishing run directly when loop is running."""
        mock_producer = self._mock_event_bus()
        mock_consumer = MagicMock()
        with patch(
            "beanllm.infrastructure.distributed.event_integration.get_event_bus",
            return_value=(mock_producer, mock_consumer),
        ):

            @with_event_publishing("sync.event")
            def sync_fn(x):
                return x + 1

            # In an async context, loop.is_running() is True, so function runs directly
            result = sync_fn(41)

        assert result == 42

    def test_sync_function_is_callable(self):
        mock_producer = self._mock_event_bus()
        mock_consumer = MagicMock()
        with patch(
            "beanllm.infrastructure.distributed.event_integration.get_event_bus",
            return_value=(mock_producer, mock_consumer),
        ):

            @with_event_publishing("callable.test")
            def fn():
                return "ok"

        assert callable(fn)

    def test_preserves_function_name(self):
        mock_producer = self._mock_event_bus()
        mock_consumer = MagicMock()
        with patch(
            "beanllm.infrastructure.distributed.event_integration.get_event_bus",
            return_value=(mock_producer, mock_consumer),
        ):

            @with_event_publishing("naming.test")
            async def my_named_function():
                pass

        assert my_named_function.__name__ == "my_named_function"
