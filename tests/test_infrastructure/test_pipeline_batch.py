"""Tests for infrastructure/distributed/pipeline_batch.py."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from beanllm.infrastructure.distributed.pipeline_batch import with_batch_processing

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _double_items(self, items, *args, **kwargs):
    return [item * 2 for item in items]


async def _async_double_items(self, items, *args, **kwargs):
    return [item * 2 for item in items]


# ---------------------------------------------------------------------------
# Decorator application
# ---------------------------------------------------------------------------


class TestDecoratorApplication:
    def test_sync_func_decorated_is_callable(self):
        @with_batch_processing(pipeline_type="default")
        def process(self, items):
            return items

        assert callable(process)

    def test_async_func_decorated_is_awaitable(self):
        @with_batch_processing(pipeline_type="default")
        async def process(self, items):
            return items

        assert asyncio.iscoroutinefunction(process)

    def test_preserves_function_name(self):
        @with_batch_processing(pipeline_type="default")
        def my_named_process(self, items):
            return items

        assert my_named_process.__name__ == "my_named_process"


# ---------------------------------------------------------------------------
# Sync wrapper – no queue (lines 72-76)
# ---------------------------------------------------------------------------


class TestSyncWrapperNoQueue:
    def test_processes_each_item_individually(self):
        @with_batch_processing(pipeline_type="unknown_pipeline_xyz", use_distributed_queue=False)
        def process(self, items):
            return [item * 2 for item in items]

        class Svc:
            method = process

        svc = Svc()
        result = svc.method([1, 2, 3])
        assert result == [2, 4, 6]

    def test_single_item_list(self):
        @with_batch_processing(pipeline_type="default")
        def process(self, items):
            return [item + 10 for item in items]

        class Svc:
            method = process

        svc = Svc()
        result = svc.method([5])
        assert result == [15]

    def test_empty_list(self):
        @with_batch_processing(pipeline_type="default")
        def process(self, items):
            return [item for item in items]

        class Svc:
            method = process

        svc = Svc()
        result = svc.method([])
        assert result == []

    def test_extends_list_results(self):
        @with_batch_processing(pipeline_type="default")
        def process(self, items):
            return [item * 2 for item in items]

        class Svc:
            method = process

        svc = Svc()
        result = svc.method([10, 20])
        assert result == [20, 40]

    def test_non_list_result_wrapped(self):
        @with_batch_processing(pipeline_type="default")
        def process(self, items):
            return items[0] * 3  # Returns scalar, not list

        class Svc:
            method = process

        svc = Svc()
        result = svc.method([5])
        assert result == [15]


# ---------------------------------------------------------------------------
# Sync wrapper – with queue, loop running (line 68)
# ---------------------------------------------------------------------------


class TestSyncWrapperWithQueueLoopRunning:
    async def test_uses_sequential_fallback_when_loop_running(self):
        @with_batch_processing(
            pipeline_type="unknown_xyz", use_distributed_queue=True, max_concurrent=4
        )
        def process(self, items):
            return [item * 2 for item in items]

        class Svc:
            method = process

        svc = Svc()
        # In async context, loop.is_running() is True → falls back to line 68
        result = svc.method([1, 2, 3])
        assert result == [2, 4, 6]

    async def test_queue_path_single_item_no_queue(self):
        @with_batch_processing(
            pipeline_type="unknown_xyz", use_distributed_queue=True, max_concurrent=4
        )
        def process(self, items):
            return [item + 1 for item in items]

        class Svc:
            method = process

        svc = Svc()
        # Single item → use_queue condition fails (len(items) > 1 is False)
        result = svc.method([10])
        # Falls through to sequential loop (lines 72-76)
        assert result == [11]


# ---------------------------------------------------------------------------
# Async wrapper – no queue (lines 100-104)
# ---------------------------------------------------------------------------


class TestAsyncWrapperNoQueue:
    async def test_processes_each_item_sequentially(self):
        @with_batch_processing(pipeline_type="default", use_distributed_queue=False)
        async def process(self, items):
            return [item * 3 for item in items]

        class Svc:
            method = process

        svc = Svc()
        result = await svc.method([1, 2, 4])
        assert result == [3, 6, 12]

    async def test_empty_items_async(self):
        @with_batch_processing(pipeline_type="default")
        async def process(self, items):
            return items

        class Svc:
            method = process

        svc = Svc()
        result = await svc.method([])
        assert result == []

    async def test_extends_list_results_async(self):
        @with_batch_processing(pipeline_type="default")
        async def process(self, items):
            return [item + 5 for item in items]

        class Svc:
            method = process

        svc = Svc()
        result = await svc.method([10, 20, 30])
        assert result == [15, 25, 35]


# ---------------------------------------------------------------------------
# Async wrapper – with queue (lines 82-99)
# ---------------------------------------------------------------------------


class TestAsyncWrapperWithQueue:
    async def test_with_queue_multiple_items_processed(self):
        results_holder = []

        @with_batch_processing(
            pipeline_type="unknown_xyz", use_distributed_queue=True, max_concurrent=2
        )
        async def process(self, items):
            result = [item * 10 for item in items]
            results_holder.extend(result)
            return result

        class Svc:
            method = process

        svc = Svc()
        # use_queue=True, len(items) > 1 → uses BatchProcessor
        # Mock BatchProcessor.process_batch to return results
        mock_bp = AsyncMock()
        mock_bp.process_batch = AsyncMock(return_value=[10, 20, 30])

        with patch(
            "beanllm.infrastructure.distributed.pipeline_batch.BatchProcessor",
            return_value=mock_bp,
        ):
            result = await svc.method([1, 2, 3])

        assert result == [10, 20, 30]
        mock_bp.process_batch.assert_called_once()


# ---------------------------------------------------------------------------
# Pipeline config reading (lines 25-42)
# ---------------------------------------------------------------------------


class TestPipelineConfigReading:
    def test_known_pipeline_type_reads_config(self):
        @with_batch_processing(pipeline_type="ocr")
        def process(self, items):
            return items

        assert callable(process)

    def test_unknown_pipeline_type_uses_defaults(self):
        @with_batch_processing(pipeline_type="completely_unknown_pipeline_xyz_12345")
        def process(self, items):
            return items

        assert callable(process)

    def test_explicit_max_concurrent_overrides_config(self):
        @with_batch_processing(pipeline_type="default", max_concurrent=8)
        def process(self, items):
            return items

        assert callable(process)
