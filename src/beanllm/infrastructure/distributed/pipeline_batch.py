"""Pipeline decorators - Batch processing decorator."""

from __future__ import annotations

import asyncio
import functools
from typing import Any, Callable, List, Optional

from beanllm.infrastructure.distributed.config import get_distributed_config
from beanllm.infrastructure.distributed.task_processor import BatchProcessor

try:
    from beanllm.utils.logging import get_logger
except ImportError:
    import logging

    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


logger = get_logger(__name__)


def with_batch_processing(
    pipeline_type: str = "default",
    max_concurrent: Optional[int] = None,
    use_distributed_queue: Optional[bool] = None,
) -> Callable[[Callable], Callable]:
    """
    Batch processing decorator. Applies to functions that take a list of items.
    """

    def decorator(func: Callable) -> Callable:
        config = get_distributed_config()
        pipeline_config = getattr(config, pipeline_type, None)
        if pipeline_config is None:
            use_queue = use_distributed_queue or False
            max_workers = max_concurrent or 4
        else:
            use_queue = (
                use_distributed_queue
                if use_distributed_queue is not None
                else pipeline_config.use_distributed_queue
            )
            max_workers = (
                max_concurrent
                if max_concurrent is not None
                else getattr(pipeline_config, "max_concurrent", 4)
            )

        @functools.wraps(func)
        def sync_wrapper(self: Any, items: list[Any], *args: Any, **kwargs: Any) -> List[Any]:
            if use_queue and len(items) > 1:
                processor = BatchProcessor(
                    task_type=f"{pipeline_type}.batch", max_concurrent=max_workers
                )

                async def _batch_async() -> List[Any]:
                    tasks_data = [{"item": item, "args": args, "kwargs": kwargs} for item in items]

                    def handler(task_data: dict[str, Any]) -> Any:
                        return func(
                            self, [task_data["item"]], *task_data["args"], **task_data["kwargs"]
                        )[0]

                    results = await processor.process_batch(
                        task_name=f"{pipeline_type}.batch",
                        tasks_data=tasks_data,
                        handler=handler,
                    )
                    return [r for r in results if r is not None]

                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        return [func(self, [item], *args, **kwargs)[0] for item in items]
                    return loop.run_until_complete(_batch_async())
                except RuntimeError:
                    return asyncio.run(_batch_async())
            results = []
            for item in items:
                result = func(self, [item], *args, **kwargs)
                results.extend(result if isinstance(result, list) else [result])
            return results

        @functools.wraps(func)
        async def async_wrapper(
            self: Any, items: List[Any], *args: Any, **kwargs: Any
        ) -> List[Any]:
            if use_queue and len(items) > 1:
                processor = BatchProcessor(
                    task_type=f"{pipeline_type}.batch", max_concurrent=max_workers
                )

                async def process_item(task_data: dict[str, Any]) -> Any:
                    result = await func(
                        self, [task_data["item"]], *task_data["args"], **task_data["kwargs"]
                    )
                    return result[0] if isinstance(result, list) and len(result) > 0 else result

                tasks_data = [{"item": item, "args": args, "kwargs": kwargs} for item in items]
                results = await processor.process_batch(
                    task_name=f"{pipeline_type}.batch",
                    tasks_data=tasks_data,
                    handler=process_item,
                )
                return [r for r in results if r is not None]
            sequential_results = []
            for item in items:
                result = await func(self, [item], *args, **kwargs)
                sequential_results.extend(result if isinstance(result, list) else [result])
            return sequential_results

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
