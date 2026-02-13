"""
작업 처리기 (Task Processor)

장기 작업을 큐에 추가하고 분산 처리
기존 최적화 패턴 참고: 에러 처리, 로깅, Helper 메서드
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional

from .factory import get_task_queue
from .messaging import DistributedErrorHandler, MessageProducer
from .utils import sanitize_error_message

try:
    from beanllm.utils.logging import get_logger
except ImportError:
    import logging

    def get_logger(name: str) -> logging.Logger:  # type: ignore[misc]
        return logging.getLogger(name)


logger = get_logger(__name__)


class TaskProcessor:
    """
    작업 처리기

    장기 작업을 큐에 추가하고 분산 처리
    """

    def __init__(self, task_type: str = "llm.tasks"):
        """
        Args:
            task_type: 작업 타입 (예: "ocr.tasks", "audio.tasks", "embedding.tasks")
        """
        self.task_queue = get_task_queue(task_type)
        self.message_producer = MessageProducer()
        self.error_handler = DistributedErrorHandler()
        self.task_type = task_type

    async def enqueue_task(
        self,
        task_name: str,
        task_data: Dict[str, Any],
        priority: int = 0,
    ) -> str:
        """
        작업을 큐에 추가

        Args:
            task_name: 작업 이름 (예: "ocr.recognize", "audio.transcribe")
            task_data: 작업 데이터
            priority: 우선순위 (높을수록 우선 처리)

        Returns:
            task_id: 작업 ID
        """
        try:
            # 작업 큐에 추가
            task_id = await self.task_queue.enqueue(
                task_type=f"{self.task_type}:{task_name}",
                data=task_data,
                priority=priority,
            )

            # 메시지 발행 (이벤트 로그)
            await self.message_producer.publish_request(
                request_type=f"{self.task_type}:{task_name}",
                request_data={
                    "task_id": task_id,
                    **task_data,
                },
            )

            logger.info(f"Task enqueued: {task_id} ({task_name})")
            return task_id
        except Exception as e:
            logger.error(
                f"Failed to enqueue task {task_name}: {sanitize_error_message(str(e))}",
                exc_info=True,
            )
            raise

    async def process_task(
        self,
        task_name: str,
        handler: Callable,
        timeout: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        작업 큐에서 작업을 가져와 처리

        Args:
            task_name: 작업 이름
            handler: 작업 처리 함수 (async function)
            timeout: 대기 시간 (초)

        Returns:
            처리 결과 또는 None (타임아웃)
        """
        try:
            # 작업 큐에서 가져오기
            task = await self.task_queue.dequeue(
                task_type=f"{self.task_type}:{task_name}",
                timeout=timeout,
            )

            if task is None:
                return None

            task_id = task.get("task_id")
            task_data = task.get("data", {})

            # 작업 시작 이벤트 발행
            await self.message_producer.publish_event(
                f"{self.task_type}:{task_name}.started",
                {
                    "task_id": task_id,
                    "data": task_data,
                },
            )

            try:
                # 작업 처리
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(task_data)
                else:
                    # 동기 함수는 executor에서 실행
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, handler, task_data)

                # 작업 완료 이벤트 발행
                await self.message_producer.publish_event(
                    f"{self.task_type}:{task_name}.completed",
                    {
                        "task_id": task_id,
                        "result": result,
                    },
                )

                logger.info(f"Task completed: {task_id} ({task_name})")
                return {"task_id": task_id, "result": result}
            except Exception as e:
                # 오류 처리
                await self.error_handler.handle_error(
                    request_id=task_id or "unknown",
                    error=e,
                    operation=f"{self.task_type}:{task_name}",
                    context={"task_data": task_data},
                )
                raise
        except Exception as e:
            logger.error(
                f"Failed to process task {task_name}: {sanitize_error_message(str(e))}",
                exc_info=True,
            )
            return None

    async def batch_enqueue(
        self,
        task_name: str,
        tasks_data: List[Dict[str, Any]],
        priority: int = 0,
    ) -> List[str]:
        """
        여러 작업을 일괄 추가

        Args:
            task_name: 작업 이름
            tasks_data: 작업 데이터 리스트
            priority: 우선순위

        Returns:
            task_id 리스트
        """
        task_ids = []
        for task_data in tasks_data:
            task_id = await self.enqueue_task(task_name, task_data, priority=priority)
            task_ids.append(task_id)
        return task_ids

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """작업 상태 조회"""
        return await self.task_queue.get_task_status(task_id)


class BatchProcessor:
    """
    배치 처리기

    여러 작업을 병렬로 처리 (작업 큐 사용)
    """

    def __init__(self, task_type: str = "llm.tasks", max_concurrent: int = 10):
        """
        Args:
            task_type: 작업 타입
            max_concurrent: 최대 동시 처리 수
        """
        self.task_processor = TaskProcessor(task_type)
        self.max_concurrent = max_concurrent
        self.concurrency_controller = None
        try:
            from .messaging import ConcurrencyController

            self.concurrency_controller = ConcurrencyController()
        except Exception as e:
            logger.debug(f"ConcurrencyController not available (continuing without it): {e}")

    async def process_batch(
        self,
        task_name: str,
        tasks_data: List[Dict[str, Any]],
        handler: Callable,
        priority: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        배치 작업 처리

        Args:
            task_name: 작업 이름
            tasks_data: 작업 데이터 리스트
            handler: 작업 처리 함수
            priority: 우선순위

        Returns:
            처리 결과 리스트
        """
        # 1. 모든 작업을 큐에 추가
        task_ids = await self.task_processor.batch_enqueue(task_name, tasks_data, priority=priority)

        # 2. 병렬로 처리
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_one(task_id: str, task_data: Dict[str, Any]):
            """단일 작업 처리"""
            async with semaphore:
                # 동시성 제어 (분산 또는 인메모리)
                if self.concurrency_controller:
                    async with await self.concurrency_controller.with_concurrency_control(
                        f"{self.task_processor.task_type}:{task_name}",
                        max_concurrent=self.max_concurrent,
                    ):
                        if asyncio.iscoroutinefunction(handler):
                            return await handler(task_data)
                        else:
                            loop = asyncio.get_event_loop()
                            return await loop.run_in_executor(None, handler, task_data)
                else:
                    if asyncio.iscoroutinefunction(handler):
                        return await handler(task_data)
                    else:
                        loop = asyncio.get_event_loop()
                        return await loop.run_in_executor(None, handler, task_data)

        # 3. 병렬 실행
        tasks = [
            process_one(task_id, task_data) for task_id, task_data in zip(task_ids, tasks_data)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 4. 결과 정리
        final_results: List[Any] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task_id = task_ids[i]
                await self.task_processor.error_handler.handle_error(
                    request_id=task_id,
                    error=result,
                    operation=f"{self.task_processor.task_type}:{task_name}",
                )
                final_results.append({"error": str(result)})
            else:
                final_results.append(result)

        return final_results

    async def process_items(
        self,
        items: List[Any],
        handler: Callable,
        max_concurrent: Optional[int] = None,
    ) -> List[Any]:
        """
        배치 처리 (items를 직접 받아서 처리)

        Args:
            items: 처리할 항목 리스트 (Path, str, Dict 등)
            handler: 작업 처리 함수 (동기 또는 비동기)
            max_concurrent: 최대 동시 처리 수 (None이면 self.max_concurrent 사용)

        Returns:
            처리 결과 리스트
        """
        max_concurrent = max_concurrent or self.max_concurrent

        # 동시성 제어 (분산 또는 인메모리)
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_one(item: Any):
            """단일 항목 처리"""
            async with semaphore:
                if self.concurrency_controller:
                    async with await self.concurrency_controller.with_concurrency_control(
                        f"{self.task_processor.task_type}:batch",
                        max_concurrent=max_concurrent,
                    ):
                        if asyncio.iscoroutinefunction(handler):
                            return await handler(item)
                        else:
                            loop = asyncio.get_event_loop()
                            return await loop.run_in_executor(None, handler, item)
                else:
                    if asyncio.iscoroutinefunction(handler):
                        return await handler(item)
                    else:
                        loop = asyncio.get_event_loop()
                        return await loop.run_in_executor(None, handler, item)

        # 병렬 실행
        tasks = [process_one(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 정리
        final_results: List[Any] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                await self.task_processor.error_handler.handle_error(
                    request_id=f"batch_item_{i}",
                    error=result,
                    operation=f"{self.task_processor.task_type}:batch",
                )
                final_results.append(None)  # 에러 시 None 반환
            else:
                final_results.append(result)

        return final_results
