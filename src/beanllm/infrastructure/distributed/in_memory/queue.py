"""
인메모리 작업 큐 (asyncio.Queue 래핑)
"""

import asyncio
import time
import uuid
from typing import Any, Dict, Optional, cast

from beanllm.infrastructure.distributed.interfaces import TaskQueueInterface


class InMemoryTaskQueue(TaskQueueInterface):
    """
    인메모리 작업 큐

    asyncio.Queue를 사용하여 인메모리 작업 큐 구현
    """

    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = {}
        self._task_status: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    def _get_queue(self, task_type: str) -> asyncio.Queue:
        """작업 타입에 해당하는 큐 반환 (없으면 생성)"""
        if task_type not in self._queues:
            self._queues[task_type] = asyncio.Queue()
        return self._queues[task_type]

    async def enqueue(self, task_type: str, data: Dict[str, Any], priority: int = 0) -> str:
        """작업 큐에 추가"""
        task_id = str(uuid.uuid4())

        task = {
            "task_id": task_id,
            "task_type": task_type,
            "priority": priority,
            "data": data,
            "created_at": time.time(),
            "status": "pending",
        }

        # 우선순위가 높을수록 먼저 처리 (음수로 변환하여 정렬)
        queue = self._get_queue(task_type)
        await queue.put((-priority, task))

        # 작업 상태 저장
        async with self._lock:
            self._task_status[task_id] = {
                "status": "pending",
                "created_at": task["created_at"],
            }

        return task_id

    async def dequeue(
        self, task_type: str, timeout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """작업 큐에서 가져오기"""
        queue = self._get_queue(task_type)

        try:
            if timeout:
                priority, task = await asyncio.wait_for(queue.get(), timeout=timeout)
            else:
                priority, task = await queue.get()

            # 작업 상태 업데이트
            async with self._lock:
                if task["task_id"] in self._task_status:
                    self._task_status[task["task_id"]]["status"] = "processing"
                    self._task_status[task["task_id"]]["started_at"] = time.time()

            return cast(Optional[Dict[str, Any]], task)
        except asyncio.TimeoutError:
            return None

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """작업 상태 조회"""
        async with self._lock:
            return cast(Optional[Dict[str, Any]], self._task_status.get(task_id))
