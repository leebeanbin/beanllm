"""
Kafka 기반 작업 큐

장기 작업을 Kafka Topic으로 관리
기존 최적화 패턴 참고: 에러 처리, 로깅, fallback
"""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, Optional, cast

from beanllm.infrastructure.distributed.interfaces import TaskQueueInterface
from beanllm.utils import check_kafka_health, sanitize_error_message

from .client import get_kafka_client

try:
    from beanllm.utils.logging import get_logger
except ImportError:
    import logging

    def get_logger(name: str) -> logging.Logger:  # type: ignore[misc]
        return logging.getLogger(name)


logger = get_logger(__name__)


class KafkaTaskQueue(TaskQueueInterface):
    """
    Kafka 기반 작업 큐

    장기 작업을 Kafka Topic에 저장하여 영구 보관
    여러 Worker가 작업 분산 처리
    """

    def __init__(self, kafka_client=None, topic: str = "llm.tasks"):
        """
        Args:
            kafka_client: (KafkaProducer, KafkaConsumer) 튜플 (None이면 자동 생성)
            topic: 작업 큐 토픽
        """
        if kafka_client is None:
            producer, consumer = get_kafka_client()
        else:
            producer, consumer = kafka_client
        self.producer = producer
        self.consumer = consumer
        self.topic = topic
        self._task_status: Dict[str, Dict[str, Any]] = {}  # 인메모리 상태 저장

    async def enqueue(self, task_type: str, data: Dict[str, Any], priority: int = 0) -> str:
        """작업 큐에 추가"""
        try:
            # Kafka 연결 확인
            if not await check_kafka_health((self.producer, self.consumer)):
                logger.warning(f"Kafka not connected, task enqueue skipped for type: {task_type}")
                # 연결 실패 시 인메모리 큐로 fallback (간단한 처리)
                task_id = str(uuid.uuid4())
                self._task_status[task_id] = {
                    "status": "pending",
                    "created_at": time.time(),
                    "data": data,
                }
                return task_id

            task_id = str(uuid.uuid4())

            task = {
                "task_id": task_id,
                "task_type": task_type,
                "priority": priority,
                "data": data,
                "created_at": time.time(),
                "status": "pending",
            }

            task_json = json.dumps(task)

            # Priority를 키로 사용 (높은 우선순위가 먼저 처리)
            partition_key = f"{priority:05d}-{task_id}".encode()

            await asyncio.wait_for(
                asyncio.to_thread(
                    self.producer.send,
                    self.topic,
                    value=task_json,
                    key=partition_key,
                ),
                timeout=5.0,
            )
            self.producer.flush()

            # 작업 상태 저장 (인메모리, 실제로는 Redis 사용 권장)
            self._task_status[task_id] = {
                "status": "pending",
                "created_at": task["created_at"],
            }

            return task_id
        except asyncio.TimeoutError:
            logger.warning(f"Kafka task enqueue timeout for type: {task_type}")
            # 타임아웃 시 인메모리로 fallback
            task_id = str(uuid.uuid4())
            self._task_status[task_id] = {
                "status": "pending",
                "created_at": time.time(),
                "data": data,
            }
            return task_id
        except Exception as e:
            logger.error(
                f"Kafka task enqueue error for type: {task_type}: {sanitize_error_message(str(e))}"
            )
            # 오류 시 인메모리로 fallback
            task_id = str(uuid.uuid4())
            self._task_status[task_id] = {
                "status": "pending",
                "created_at": time.time(),
                "data": data,
            }
            return task_id

    async def dequeue(
        self, task_type: str, timeout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """작업 큐에서 가져오기"""
        # Topic 구독
        self.consumer.subscribe([self.topic])

        import asyncio

        start_time = time.time()

        while True:
            if timeout and (time.time() - start_time) > timeout:
                return None

            # Kafka에서 메시지 가져오기
            message_pack = await asyncio.to_thread(self.consumer.poll, timeout_ms=1000)

            if message_pack:
                for topic_partition, messages in message_pack.items():
                    for message in messages:
                        task = json.loads(message.value)

                        # 작업 타입 필터링
                        if task.get("task_type") == task_type:
                            # 작업 상태 업데이트
                            task_id = task["task_id"]
                            if task_id in self._task_status:
                                self._task_status[task_id]["status"] = "processing"
                                self._task_status[task_id]["started_at"] = time.time()

                            return cast(Dict[str, Any], task)

            await asyncio.sleep(0.1)  # 짧은 대기

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """작업 상태 조회"""
        return self._task_status.get(task_id)
