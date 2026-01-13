"""
Kafka 기반 이벤트 스트리밍

이벤트 발행 및 구독을 Kafka로 구현
기존 최적화 패턴 참고: 에러 처리, 로깅, fallback
"""

import asyncio
import json
import time
import uuid
from typing import Any, AsyncIterator, Callable, Dict

from ..interfaces import EventProducerInterface, EventConsumerInterface
from ..utils import check_kafka_health, sanitize_error_message
from .client import get_kafka_client

try:
    from beanllm.utils.logging import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


class KafkaEventProducer(EventProducerInterface):
    """
    Kafka 기반 이벤트 발행자

    모든 이벤트를 Kafka Topic에 발행하여 영구 저장
    """

    def __init__(self, kafka_client=None):
        """
        Args:
            kafka_client: (KafkaProducer, KafkaConsumer) 튜플 (None이면 자동 생성)
        """
        if kafka_client is None:
            producer, _ = get_kafka_client()
        else:
            producer, _ = kafka_client
        self.producer = producer

    async def publish(self, topic: str, event: Dict[str, Any]):
        """이벤트 발행"""
        try:
            # Kafka 연결 확인
            if not await check_kafka_health((self.producer, None)):
                logger.warning(f"Kafka not connected, event publish skipped for topic: {topic}")
                return  # 연결 실패 시 발행 건너뛰기

            # 이벤트에 메타데이터 추가
            event_with_meta = {
                "event_id": str(uuid.uuid4()),
                "topic": topic,
                "timestamp": time.time(),
                "data": event,
            }

            event_json = json.dumps(event_with_meta)

            # 비동기적으로 발행
            await asyncio.wait_for(
                asyncio.to_thread(
                    self.producer.send,
                    topic,
                    value=event_json,
                    key=event.get("request_id", "").encode()
                    if event.get("request_id")
                    else None,
                ),
                timeout=5.0,
            )

            # 즉시 플러시 (선택적)
            self.producer.flush()
        except asyncio.TimeoutError:
            logger.warning(f"Kafka event publish timeout for topic: {topic}")
        except Exception as e:
            logger.error(
                f"Kafka event publish error for topic: {topic}: {sanitize_error_message(str(e))}"
            )


class KafkaEventConsumer(EventConsumerInterface):
    """
    Kafka 기반 이벤트 구독자

    Kafka Topic에서 이벤트를 구독
    """

    def __init__(self, kafka_client=None):
        """
        Args:
            kafka_client: (KafkaProducer, KafkaConsumer) 튜플 (None이면 자동 생성)
        """
        if kafka_client is None:
            _, consumer = get_kafka_client()
        else:
            _, consumer = kafka_client
        self.consumer = consumer

    async def subscribe(self, topic: str, handler: Any) -> AsyncIterator[Dict[str, Any]]:
        """이벤트 구독"""
        # Topic 구독
        self.consumer.subscribe([topic])

        # 이벤트 스트림
        import asyncio

        while True:
            # Kafka Consumer는 동기적이므로 비동기로 래핑
            message_pack = await asyncio.to_thread(
                self.consumer.poll, timeout_ms=1000
            )

            if message_pack:
                for topic_partition, messages in message_pack.items():
                    for message in messages:
                        try:
                            event = json.loads(message.value)
                            # 핸들러 호출
                            if asyncio.iscoroutinefunction(handler):
                                await handler(event)
                            else:
                                handler(event)
                            yield event
                        except Exception as e:
                            # 오류 처리
                            import logging
                            logging.error(f"Error processing event: {e}", exc_info=True)

