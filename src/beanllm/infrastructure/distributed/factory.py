"""
분산 아키텍처 팩토리 함수

환경변수에 따라 분산/인메모리 구현을 자동 선택합니다.
"""

import os
from typing import Optional, Tuple

from .interfaces import (
    CacheInterface,
    DistributedLockInterface,
    EventConsumerInterface,
    EventProducerInterface,
    RateLimiterInterface,
    TaskQueueInterface,
)

# 환경변수로 분산 모드 활성화 여부 확인
USE_DISTRIBUTED = os.getenv("USE_DISTRIBUTED", "false").lower() == "true"


def get_rate_limiter() -> RateLimiterInterface:
    """
    Rate Limiter 인스턴스 반환

    환경변수 USE_DISTRIBUTED에 따라:
    - true: RedisRateLimiter (분산)
    - false: InMemoryRateLimiter (인메모리)

    Returns:
        RateLimiterInterface 인스턴스
    """
    if USE_DISTRIBUTED:
        from .redis.client import get_redis_client
        from .redis.rate_limiter import RedisRateLimiter

        return RedisRateLimiter(get_redis_client())
    else:
        from .in_memory.rate_limiter import InMemoryRateLimiter

        return InMemoryRateLimiter()


def get_cache(max_size: int = 1000, ttl: Optional[int] = None) -> CacheInterface:
    """
    Cache 인스턴스 반환

    Args:
        max_size: 최대 캐시 크기 (인메모리 모드에서만 사용)
        ttl: 기본 TTL (초)

    Returns:
        CacheInterface 인스턴스
    """
    if USE_DISTRIBUTED:
        from .redis.cache import RedisCache
        from .redis.client import get_redis_client

        return RedisCache(get_redis_client(), ttl=ttl)
    else:
        from .in_memory.cache import InMemoryCache

        return InMemoryCache(max_size=max_size, ttl=ttl)


def get_task_queue(topic: str) -> TaskQueueInterface:
    """
    작업 큐 인스턴스 반환

    Args:
        topic: 작업 큐 토픽 (예: "ocr.tasks", "embedding.tasks")

    Returns:
        TaskQueueInterface 인스턴스
    """
    if USE_DISTRIBUTED:
        from .kafka.client import get_kafka_client
        from .kafka.queue import KafkaTaskQueue

        return KafkaTaskQueue(get_kafka_client(), topic)
    else:
        from .in_memory.queue import InMemoryTaskQueue

        return InMemoryTaskQueue()


def get_event_bus() -> Tuple[EventProducerInterface, EventConsumerInterface]:
    """
    이벤트 버스 인스턴스 반환

    Returns:
        (EventProducer, EventConsumer) 튜플
    """
    if USE_DISTRIBUTED:
        from .kafka.client import get_kafka_client
        from .kafka.events import KafkaEventConsumer, KafkaEventProducer

        kafka_client = get_kafka_client()
        return KafkaEventProducer(kafka_client), KafkaEventConsumer(kafka_client)
    else:
        from .in_memory.events import InMemoryEventBus

        bus = InMemoryEventBus()
        return bus, bus


def get_distributed_lock() -> DistributedLockInterface:
    """
    분산 락 인스턴스 반환

    Returns:
        DistributedLockInterface 인스턴스
    """
    if USE_DISTRIBUTED:
        from .redis.client import get_redis_client
        from .redis.lock import RedisLock

        return RedisLock(get_redis_client())
    else:
        from .in_memory.lock import InMemoryLock

        return InMemoryLock()
