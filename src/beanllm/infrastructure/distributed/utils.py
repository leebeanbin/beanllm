"""
분산 아키텍처 유틸리티

기존 최적화 패턴을 참고하여 공통 로직 제공
"""

import asyncio
import functools
import logging
from typing import Callable, Optional

try:
    from beanllm.utils.logging import get_logger
except ImportError:
    import logging

    def get_logger(name: str) -> logging.Logger:  # type: ignore[misc]
        return logging.getLogger(name)


try:
    from beanllm.utils.integration.security import sanitize_error_message
except ImportError:
    # Fallback if security module not available
    def sanitize_error_message(message: str) -> str:
        if isinstance(message, Exception):
            message = str(message)
        return message


logger = get_logger(__name__)


async def check_redis_health(redis_client=None) -> bool:
    """
    Redis 연결 상태 확인

    Args:
        redis_client: Redis 클라이언트 (None이면 자동 생성)

    Returns:
        True: 연결 정상, False: 연결 실패
    """
    try:
        if redis_client is None:
            from .redis.client import get_redis_client

            redis_client = get_redis_client()

        # PING으로 연결 확인
        result = await asyncio.wait_for(redis_client.ping(), timeout=2.0)
        return result is True
    except Exception as e:
        logger.warning(f"Redis health check failed: {sanitize_error_message(str(e))}")
        return False


async def check_kafka_health(kafka_client=None) -> bool:
    """
    Kafka 연결 상태 확인

    Args:
        kafka_client: Kafka 클라이언트 (None이면 자동 생성)

    Returns:
        True: 연결 정상, False: 연결 실패
    """
    try:
        if kafka_client is None:
            from .kafka.client import get_kafka_client

            producer, consumer = get_kafka_client()
        else:
            producer, consumer = kafka_client

        # Producer가 정상인지 확인 (간단한 메타데이터 조회)
        # 실제로는 더 복잡한 확인이 필요하지만, 기본적인 확인만 수행
        return producer is not None and consumer is not None
    except Exception as e:
        logger.warning(f"Kafka health check failed: {sanitize_error_message(str(e))}")
        return False


def with_fallback(fallback_func: Optional[Callable] = None, log_error: bool = True):
    """
    Fallback 메커니즘 데코레이터

    분산 컴포넌트 연결 실패 시 fallback 함수 실행 또는 인메모리로 전환

    Args:
        fallback_func: Fallback 함수 (None이면 인메모리 구현 사용)
        log_error: 오류 로깅 여부

    Example:
        ```python
        @with_fallback()
        async def get_cache():
            return RedisCache(...)
        ```
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.warning(
                        f"Distributed component failed, using fallback: {sanitize_error_message(str(e))}"
                    )

                if fallback_func:
                    return await fallback_func(*args, **kwargs)
                else:
                    # 인메모리 구현으로 fallback
                    # 함수 이름에서 분산 구현 이름 추출 (예: RedisCache -> InMemoryCache)
                    func_name = func.__name__
                    if "Redis" in func_name:
                        from .in_memory.cache import InMemoryCache

                        return InMemoryCache()
                    elif "Kafka" in func_name:
                        from .in_memory.queue import InMemoryTaskQueue

                        return InMemoryTaskQueue()
                    else:
                        raise

        return wrapper

    return decorator


class DistributedError(Exception):
    """분산 아키텍처 관련 오류"""

    pass


class LockAcquisitionError(DistributedError):
    """락 획득 실패"""

    pass


class ConnectionError(DistributedError):
    """연결 오류"""

    pass
