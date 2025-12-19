"""
Retry Logic
재시도 로직 (독립적)
"""

import asyncio
import time
from functools import wraps
from typing import Callable, Tuple, Type

from .logger import get_logger

logger = get_logger(__name__)


def retry(
    max_attempts: int = 3,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    재시도 데코레이터

    Args:
        max_attempts: 최대 시도 횟수
        backoff: 백오프 배율 (지수 백오프)
        exceptions: 재시도할 예외 타입들

    Example:
        @retry(max_attempts=3, backoff=2.0, exceptions=(TimeoutError, ConnectionError))
        async def fetch_data():
            ...
    """

    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        logger.error(f"Failed after {max_attempts} attempts: {func.__name__}")
                        raise

                    wait_time = backoff ** (attempt - 1)
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)

            raise last_exception

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        logger.error(f"Failed after {max_attempts} attempts: {func.__name__}")
                        raise

                    wait_time = backoff ** (attempt - 1)
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)

            raise last_exception

        # async 함수인지 확인
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
