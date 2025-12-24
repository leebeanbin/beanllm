"""
Error Handler Decorators - 에러 처리 공통 기능
책임: 에러 처리 패턴 재사용 (DRY 원칙)
"""

import functools
import inspect
from typing import Any, Callable, TypeVar

from ..utils.logger import get_logger

T = TypeVar("T")

logger = get_logger(__name__)


def handle_errors(
    error_message: str = None,
    reraise: bool = True,
    default_return: Any = None,
):
    """
    에러 처리 데코레이터

    책임:
    - try-catch 패턴 재사용
    - 에러 로깅
    - 에러 변환 (선택적)

    Args:
        error_message: 커스텀 에러 메시지
        reraise: 에러를 다시 발생시킬지 여부
        default_return: 에러 발생 시 반환할 기본값

    Example:
        @handle_errors(error_message="Chat failed", reraise=True)
        async def handle_chat(self, ...):
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # async generator 함수인지 확인
        if inspect.isasyncgenfunction(func):
            # async generator 함수인 경우
            @functools.wraps(func)
            async def async_gen_wrapper(*args, **kwargs):
                func_name = func.__name__
                try:
                    # async generator를 직접 반환 (await 사용 안 함)
                    async for item in func(*args, **kwargs):
                        yield item
                except ValueError as e:
                    error_msg = error_message or f"{func_name} validation failed"
                    logger.error(f"{error_msg}: {e}")
                    if reraise:
                        raise
                    if default_return is not None:
                        yield default_return
                except Exception as e:
                    error_msg = error_message or f"{func_name} failed"
                    logger.error(f"{error_msg}: {e}")
                    if reraise:
                        raise
                    if default_return is not None:
                        yield default_return

            return async_gen_wrapper
        # 동기 generator 함수인지 확인
        elif inspect.isgeneratorfunction(func):
            # 동기 generator 함수인 경우
            @functools.wraps(func)
            def sync_gen_wrapper(*args, **kwargs):
                func_name = func.__name__
                try:
                    # 동기 generator를 직접 반환
                    for item in func(*args, **kwargs):
                        yield item
                except ValueError as e:
                    error_msg = error_message or f"{func_name} validation failed"
                    logger.error(f"{error_msg}: {e}")
                    if reraise:
                        raise
                    if default_return is not None:
                        yield default_return
                except Exception as e:
                    error_msg = error_message or f"{func_name} failed"
                    logger.error(f"{error_msg}: {e}")
                    if reraise:
                        raise
                    if default_return is not None:
                        yield default_return

            return sync_gen_wrapper
        else:
            # 일반 async 함수인 경우
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                func_name = func.__name__
                try:
                    return await func(*args, **kwargs)
                except ValueError as e:
                    error_msg = error_message or f"{func_name} validation failed"
                    logger.error(f"{error_msg}: {e}")
                    if reraise:
                        raise
                    return default_return
                except Exception as e:
                    error_msg = error_message or f"{func_name} failed"
                    logger.error(f"{error_msg}: {e}")
                    if reraise:
                        raise
                    return default_return

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                func_name = func.__name__
                try:
                    return func(*args, **kwargs)
                except ValueError as e:
                    error_msg = error_message or f"{func_name} validation failed"
                    logger.error(f"{error_msg}: {e}")
                    if reraise:
                        raise
                    return default_return
                except Exception as e:
                    error_msg = error_message or f"{func_name} failed"
                    logger.error(f"{error_msg}: {e}")
                    if reraise:
                        raise
                    return default_return

            # async 함수인지 확인
            if hasattr(func, "__code__") and "coroutine" in str(type(func)):
                return async_wrapper
            return sync_wrapper

    return decorator


def log_errors(func: Callable[..., T]) -> Callable[..., T]:
    """
    에러 로깅 데코레이터 (에러만 로깅, 재발생)

    책임:
    - 에러 발생 시 로깅만
    - 에러는 그대로 재발생

    Example:
        @log_errors
        async def my_function(...):
            ...
    """

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        func_name = func.__name__
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{func_name} error: {e}", exc_info=True)
            raise

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        func_name = func.__name__
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{func_name} error: {e}", exc_info=True)
            raise

    # async 함수인지 확인
    if hasattr(func, "__code__") and "coroutine" in str(type(func)):
        return async_wrapper
    return sync_wrapper
