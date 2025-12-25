"""
Logger Decorators - 로깅 공통 기능
책임: 로깅 패턴 재사용 (DRY 원칙)
"""

import functools
import inspect
import time
from typing import Callable, TypeVar

from ..utils.logger import get_logger

T = TypeVar("T")

logger = get_logger(__name__)


def log_execution(func: Callable[..., T]) -> Callable[..., T]:
    """
    함수 실행 로깅 데코레이터

    책임:
    - 함수 시작/종료 로깅
    - 실행 시간 측정
    - 파라미터 로깅 (선택적)

    Example:
        @log_execution
        async def my_function(arg1, arg2):
            ...
    """

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.debug(f"Executing {func_name} with args={args}, kwargs={kwargs}")
        start_time = time.time()

        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"{func_name} completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func_name} failed after {elapsed:.2f}s: {e}")
            raise

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.debug(f"Executing {func_name} with args={args}, kwargs={kwargs}")
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"{func_name} completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func_name} failed after {elapsed:.2f}s: {e}")
            raise

    # async 함수인지 확인
    if hasattr(func, "__code__") and "coroutine" in str(type(func)):
        return async_wrapper
    return sync_wrapper


def log_service_call(func: Callable[..., T]) -> Callable[..., T]:
    """
    서비스 호출 로깅 데코레이터

    책임:
    - 서비스 메서드 호출 로깅
    - 요청/응답 로깅 (선택적)
    - async generator 함수 지원

    Example:
        @log_service_call
        async def chat(self, request: ChatRequest) -> ChatResponse:
            ...

        @log_service_call
        async def stream_chat(self, request: ChatRequest) -> AsyncIterator[str]:
            ...
    """
    # async generator 함수인지 확인
    if inspect.isasyncgenfunction(func):
        # async generator 함수인 경우
        @functools.wraps(func)
        async def async_gen_wrapper(self, *args, **kwargs):
            service_name = self.__class__.__name__
            method_name = func.__name__
            logger.debug(f"Service call: {service_name}.{method_name}")

            try:
                # async generator를 직접 반환 (await 사용 안 함)
                async for item in func(self, *args, **kwargs):
                    yield item
                logger.info(f"Service call succeeded: {service_name}.{method_name}")
            except Exception as e:
                logger.error(f"Service call failed: {service_name}.{method_name} - {e}")
                raise

        return async_gen_wrapper
    else:
        # 일반 async 함수인 경우
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            service_name = self.__class__.__name__
            method_name = func.__name__
            logger.debug(f"Service call: {service_name}.{method_name}")

            try:
                result = await func(self, *args, **kwargs)
                logger.info(f"Service call succeeded: {service_name}.{method_name}")
                return result
            except Exception as e:
                logger.error(f"Service call failed: {service_name}.{method_name} - {e}")
                raise

        return wrapper


def log_handler_call(func: Callable[..., T]) -> Callable[..., T]:
    """
    Handler 호출 로깅 데코레이터

    책임:
    - Handler 메서드 호출 로깅
    - 요청 파라미터 로깅
    - async generator 함수 지원
    - 동기 generator 함수 지원

    Example:
        @log_handler_call
        async def handle_chat(self, messages, model, ...):
            ...

        @log_handler_call
        async def handle_stream_chat(self, messages, model, ...) -> AsyncIterator[str]:
            ...

        @log_handler_call
        def handle_stream(self, ...) -> Iterator[tuple]:
            ...
    """
    # async generator 함수인지 확인
    if inspect.isasyncgenfunction(func):
        # async generator 함수인 경우
        @functools.wraps(func)
        async def async_gen_wrapper(self, *args, **kwargs):
            handler_name = self.__class__.__name__
            method_name = func.__name__

            # 민감한 정보 제외하고 로깅
            safe_kwargs = {k: v for k, v in kwargs.items() if k not in ["api_key", "password"]}
            logger.info(f"Handler call: {handler_name}.{method_name} with {safe_kwargs}")

            try:
                # async generator를 직접 반환 (await 사용 안 함)
                async for item in func(self, *args, **kwargs):
                    yield item
                logger.info(f"Handler call succeeded: {handler_name}.{method_name}")
            except Exception as e:
                logger.error(f"Handler call failed: {handler_name}.{method_name} - {e}")
                raise

        return async_gen_wrapper
    elif inspect.isgeneratorfunction(func):
        # 동기 generator 함수인 경우
        @functools.wraps(func)
        def sync_gen_wrapper(self, *args, **kwargs):
            handler_name = self.__class__.__name__
            method_name = func.__name__

            # 민감한 정보 제외하고 로깅
            safe_kwargs = {k: v for k, v in kwargs.items() if k not in ["api_key", "password"]}
            logger.info(f"Handler call: {handler_name}.{method_name} with {safe_kwargs}")

            try:
                # 동기 generator를 직접 반환
                for item in func(self, *args, **kwargs):
                    yield item
                logger.info(f"Handler call succeeded: {handler_name}.{method_name}")
            except Exception as e:
                logger.error(f"Handler call failed: {handler_name}.{method_name} - {e}")
                raise

        return sync_gen_wrapper
    elif inspect.iscoroutinefunction(func):
        # 일반 async 함수인 경우
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            handler_name = self.__class__.__name__
            method_name = func.__name__

            # 민감한 정보 제외하고 로깅
            safe_kwargs = {k: v for k, v in kwargs.items() if k not in ["api_key", "password"]}
            logger.info(f"Handler call: {handler_name}.{method_name} with {safe_kwargs}")

            try:
                result = await func(self, *args, **kwargs)
                logger.info(f"Handler call succeeded: {handler_name}.{method_name}")
                return result
            except Exception as e:
                logger.error(f"Handler call failed: {handler_name}.{method_name} - {e}")
                raise

        return wrapper
    else:
        # 동기 함수인 경우
        @functools.wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            handler_name = self.__class__.__name__
            method_name = func.__name__

            # 민감한 정보 제외하고 로깅
            safe_kwargs = {k: v for k, v in kwargs.items() if k not in ["api_key", "password"]}
            logger.info(f"Handler call: {handler_name}.{method_name} with {safe_kwargs}")

            try:
                result = func(self, *args, **kwargs)
                logger.info(f"Handler call succeeded: {handler_name}.{method_name}")
                return result
            except Exception as e:
                logger.error(f"Handler call failed: {handler_name}.{method_name} - {e}")
                raise

        return sync_wrapper
