"""
Provider Error Handler Decorator - Provider 전용 에러 핸들링
책임: Provider 에러 처리 패턴 재사용 (DRY 원칙)

모든 Provider의 chat(), stream_chat() 메서드에서 반복되는:
- try-except 패턴
- 에러 로깅
- 에러 메시지 마스킹
- ProviderError 변환

을 자동으로 처리합니다.
"""

import functools
import inspect
from typing import Callable, Optional, TypeVar

from beanllm.utils.exceptions import ProviderError
from beanllm.utils.integration.security import sanitize_error_message
from beanllm.utils.logging import get_logger

T = TypeVar("T")

logger = get_logger(__name__)


def provider_error_handler(
    operation: Optional[str] = None,
    provider_name: Optional[str] = None,
    api_error_types: tuple = (),
    custom_error_message: Optional[str] = None,
):
    """
    Provider 에러 핸들링 데코레이터

    모든 Provider의 chat(), stream_chat() 메서드에서 반복되는 에러 처리 패턴을 자동화합니다.

    기능:
    - 자동 에러 메시지 마스킹 (API 키 보호)
    - 자동 로깅
    - ProviderError 변환
    - async/sync/generator 모두 지원

    Args:
        operation: 작업 이름 (예: "chat", "stream_chat", "list_models")
                   None이면 함수 이름 사용
        provider_name: Provider 이름 (예: "OpenAI", "Claude")
                       None이면 self.name 또는 클래스 이름 사용
        api_error_types: API 에러 타입 튜플 (예: (APITimeoutError, APIError))
        custom_error_message: 커스텀 에러 메시지

    Example:
        ```python
        @provider_error_handler(operation="chat", api_error_types=(APITimeoutError, APIError))
        async def chat(self, messages, model, ...):
            # API 호출만 작성 (에러 처리 불필요)
            response = await self.client.chat.completions.create(...)
            return LLMResponse(...)
        ```

    Returns:
        Decorator 함수
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        func_name = func.__name__
        op_name = operation or func_name

        # async generator 함수인지 확인
        if inspect.isasyncgenfunction(func):

            @functools.wraps(func)
            async def async_gen_wrapper(self, *args, **kwargs):
                provider = provider_name or getattr(self, "name", self.__class__.__name__)

                try:
                    async for item in func(self, *args, **kwargs):
                        yield item
                except api_error_types as e:
                    error_str = sanitize_error_message(e)
                    error_msg = custom_error_message or f"{provider} {op_name} API error"
                    logger.error(f"{provider} {op_name} error: {error_str}")
                    raise ProviderError(f"{error_msg}: {error_str}") from e
                except Exception as e:
                    error_str = sanitize_error_message(e)
                    error_msg = custom_error_message or f"{provider} {op_name} failed"
                    logger.error(f"{provider} {op_name} error: {error_str}")
                    raise ProviderError(f"{error_msg}: {error_str}") from e

            return async_gen_wrapper

        # 동기 generator 함수인지 확인
        elif inspect.isgeneratorfunction(func):

            @functools.wraps(func)
            def sync_gen_wrapper(self, *args, **kwargs):
                provider = provider_name or getattr(self, "name", self.__class__.__name__)

                try:
                    for item in func(self, *args, **kwargs):
                        yield item
                except api_error_types as e:
                    error_str = sanitize_error_message(e)
                    error_msg = custom_error_message or f"{provider} {op_name} API error"
                    logger.error(f"{provider} {op_name} error: {error_str}")
                    raise ProviderError(f"{error_msg}: {error_str}") from e
                except Exception as e:
                    error_str = sanitize_error_message(e)
                    error_msg = custom_error_message or f"{provider} {op_name} failed"
                    logger.error(f"{provider} {op_name} error: {error_str}")
                    raise ProviderError(f"{error_msg}: {error_str}") from e

            return sync_gen_wrapper

        # 일반 async 함수인 경우
        elif inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(self, *args, **kwargs):
                provider = provider_name or getattr(self, "name", self.__class__.__name__)

                try:
                    return await func(self, *args, **kwargs)
                except api_error_types as e:
                    error_str = sanitize_error_message(e)
                    error_msg = custom_error_message or f"{provider} {op_name} API error"
                    logger.error(f"{provider} {op_name} error: {error_str}")
                    raise ProviderError(f"{error_msg}: {error_str}") from e
                except Exception as e:
                    error_str = sanitize_error_message(e)
                    error_msg = custom_error_message or f"{provider} {op_name} failed"
                    logger.error(f"{provider} {op_name} error: {error_str}")
                    raise ProviderError(f"{error_msg}: {error_str}") from e

            return async_wrapper

        # 동기 함수인 경우
        else:

            @functools.wraps(func)
            def sync_wrapper(self, *args, **kwargs):
                provider = provider_name or getattr(self, "name", self.__class__.__name__)

                try:
                    return func(self, *args, **kwargs)
                except api_error_types as e:
                    error_str = sanitize_error_message(e)
                    error_msg = custom_error_message or f"{provider} {op_name} API error"
                    logger.error(f"{provider} {op_name} error: {error_str}")
                    raise ProviderError(f"{error_msg}: {error_str}") from e
                except Exception as e:
                    error_str = sanitize_error_message(e)
                    error_msg = custom_error_message or f"{provider} {op_name} failed"
                    logger.error(f"{provider} {op_name} error: {error_str}")
                    raise ProviderError(f"{error_msg}: {error_str}") from e

            return sync_wrapper

    return decorator
