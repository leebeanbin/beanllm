"""
Validation Decorators - 입력 검증 공통 기능
책임: 입력 검증 패턴 재사용 (DRY 원칙)
"""

import functools
import inspect
from typing import AsyncIterator, Callable, Dict, List, TypeVar

from .validation_utils import _get_bound_args, _validate_parameters

T = TypeVar("T")


def validate_input(
    required_params: List[str] = None,
    param_types: Dict[str, type] = None,
    param_ranges: Dict[str, tuple] = None,
):
    """
    입력 검증 데코레이터

    책임:
    - 필수 파라미터 검증
    - 타입 검증
    - 범위 검증

    Args:
        required_params: 필수 파라미터 리스트
        param_types: 파라미터 타입 딕셔너리 {"param": type}
        param_ranges: 파라미터 범위 딕셔너리 {"param": (min, max)}

    Example:
        @validate_input(
            required_params=["messages", "model"],
            param_types={"temperature": float},
            param_ranges={"temperature": (0, 2), "max_tokens": (1, None)}
        )
        async def handle_chat(self, messages, model, temperature=None, ...):
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # async generator 함수인지 확인
        if inspect.isasyncgenfunction(func):
            # async generator 함수인 경우
            @functools.wraps(func)
            async def async_gen_wrapper(*args, **kwargs):
                # 공통 검증 로직 사용 (DRY)
                bound_args = _get_bound_args(func, *args, **kwargs)
                _validate_parameters(bound_args, required_params, param_types, param_ranges)
                
                # async generator를 직접 반환 (await 사용 안 함)
                async for item in func(*args, **kwargs):
                    yield item

            return async_gen_wrapper
        # 동기 generator 함수인지 확인
        elif inspect.isgeneratorfunction(func):
            # 동기 generator 함수인 경우
            @functools.wraps(func)
            def sync_gen_wrapper(*args, **kwargs):
                # 공통 검증 로직 사용 (DRY)
                bound_args = _get_bound_args(func, *args, **kwargs)
                _validate_parameters(bound_args, required_params, param_types, param_ranges)
                
                # 동기 generator를 직접 반환
                for item in func(*args, **kwargs):
                    yield item

            return sync_gen_wrapper
        else:
            # 일반 async 함수인 경우
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # 공통 검증 로직 사용 (DRY)
                bound_args = _get_bound_args(func, *args, **kwargs)
                _validate_parameters(bound_args, required_params, param_types, param_ranges)
                
                return await func(*args, **kwargs)

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # 공통 검증 로직 사용 (DRY)
                bound_args = _get_bound_args(func, *args, **kwargs)
                _validate_parameters(bound_args, required_params, param_types, param_ranges)
                
                return func(*args, **kwargs)

            # async 함수인지 확인
            if hasattr(func, "__code__") and "coroutine" in str(type(func)):
                return async_wrapper
            return sync_wrapper

    return decorator
