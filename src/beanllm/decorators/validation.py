"""
Validation Decorators - 입력 검증 공통 기능
책임: 입력 검증 패턴 재사용 (DRY 원칙)
"""

import functools
import inspect
from typing import Callable, Dict, List, TypeVar

try:
    from .validation_utils import _get_bound_args, _validate_parameters
except ImportError:
    # Fallback: 직접 구현 (validation_utils가 없는 경우)
    from typing import Any

    def _get_bound_args(func: Any, *args: Any, **kwargs: Any):
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return bound_args

    def _validate_parameters(bound_args, required_params=None, param_types=None, param_ranges=None):
        if required_params:
            for param in required_params:
                if param not in bound_args.arguments or bound_args.arguments[param] is None:
                    raise ValueError(f"Required parameter '{param}' is missing or None")
        if param_types:
            for param, expected_type in param_types.items():
                if param in bound_args.arguments:
                    value = bound_args.arguments[param]
                    if value is not None and not isinstance(value, expected_type):
                        raise TypeError(
                            f"Parameter '{param}' must be of type {expected_type.__name__}, got {type(value).__name__}"
                        )
        if param_ranges:
            for param, (min_val, max_val) in param_ranges.items():
                if param in bound_args.arguments:
                    value = bound_args.arguments[param]
                    if value is not None:
                        if min_val is not None and value < min_val:
                            raise ValueError(
                                f"Parameter '{param}' must be >= {min_val}, got {value}"
                            )
                        if max_val is not None and value > max_val:
                            raise ValueError(
                                f"Parameter '{param}' must be <= {max_val}, got {value}"
                            )


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
