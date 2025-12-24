"""
Validation Utils - 검증 공통 로직 (DRY 원칙)
책임: 검증 로직 중복 제거
"""

import inspect
from typing import Any, Dict, List


def _get_bound_args(func: Any, *args: Any, **kwargs: Any) -> inspect.BoundArguments:
    """
    함수 시그니처에서 파라미터 추출 (공통 로직)

    Args:
        func: 함수
        *args: 위치 인자
        **kwargs: 키워드 인자

    Returns:
        BoundArguments: 바인딩된 인자
    """
    sig = inspect.signature(func)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    return bound_args


def _validate_parameters(
    bound_args: inspect.BoundArguments,
    required_params: List[str] = None,
    param_types: Dict[str, type] = None,
    param_ranges: Dict[str, tuple] = None,
) -> None:
    """
    파라미터 검증 공통 로직 (DRY)

    Args:
        bound_args: 바인딩된 인자
        required_params: 필수 파라미터 리스트
        param_types: 파라미터 타입 딕셔너리 {"param": type}
        param_ranges: 파라미터 범위 딕셔너리 {"param": (min, max)}

    Raises:
        ValueError: 필수 파라미터 누락 또는 범위 위반
        TypeError: 타입 불일치
    """
    # 필수 파라미터 검증
    if required_params:
        for param in required_params:
            if param not in bound_args.arguments or bound_args.arguments[param] is None:
                raise ValueError(f"Required parameter '{param}' is missing or None")

    # 타입 검증
    if param_types:
        for param, expected_type in param_types.items():
            if param in bound_args.arguments:
                value = bound_args.arguments[param]
                if value is not None:
                    # 튜플 타입 지원 (여러 타입 허용)
                    if isinstance(expected_type, tuple):
                        if not isinstance(value, expected_type):
                            type_names = ", ".join(t.__name__ for t in expected_type)
                            raise TypeError(
                                f"Parameter '{param}' must be one of types ({type_names}), "
                                f"got {type(value).__name__}"
                            )
                    elif not isinstance(value, expected_type):
                        raise TypeError(
                            f"Parameter '{param}' must be of type {expected_type.__name__}, "
                            f"got {type(value).__name__}"
                        )

    # 범위 검증
    if param_ranges:
        for param, (min_val, max_val) in param_ranges.items():
            if param in bound_args.arguments:
                value = bound_args.arguments[param]
                if value is not None:
                    if min_val is not None and value < min_val:
                        raise ValueError(f"Parameter '{param}' must be >= {min_val}, got {value}")
                    if max_val is not None and value > max_val:
                        raise ValueError(f"Parameter '{param}' must be <= {max_val}, got {value}")


__all__ = [
    "_get_bound_args",
    "_validate_parameters",
]
