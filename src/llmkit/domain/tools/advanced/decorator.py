"""
Advanced Tool Decorator - 고급 도구 데코레이터
"""

import inspect
import json
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional

from .schema import SchemaGenerator
from .validator import ToolValidator


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    schema: Optional[Dict[str, Any]] = None,
    validate: bool = True,
    retry: int = 1,
    cache: bool = False,
    cache_ttl: int = 300,
):
    """
    고급 도구 데코레이터

    기능:
    - 자동 스키마 생성
    - 입력 검증
    - 재시도 로직
    - 결과 캐싱

    Args:
        name: 도구 이름 (기본값: 함수 이름)
        description: 도구 설명
        schema: 커스텀 JSON Schema (자동 생성 대신)
        validate: 입력 검증 활성화
        retry: 재시도 횟수
        cache: 결과 캐싱 활성화
        cache_ttl: 캐시 유효 시간 (초)

    Example:
        >>> @tool(description="Calculate sum", validate=True, retry=3)
        ... def add(a: int, b: int) -> int:
        ...     return a + b
        >>> add.schema
        {'type': 'object', 'properties': {...}, ...}
    """

    def decorator(func: Callable) -> Callable:
        # Generate schema
        func_schema = schema or SchemaGenerator.from_function(func)
        func_name = name or func.__name__
        func_description = description or func.__doc__ or ""

        # Cache storage
        _cache = {} if cache else None

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Convert args to kwargs for validation
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            params = bound.arguments

            # Validate input
            if validate:
                is_valid, error = ToolValidator.validate(params, func_schema)
                if not is_valid:
                    raise ValueError(f"Tool validation failed: {error}")

            # Check cache
            if cache:
                cache_key = json.dumps(params, sort_keys=True)
                if cache_key in _cache:
                    cached_result, cached_time = _cache[cache_key]
                    if time.time() - cached_time < cache_ttl:
                        return cached_result

            # Execute with retry
            last_exception = None
            for attempt in range(retry):
                try:
                    result = func(**params)

                    # Store in cache
                    if cache:
                        _cache[cache_key] = (result, time.time())

                    return result

                except Exception as e:
                    last_exception = e
                    if attempt < retry - 1:
                        wait_time = 2**attempt
                        time.sleep(wait_time)

            raise last_exception

        # Attach metadata
        wrapper.schema = func_schema
        wrapper.tool_name = func_name
        wrapper.tool_description = func_description
        wrapper.is_tool = True

        return wrapper

    return decorator
