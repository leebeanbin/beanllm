"""
Dynamic Schema Generation - 동적 스키마 생성
"""

import inspect
from typing import Any, Callable, Dict, Type, Union, get_args, get_origin, get_type_hints

try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = None


class SchemaGenerator:
    """
    동적 스키마 생성기

    Python 함수의 타입 힌트로부터 JSON Schema를 자동 생성합니다.

    Mathematical Foundation:
        Type Inference: Γ ⊢ e: τ
        where Γ is type environment, e is expression, τ is type

        For function f with signature f: T₁ × T₂ × ... × Tₙ → R:
        Schema(f) = {
            "type": "object",
            "properties": {pᵢ: Schema(Tᵢ) for i in 1..n},
            "required": [pᵢ for i in 1..n if pᵢ has no default]
        }
    """

    _type_mapping = {
        int: {"type": "integer"},
        float: {"type": "number"},
        str: {"type": "string"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }

    @classmethod
    def from_function(cls, func: Callable) -> Dict[str, Any]:
        """
        함수로부터 JSON Schema 생성

        Args:
            func: Python 함수

        Returns:
            JSON Schema dict

        Example:
            >>> def greet(name: str, age: int = 25) -> str:
            ...     return f"Hello {name}, age {age}"
            >>> schema = SchemaGenerator.from_function(greet)
            >>> schema['properties']['name']
            {'type': 'string'}
        """
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name in type_hints:
                param_type = type_hints[param_name]
                properties[param_name] = cls._type_to_schema(param_type)

                # Add description from docstring if available
                if func.__doc__:
                    # Simple parsing - can be enhanced
                    properties[param_name]["description"] = f"Parameter {param_name}"

                # Required if no default value
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "description": func.__doc__ or f"Schema for {func.__name__}",
        }

    @classmethod
    def _type_to_schema(cls, type_hint: Type) -> Dict[str, Any]:
        """타입 힌트를 JSON Schema로 변환"""
        origin = get_origin(type_hint)

        # Handle Optional[T] -> Union[T, None]
        if origin is Union:
            args = get_args(type_hint)
            # Filter out NoneType
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                return cls._type_to_schema(non_none_args[0])

        # Handle List[T]
        if origin is list:
            args = get_args(type_hint)
            if args:
                return {"type": "array", "items": cls._type_to_schema(args[0])}
            return {"type": "array"}

        # Handle Dict[K, V]
        if origin is dict:
            return {"type": "object"}

        # Base types
        if type_hint in cls._type_mapping:
            return cls._type_mapping[type_hint].copy()

        # Enum
        from enum import Enum

        if isinstance(type_hint, type) and issubclass(type_hint, Enum):
            return {"type": "string", "enum": [e.value for e in type_hint]}

        # Fallback
        return {"type": "object"}

    @classmethod
    def from_pydantic(cls, model: Type[BaseModel]) -> Dict[str, Any]:
        """
        Pydantic 모델로부터 JSON Schema 생성

        Args:
            model: Pydantic BaseModel 클래스

        Returns:
            JSON Schema dict
        """
        if BaseModel is None:
            raise ImportError("pydantic is required for from_pydantic method")
        return model.schema()
