"""
Tool Registry - 도구 레지스트리
"""

from typing import Any, Callable, Dict, List, Optional

from .schema import SchemaGenerator


class ToolRegistry:
    """
    도구 레지스트리

    모든 도구를 중앙에서 관리하고, 이름으로 검색/실행할 수 있습니다.

    Mathematical Foundation:
        Registry as Mapping:
        R: ToolName → Tool

        where ToolName is string identifier
        and Tool is (function, schema, metadata)

        Lookup: R[name] → Tool or ∅ (empty if not found)
    """

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        func: Callable,
        name: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        **metadata,
    ):
        """
        도구 등록

        Args:
            func: 도구 함수
            name: 도구 이름 (기본값: 함수 이름)
            schema: JSON Schema
            **metadata: 추가 메타데이터
        """
        tool_name = name or getattr(func, "tool_name", func.__name__)
        tool_schema = schema or getattr(func, "schema", SchemaGenerator.from_function(func))

        self._tools[tool_name] = func
        self._schemas[tool_name] = tool_schema
        self._metadata[tool_name] = {
            "description": getattr(func, "tool_description", func.__doc__ or ""),
            **metadata,
        }

    def get(self, name: str) -> Optional[Callable]:
        """도구 조회"""
        return self._tools.get(name)

    def get_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """스키마 조회"""
        return self._schemas.get(name)

    def list_tools(self) -> List[str]:
        """등록된 모든 도구 이름 목록"""
        return list(self._tools.keys())

    def execute(self, name: str, **params) -> Any:
        """
        이름으로 도구 실행

        Args:
            name: 도구 이름
            **params: 도구 파라미터

        Returns:
            도구 실행 결과

        Raises:
            KeyError: 도구가 없는 경우
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry")

        tool = self._tools[name]
        return tool(**params)

    def to_openai_format(self) -> List[Dict[str, Any]]:
        """
        OpenAI function calling 형식으로 변환

        Returns:
            OpenAI tools 리스트
        """
        tools = []
        for name in self._tools:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": self._metadata[name].get("description", ""),
                        "parameters": self._schemas[name],
                    },
                }
            )
        return tools


# ============================================================================
# Global Registry Instance
# ============================================================================

# 전역 레지스트리
default_registry = ToolRegistry()
