"""
Tool Registry - 도구 레지스트리
"""

from typing import Any, Callable, Dict, List, Optional

from ...utils.logger import get_logger
from .tool import Tool

logger = get_logger(__name__)


class ToolRegistry:
    """
    도구 레지스트리

    Example:
        ```python
        from beanllm.domain.tools import ToolRegistry, Tool

        registry = ToolRegistry()

        @registry.register
        def search(query: str) -> str:
            '''웹 검색'''
            return f"Results: {query}"

        @registry.register
        def calculator(a: float, b: float) -> float:
            '''계산'''
            return a + b

        # 모든 도구 가져오기
        tools = registry.get_all()

        # 특정 도구 실행
        result = registry.execute("search", {"query": "Python"})
        ```
    """

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(
        self,
        func: Optional[Callable] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """
        도구 등록 (데코레이터로 사용 가능)

        Example:
            ```python
            @registry.register
            def my_tool(x: int) -> int:
                return x * 2
            ```
        """

        def decorator(f: Callable) -> Callable:
            tool = Tool.from_function(f, name=name, description=description)
            self.tools[tool.name] = tool
            logger.info(f"Registered tool: {tool.name}")
            return f

        if func is None:
            return decorator
        else:
            return decorator(func)

    def add_tool(self, tool: Tool):
        """도구 추가"""
        self.tools[tool.name] = tool
        logger.info(f"Added tool: {tool.name}")

    def get_tool(self, name: str) -> Optional[Tool]:
        """도구 가져오기"""
        return self.tools.get(name)

    def get_all(self) -> List[Tool]:
        """모든 도구 가져오기"""
        return list(self.tools.values())

    def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        """도구 실행"""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")
        return tool.execute(arguments)

    def to_openai_format(self) -> List[Dict]:
        """OpenAI Function Calling 형식"""
        return [tool.to_openai_format() for tool in self.tools.values()]

    def to_anthropic_format(self) -> List[Dict]:
        """Anthropic Tool 형식"""
        return [tool.to_anthropic_format() for tool in self.tools.values()]


# 전역 레지스트리
_global_registry = ToolRegistry()


def register_tool(
    func: Optional[Callable] = None, name: Optional[str] = None, description: Optional[str] = None
):
    """
    전역 레지스트리에 도구 등록

    Example:
        ```python
        from beanllm.domain.tools import register_tool

        @register_tool
        def my_tool(x: int) -> int:
            '''My tool'''
            return x * 2
        ```
    """
    return _global_registry.register(func, name, description)


def get_tool(name: str) -> Optional[Tool]:
    """전역 레지스트리에서 도구 가져오기"""
    return _global_registry.get_tool(name)


def get_all_tools() -> List[Tool]:
    """전역 레지스트리의 모든 도구"""
    return _global_registry.get_all()
