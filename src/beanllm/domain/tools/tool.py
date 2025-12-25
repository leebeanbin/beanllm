"""
Tool - Function Calling 도구 정의
"""

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ...utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ToolParameter:
    """도구 파라미터"""

    name: str
    type: str  # string, number, boolean, object, array
    description: str
    required: bool = True
    enum: Optional[List[str]] = None


@dataclass
class Tool:
    """
    도구 (Function)

    Example:
        ```python
        from beanllm.domain.tools import Tool

        def search(query: str) -> str:
            '''웹 검색'''
            return f"Search results for: {query}"

        tool = Tool.from_function(search)
        result = tool.execute({"query": "Python"})
        ```
    """

    name: str
    description: str
    parameters: List[ToolParameter]
    function: Callable
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_openai_format(self) -> Dict:
        """OpenAI Function Calling 형식으로 변환"""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {"type": param.type, "description": param.description}
            if param.enum:
                prop["enum"] = param.enum

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        }

    def to_anthropic_format(self) -> Dict:
        """Anthropic Tool 형식으로 변환"""
        input_schema = {"type": "object", "properties": {}, "required": []}

        for param in self.parameters:
            input_schema["properties"][param.name] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                input_schema["properties"][param.name]["enum"] = param.enum

            if param.required:
                input_schema["required"].append(param.name)

        return {"name": self.name, "description": self.description, "input_schema": input_schema}

    def execute(self, arguments: Dict[str, Any]) -> Any:
        """
        도구 실행

        Args:
            arguments: 도구 파라미터

        Returns:
            도구 실행 결과
        """
        try:
            logger.debug(f"Executing tool {self.name} with args: {arguments}")
            result = self.function(**arguments)
            logger.debug(f"Tool {self.name} result: {result}")
            return result
        except Exception as e:
            logger.error(f"Tool {self.name} error: {e}")
            raise

    @classmethod
    def from_function(
        cls, func: Callable, name: Optional[str] = None, description: Optional[str] = None
    ) -> "Tool":
        """
        Python 함수에서 Tool 생성

        Args:
            func: Python 함수
            name: 도구 이름 (기본: 함수 이름)
            description: 설명 (기본: docstring)

        Returns:
            Tool 인스턴스

        Example:
            ```python
            def calculator(operation: str, a: float, b: float) -> float:
                '''간단한 계산기'''
                if operation == "add":
                    return a + b
                elif operation == "subtract":
                    return a - b
                elif operation == "multiply":
                    return a * b
                elif operation == "divide":
                    return a / b

            tool = Tool.from_function(calculator)
            ```
        """
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or "No description"

        # 함수 시그니처 분석
        sig = inspect.signature(func)
        parameters = []

        for param_name, param in sig.parameters.items():
            # 타입 힌트에서 타입 추출
            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                if param.annotation is int or param.annotation is float:
                    param_type = "number"
                elif param.annotation is bool:
                    param_type = "boolean"
                elif param.annotation is list:
                    param_type = "array"
                elif param.annotation is dict:
                    param_type = "object"

            # 필수 여부
            required = param.default == inspect.Parameter.empty

            parameters.append(
                ToolParameter(
                    name=param_name,
                    type=param_type,
                    description=f"Parameter {param_name}",
                    required=required,
                )
            )

        return cls(
            name=tool_name,
            description=tool_description.strip(),
            parameters=parameters,
            function=func,
        )
