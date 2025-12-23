"""
Tools 테스트 - 도구 시스템 테스트
"""

import pytest
from unittest.mock import Mock

from llmkit.domain.tools import Tool, ToolParameter, ToolRegistry, register_tool, get_tool


class TestTool:
    """Tool 테스트"""

    def test_tool_from_function(self):
        """함수로부터 Tool 생성 테스트"""

        def test_function(query: str) -> str:
            """Test function"""
            return f"Result: {query}"

        tool = Tool.from_function(test_function)

        assert isinstance(tool, Tool)
        assert tool.name == "test_function"
        assert tool.description == "Test function"
        assert len(tool.parameters) > 0

    def test_tool_execute(self):
        """Tool 실행 테스트"""

        def add(a: int, b: int) -> int:
            """Add two numbers"""
            return a + b

        tool = Tool.from_function(add)
        result = tool.execute({"a": 2, "b": 3})

        assert result == 5

    def test_tool_to_openai_format(self):
        """OpenAI 형식 변환 테스트"""

        def search(query: str) -> str:
            """Search function"""
            return f"Results: {query}"

        tool = Tool.from_function(search)
        openai_format = tool.to_openai_format()

        assert isinstance(openai_format, dict)
        assert "type" in openai_format
        assert openai_format["type"] == "function"

    def test_tool_to_anthropic_format(self):
        """Anthropic 형식 변환 테스트"""

        def search(query: str) -> str:
            """Search function"""
            return f"Results: {query}"

        tool = Tool.from_function(search)
        anthropic_format = tool.to_anthropic_format()

        assert isinstance(anthropic_format, dict)
        assert "name" in anthropic_format
        assert "input_schema" in anthropic_format


class TestToolRegistry:
    """ToolRegistry 테스트"""

    @pytest.fixture
    def registry(self):
        """ToolRegistry 인스턴스"""
        return ToolRegistry()

    def test_register_tool(self, registry):
        """도구 등록 테스트"""

        def test_tool(x: int) -> int:
            """Test tool"""
            return x * 2

        registry.register(test_tool)
        tool = registry.get_tool("test_tool")

        assert tool is not None
        assert tool.name == "test_tool"

    def test_register_decorator(self, registry):
        """데코레이터로 도구 등록 테스트"""

        @registry.register
        def multiply(a: float, b: float) -> float:
            """Multiply two numbers"""
            return a * b

        tool = registry.get_tool("multiply")

        assert tool is not None
        assert tool.name == "multiply"

    def test_add_tool(self, registry):
        """도구 추가 테스트"""

        def test_tool(x: str) -> str:
            """Test tool"""
            return f"Result: {x}"

        tool = Tool.from_function(test_tool)
        registry.add_tool(tool)

        retrieved = registry.get_tool("test_tool")
        assert retrieved is not None

    def test_get_all_tools(self, registry):
        """모든 도구 조회 테스트"""

        def tool1(x: int) -> int:
            return x

        def tool2(y: str) -> str:
            return y

        registry.register(tool1)
        registry.register(tool2)

        all_tools = registry.get_all()

        assert isinstance(all_tools, list)
        assert len(all_tools) >= 2

    def test_execute_tool(self, registry):
        """도구 실행 테스트"""

        def add(a: int, b: int) -> int:
            """Add numbers"""
            return a + b

        registry.register(add)
        result = registry.execute("add", {"a": 5, "b": 3})

        assert result == 8

    def test_global_register_tool(self):
        """전역 레지스트리 도구 등록 테스트"""

        @register_tool
        def global_tool(x: int) -> int:
            """Global tool"""
            return x * 2

        tool = get_tool("global_tool")

        assert tool is not None
        assert tool.name == "global_tool"


