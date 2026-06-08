"""
Advanced Tools 테스트 - SchemaGenerator, ToolValidator, ToolRegistry, @tool decorator
"""

from enum import Enum
from typing import List, Optional

import pytest

from beanllm.domain.tools.advanced.decorator import tool
from beanllm.domain.tools.advanced.registry import ToolRegistry, default_registry
from beanllm.domain.tools.advanced.schema import SchemaGenerator
from beanllm.domain.tools.advanced.validator import ToolValidator


class TestSchemaGenerator:
    def test_from_function_basic(self) -> None:
        def greet(name: str, age: int) -> str:
            return f"Hello {name}"

        schema = SchemaGenerator.from_function(greet)
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["properties"]["age"]["type"] == "integer"

    def test_from_function_required_fields(self) -> None:
        def func(required: str, optional: int = 10) -> str:
            return required

        schema = SchemaGenerator.from_function(func)
        assert "required" in schema["required"]
        assert "optional" not in schema["required"]

    def test_from_function_optional_type(self) -> None:
        def func(value: Optional[str] = None) -> None:
            pass

        schema = SchemaGenerator.from_function(func)
        assert "value" in schema["properties"]
        # Optional[str] should become string schema
        assert schema["properties"]["value"]["type"] == "string"

    def test_from_function_list_type(self) -> None:
        def func(items: List[str]) -> None:
            pass

        schema = SchemaGenerator.from_function(func)
        assert schema["properties"]["items"]["type"] == "array"
        assert schema["properties"]["items"]["items"]["type"] == "string"

    def test_from_function_bool_type(self) -> None:
        def func(flag: bool) -> None:
            pass

        schema = SchemaGenerator.from_function(func)
        assert schema["properties"]["flag"]["type"] == "boolean"

    def test_from_function_float_type(self) -> None:
        def func(value: float) -> None:
            pass

        schema = SchemaGenerator.from_function(func)
        assert schema["properties"]["value"]["type"] == "number"

    def test_from_function_dict_type(self) -> None:
        from typing import Dict

        def func(data: dict) -> None:
            pass

        schema = SchemaGenerator.from_function(func)
        assert schema["properties"]["data"]["type"] == "object"

    def test_from_function_enum_type(self) -> None:
        class Color(Enum):
            RED = "red"
            BLUE = "blue"

        def func(color: Color) -> None:
            pass

        schema = SchemaGenerator.from_function(func)
        assert schema["properties"]["color"]["type"] == "string"
        assert "red" in schema["properties"]["color"]["enum"]
        assert "blue" in schema["properties"]["color"]["enum"]

    def test_from_function_with_docstring(self) -> None:
        def func(x: int) -> int:
            """A function with docs."""
            return x

        schema = SchemaGenerator.from_function(func)
        # Description comes from docstring
        assert "A function with docs" in schema["description"]

    def test_from_function_no_type_hints(self) -> None:
        def func(x, y):
            return x + y

        schema = SchemaGenerator.from_function(func)
        # Parameters without type hints are not in properties
        assert isinstance(schema, dict)

    def test_type_to_schema_fallback(self) -> None:
        class CustomType:
            pass

        result = SchemaGenerator._type_to_schema(CustomType)
        assert result["type"] == "object"


class TestToolValidator:
    def test_validate_valid_data(self) -> None:
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"],
        }
        is_valid, error = ToolValidator.validate({"name": "Alice", "age": 30}, schema)
        assert is_valid is True
        assert error is None

    def test_validate_missing_required(self) -> None:
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        is_valid, error = ToolValidator.validate({}, schema)
        assert is_valid is False
        assert "Missing required field: name" in error

    def test_validate_wrong_type(self) -> None:
        schema = {
            "type": "object",
            "properties": {"age": {"type": "integer"}},
            "required": [],
        }
        is_valid, error = ToolValidator.validate({"age": "not_an_int"}, schema)
        assert is_valid is False
        assert "age" in error

    def test_validate_string_type(self) -> None:
        schema = {
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        is_valid, _ = ToolValidator.validate({"name": "Alice"}, schema)
        assert is_valid is True

    def test_validate_boolean_type(self) -> None:
        schema = {
            "properties": {"flag": {"type": "boolean"}},
            "required": [],
        }
        is_valid, _ = ToolValidator.validate({"flag": True}, schema)
        assert is_valid is True

    def test_validate_number_type_with_int(self) -> None:
        schema = {
            "properties": {"value": {"type": "number"}},
            "required": [],
        }
        is_valid, _ = ToolValidator.validate({"value": 3}, schema)
        assert is_valid is True

    def test_validate_enum_valid(self) -> None:
        schema = {
            "properties": {"color": {"type": "string", "enum": ["red", "blue", "green"]}},
            "required": [],
        }
        is_valid, _ = ToolValidator.validate({"color": "red"}, schema)
        assert is_valid is True

    def test_validate_enum_invalid(self) -> None:
        schema = {
            "properties": {"color": {"type": "string", "enum": ["red", "blue"]}},
            "required": [],
        }
        is_valid, error = ToolValidator.validate({"color": "yellow"}, schema)
        assert is_valid is False
        assert "yellow" in error or "color" in error

    def test_validate_number_minimum(self) -> None:
        schema = {
            "properties": {"score": {"type": "number", "minimum": 0}},
            "required": [],
        }
        is_valid, error = ToolValidator.validate({"score": -1}, schema)
        assert is_valid is False

    def test_validate_number_maximum(self) -> None:
        schema = {
            "properties": {"score": {"type": "number", "maximum": 100}},
            "required": [],
        }
        is_valid, error = ToolValidator.validate({"score": 150}, schema)
        assert is_valid is False

    def test_validate_array_type(self) -> None:
        schema = {
            "properties": {"items": {"type": "array"}},
            "required": [],
        }
        is_valid, _ = ToolValidator.validate({"items": [1, 2, 3]}, schema)
        assert is_valid is True

    def test_validate_array_items_type(self) -> None:
        schema = {
            "properties": {
                "numbers": {
                    "type": "array",
                    "items": {"type": "integer"},
                }
            },
            "required": [],
        }
        is_valid, _ = ToolValidator.validate({"numbers": [1, 2, 3]}, schema)
        assert is_valid is True

    def test_validate_array_items_wrong_type(self) -> None:
        schema = {
            "properties": {
                "numbers": {
                    "type": "array",
                    "items": {"type": "integer"},
                }
            },
            "required": [],
        }
        is_valid, error = ToolValidator.validate({"numbers": [1, "two", 3]}, schema)
        assert is_valid is False

    def test_validate_empty_schema(self) -> None:
        schema = {"required": [], "properties": {}}
        is_valid, _ = ToolValidator.validate({"extra": "field"}, schema)
        assert is_valid is True


class TestToolRegistry:
    @pytest.fixture
    def registry(self) -> ToolRegistry:
        return ToolRegistry()

    def test_register_and_list(self, registry: ToolRegistry) -> None:
        def my_func(x: int) -> int:
            return x * 2

        registry.register(my_func)
        assert "my_func" in registry.list_tools()

    def test_register_with_name(self, registry: ToolRegistry) -> None:
        def my_func(x: int) -> int:
            return x

        registry.register(my_func, name="custom_name")
        assert "custom_name" in registry.list_tools()

    def test_get_tool(self, registry: ToolRegistry) -> None:
        def compute(a: int, b: int) -> int:
            return a + b

        registry.register(compute)
        tool_func = registry.get("compute")
        assert tool_func is not None
        assert tool_func(a=2, b=3) == 5

    def test_get_nonexistent_tool(self, registry: ToolRegistry) -> None:
        assert registry.get("nonexistent") is None

    def test_get_schema(self, registry: ToolRegistry) -> None:
        def func(name: str) -> None:
            pass

        registry.register(func)
        schema = registry.get_schema("func")
        assert schema is not None
        assert "properties" in schema

    def test_execute(self, registry: ToolRegistry) -> None:
        def add(a: int, b: int) -> int:
            return a + b

        registry.register(add)
        result = registry.execute("add", a=3, b=4)
        assert result == 7

    def test_execute_unknown_raises(self, registry: ToolRegistry) -> None:
        with pytest.raises(KeyError):
            registry.execute("unknown_tool")

    def test_to_openai_format(self, registry: ToolRegistry) -> None:
        def search(query: str) -> str:
            """Search for documents."""
            return query

        registry.register(search)
        tools_list = registry.to_openai_format()
        assert len(tools_list) == 1
        assert tools_list[0]["type"] == "function"
        assert tools_list[0]["function"]["name"] == "search"

    def test_register_with_schema(self, registry: ToolRegistry) -> None:
        def func(x: int) -> int:
            return x

        custom_schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        registry.register(func, schema=custom_schema)
        assert registry.get_schema("func") == custom_schema

    def test_default_registry_exists(self) -> None:
        assert isinstance(default_registry, ToolRegistry)


class TestToolDecorator:
    def test_basic_tool_decorator(self) -> None:
        @tool()
        def add(a: int, b: int) -> int:
            return a + b

        result = add(a=1, b=2)
        assert result == 3

    def test_tool_has_schema_attribute(self) -> None:
        @tool()
        def greet(name: str) -> str:
            return f"Hello {name}"

        assert hasattr(greet, "schema")
        assert "properties" in greet.schema

    def test_tool_has_name_attribute(self) -> None:
        @tool(name="custom_tool")
        def func() -> None:
            pass

        assert func.tool_name == "custom_tool"

    def test_tool_has_description_attribute(self) -> None:
        @tool(description="My tool description")
        def func() -> None:
            pass

        assert func.tool_description == "My tool description"

    def test_tool_has_is_tool_attribute(self) -> None:
        @tool()
        def func() -> None:
            pass

        assert func.is_tool is True

    def test_tool_with_validation(self) -> None:
        @tool(validate=True)
        def multiply(a: int, b: int) -> int:
            return a * b

        result = multiply(a=3, b=4)
        assert result == 12

    def test_tool_retry_on_failure(self) -> None:
        call_count = [0]

        @tool(retry=3)
        def flaky_func() -> str:
            call_count[0] += 1
            if call_count[0] < 3:
                raise RuntimeError("flaky")
            return "success"

        result = flaky_func()
        assert result == "success"
        assert call_count[0] == 3

    def test_tool_retry_exhausted_raises(self) -> None:
        @tool(retry=2)
        def always_fails() -> str:
            raise ValueError("always fails")

        with pytest.raises(ValueError):
            always_fails()

    def test_tool_custom_schema(self) -> None:
        custom = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": []}

        @tool(schema=custom)
        def func(x: int) -> int:
            return x

        assert func.schema == custom

    def test_tool_preserves_function_name(self) -> None:
        @tool()
        def my_special_function(x: int) -> int:
            return x

        assert my_special_function.__name__ == "my_special_function"
