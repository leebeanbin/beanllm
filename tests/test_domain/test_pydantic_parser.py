"""Tests for domain/parsers/parser_pydantic.py — PydanticOutputParser."""

import json

import pytest
from pydantic import BaseModel

from beanllm.domain.parsers.exceptions import OutputParserException
from beanllm.domain.parsers.parser_pydantic import PydanticOutputParser


class Person(BaseModel):
    name: str
    age: int
    email: str = ""


class Config(BaseModel):
    host: str
    port: int = 8080
    debug: bool = False
    tags: list = []
    extra: dict = {}


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestPydanticOutputParserInit:
    def test_creates_with_pydantic_model(self):
        parser = PydanticOutputParser(pydantic_object=Person)
        assert parser.pydantic_object is Person

    def test_stores_model_class(self):
        parser = PydanticOutputParser(pydantic_object=Config)
        assert parser.pydantic_object is Config


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------


class TestPydanticOutputParserParse:
    def test_parse_valid_json(self):
        parser = PydanticOutputParser(pydantic_object=Person)
        result = parser.parse('{"name": "Alice", "age": 30}')
        assert isinstance(result, Person)
        assert result.name == "Alice"
        assert result.age == 30

    def test_parse_json_with_extra_text_before(self):
        parser = PydanticOutputParser(pydantic_object=Person)
        text = 'Here is the result:\n{"name": "Bob", "age": 25}'
        result = parser.parse(text)
        assert result.name == "Bob"

    def test_parse_json_in_code_block(self):
        parser = PydanticOutputParser(pydantic_object=Person)
        text = '```json\n{"name": "Charlie", "age": 20}\n```'
        result = parser.parse(text)
        assert result.name == "Charlie"

    def test_parse_invalid_json_raises(self):
        parser = PydanticOutputParser(pydantic_object=Person)
        with pytest.raises(OutputParserException):
            parser.parse("not json at all")

    def test_parse_invalid_model_raises(self):
        parser = PydanticOutputParser(pydantic_object=Person)
        # age must be int, providing string
        with pytest.raises(OutputParserException):
            parser.parse('{"name": "Alice", "age": "not_an_int"}')

    def test_parse_all_optional_fields_omitted(self):
        parser = PydanticOutputParser(pydantic_object=Person)
        result = parser.parse('{"name": "Dave", "age": 40}')
        assert result.name == "Dave"
        assert result.email == ""

    def test_parse_complex_model(self):
        parser = PydanticOutputParser(pydantic_object=Config)
        text = '{"host": "localhost", "port": 3306}'
        result = parser.parse(text)
        assert result.host == "localhost"
        assert result.port == 3306


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------


class TestExtractJson:
    def setup_method(self):
        self.parser = PydanticOutputParser(pydantic_object=Person)

    def test_extracts_json_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = self.parser._extract_json(text)
        assert result == '{"key": "value"}'

    def test_extracts_bare_code_block(self):
        text = '```\n{"key": "value"}\n```'
        result = self.parser._extract_json(text)
        assert result == '{"key": "value"}'

    def test_extracts_inline_json(self):
        text = 'Text before {"key": "value"} text after'
        result = self.parser._extract_json(text)
        assert '{"key": "value"}' in result

    def test_returns_text_if_no_json_found(self):
        text = "just plain text"
        result = self.parser._extract_json(text)
        assert result == "just plain text"


# ---------------------------------------------------------------------------
# get_format_instructions
# ---------------------------------------------------------------------------


class TestGetFormatInstructions:
    def test_returns_string(self):
        parser = PydanticOutputParser(pydantic_object=Person)
        instructions = parser.get_format_instructions()
        assert isinstance(instructions, str)

    def test_contains_field_names(self):
        parser = PydanticOutputParser(pydantic_object=Person)
        instructions = parser.get_format_instructions()
        assert "name" in instructions
        assert "age" in instructions

    def test_contains_json_keyword(self):
        parser = PydanticOutputParser(pydantic_object=Person)
        instructions = parser.get_format_instructions()
        assert "json" in instructions.lower() or "JSON" in instructions

    def test_contains_required_marker_for_required_field(self):
        parser = PydanticOutputParser(pydantic_object=Person)
        instructions = parser.get_format_instructions()
        assert "required" in instructions


# ---------------------------------------------------------------------------
# _get_example_output
# ---------------------------------------------------------------------------


class TestGetExampleOutput:
    def test_returns_dict(self):
        parser = PydanticOutputParser(pydantic_object=Person)
        example = parser._get_example_output()
        assert isinstance(example, dict)

    def test_all_field_types_covered(self):
        parser = PydanticOutputParser(pydantic_object=Config)
        example = parser._get_example_output()
        assert "host" in example  # string
        assert "port" in example  # integer
        assert "debug" in example  # boolean
        assert "tags" in example  # array
        assert "extra" in example  # object

    def test_string_field_returns_example_string(self):
        parser = PydanticOutputParser(pydantic_object=Person)
        example = parser._get_example_output()
        assert isinstance(example.get("name"), str)

    def test_int_field_returns_int(self):
        parser = PydanticOutputParser(pydantic_object=Person)
        example = parser._get_example_output()
        assert isinstance(example.get("age"), int)


# ---------------------------------------------------------------------------
# get_output_type
# ---------------------------------------------------------------------------


class TestGetOutputType:
    def test_returns_type_string(self):
        parser = PydanticOutputParser(pydantic_object=Person)
        output_type = parser.get_output_type()
        assert "Person" in output_type
        assert "Pydantic" in output_type
