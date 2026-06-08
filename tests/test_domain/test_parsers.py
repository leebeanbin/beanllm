"""
Domain Parsers 테스트 — comprehensive coverage of all parser implementations.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from beanllm.domain.parsers import (
    BooleanOutputParser,
    CommaSeparatedListOutputParser,
    DatetimeOutputParser,
    EnumOutputParser,
    JSONOutputParser,
    NumberedListOutputParser,
    OutputParserException,
    RegexOutputParser,
)
from beanllm.domain.parsers.parser_retry import RetryOutputParser

# ---------------------------------------------------------------------------
# JSONOutputParser
# ---------------------------------------------------------------------------


class TestJSONOutputParser:
    @pytest.fixture
    def parser(self) -> JSONOutputParser:
        return JSONOutputParser()

    def test_parse_valid_json(self, parser: JSONOutputParser) -> None:
        result = parser.parse('{"name": "Alice", "age": 30}')
        assert result["name"] == "Alice"
        assert result["age"] == 30

    def test_parse_json_in_code_block(self, parser: JSONOutputParser) -> None:
        text = '```json\n{"key": "value"}\n```'
        result = parser.parse(text)
        assert result["key"] == "value"

    def test_parse_json_in_unnamed_block(self, parser: JSONOutputParser) -> None:
        text = '```\n{"key": "val"}\n```'
        result = parser.parse(text)
        assert result["key"] == "val"

    def test_parse_invalid_json_raises(self, parser: JSONOutputParser) -> None:
        with pytest.raises(OutputParserException):
            parser.parse("not json at all")

    def test_parse_nested_json(self, parser: JSONOutputParser) -> None:
        text = '{"user": {"name": "Bob", "scores": [1, 2, 3]}}'
        result = parser.parse(text)
        assert result["user"]["name"] == "Bob"
        assert result["user"]["scores"] == [1, 2, 3]

    def test_parse_json_embedded_in_text(self, parser: JSONOutputParser) -> None:
        text = 'Here is the output: {"status": "ok"} done'
        result = parser.parse(text)
        assert result["status"] == "ok"

    def test_parse_whitespace_trimmed(self, parser: JSONOutputParser) -> None:
        result = parser.parse('  {"a": 1}  ')
        assert result["a"] == 1

    def test_get_format_instructions_contains_json(self, parser: JSONOutputParser) -> None:
        instructions = parser.get_format_instructions()
        assert isinstance(instructions, str)
        assert len(instructions) > 0

    def test_get_output_type(self, parser: JSONOutputParser) -> None:
        assert "Dict" in parser.get_output_type()


# ---------------------------------------------------------------------------
# CommaSeparatedListOutputParser
# ---------------------------------------------------------------------------


class TestCommaSeparatedListOutputParser:
    @pytest.fixture
    def parser(self) -> CommaSeparatedListOutputParser:
        return CommaSeparatedListOutputParser()

    def test_parse_simple_list(self, parser: CommaSeparatedListOutputParser) -> None:
        result = parser.parse("apple, banana, cherry")
        assert result == ["apple", "banana", "cherry"]

    def test_parse_no_spaces(self, parser: CommaSeparatedListOutputParser) -> None:
        result = parser.parse("a,b,c")
        assert result == ["a", "b", "c"]

    def test_parse_strips_whitespace(self, parser: CommaSeparatedListOutputParser) -> None:
        result = parser.parse("  item1 ,  item2 ,  item3  ")
        assert result == ["item1", "item2", "item3"]

    def test_parse_removes_empty_items(self, parser: CommaSeparatedListOutputParser) -> None:
        result = parser.parse("a, , b, , c")
        assert result == ["a", "b", "c"]

    def test_parse_single_item(self, parser: CommaSeparatedListOutputParser) -> None:
        result = parser.parse("only_item")
        assert result == ["only_item"]

    def test_parse_removes_code_block(self, parser: CommaSeparatedListOutputParser) -> None:
        result = parser.parse("```python\nx=1\n```\napple, banana")
        assert "apple" in result
        assert "banana" in result

    def test_trailing_comma_no_empty(self, parser: CommaSeparatedListOutputParser) -> None:
        result = parser.parse("a,b,")
        assert "" not in result

    def test_get_format_instructions(self, parser: CommaSeparatedListOutputParser) -> None:
        instructions = parser.get_format_instructions()
        assert isinstance(instructions, str)
        assert "comma" in instructions.lower()

    def test_get_output_type(self, parser: CommaSeparatedListOutputParser) -> None:
        assert "List" in parser.get_output_type()


# ---------------------------------------------------------------------------
# NumberedListOutputParser
# ---------------------------------------------------------------------------


class TestNumberedListOutputParser:
    @pytest.fixture
    def parser(self) -> NumberedListOutputParser:
        return NumberedListOutputParser()

    def test_parse_dot_format(self, parser: NumberedListOutputParser) -> None:
        text = "1. First item\n2. Second item\n3. Third item"
        result = parser.parse(text)
        assert result == ["First item", "Second item", "Third item"]

    def test_parse_paren_format(self, parser: NumberedListOutputParser) -> None:
        text = "1) Item one\n2) Item two"
        result = parser.parse(text)
        assert result == ["Item one", "Item two"]

    def test_parse_dash_format(self, parser: NumberedListOutputParser) -> None:
        text = "1 - Alpha\n2 - Beta"
        result = parser.parse(text)
        assert result == ["Alpha", "Beta"]

    def test_parse_handles_various_formats(self, parser: NumberedListOutputParser) -> None:
        text = "1) Item one\n2) Item two"
        result = parser.parse(text)
        assert len(result) >= 1

    def test_skips_non_numbered_lines(self, parser: NumberedListOutputParser) -> None:
        text = "Header\n1. First\nSome text\n2. Second"
        result = parser.parse(text)
        assert "First" in result
        assert "Second" in result
        assert "Header" not in result

    def test_empty_text_returns_empty(self, parser: NumberedListOutputParser) -> None:
        result = parser.parse("")
        assert result == []

    def test_get_format_instructions(self, parser: NumberedListOutputParser) -> None:
        instructions = parser.get_format_instructions()
        assert isinstance(instructions, str)
        assert "1." in instructions or "numbered" in instructions.lower()

    def test_get_output_type(self, parser: NumberedListOutputParser) -> None:
        assert "List" in parser.get_output_type()


# ---------------------------------------------------------------------------
# BooleanOutputParser
# ---------------------------------------------------------------------------


class TestBooleanOutputParser:
    @pytest.fixture
    def parser(self) -> BooleanOutputParser:
        return BooleanOutputParser()

    def test_parse_true_values(self, parser: BooleanOutputParser) -> None:
        assert parser.parse("True") is True
        assert parser.parse("true") is True
        assert parser.parse("yes") is True
        assert parser.parse("YES") is True
        assert parser.parse("y") is True
        assert parser.parse("1") is True
        assert parser.parse("ok") is True

    def test_parse_false_values(self, parser: BooleanOutputParser) -> None:
        assert parser.parse("False") is False
        assert parser.parse("false") is False
        assert parser.parse("no") is False
        assert parser.parse("NO") is False
        assert parser.parse("n") is False
        assert parser.parse("0") is False

    def test_parse_strips_whitespace(self, parser: BooleanOutputParser) -> None:
        assert parser.parse("  yes  ") is True
        assert parser.parse("  no  ") is False

    def test_parse_invalid_raises(self, parser: BooleanOutputParser) -> None:
        with pytest.raises(OutputParserException):
            parser.parse("maybe")

    def test_parse_unknown_word_raises(self, parser: BooleanOutputParser) -> None:
        with pytest.raises(OutputParserException):
            parser.parse("affirmative")

    def test_get_output_type(self, parser: BooleanOutputParser) -> None:
        assert "bool" in parser.get_output_type().lower()

    def test_get_format_instructions_has_true_false(self, parser: BooleanOutputParser) -> None:
        instr = parser.get_format_instructions()
        assert "true" in instr.lower()
        assert "false" in instr.lower()


# ---------------------------------------------------------------------------
# DatetimeOutputParser
# ---------------------------------------------------------------------------


class TestDatetimeOutputParser:
    @pytest.fixture
    def parser(self) -> DatetimeOutputParser:
        return DatetimeOutputParser(format="%Y-%m-%d %H:%M:%S")

    def test_parse_valid_datetime(self, parser: DatetimeOutputParser) -> None:
        result = parser.parse("2026-01-15 10:30:00")
        assert isinstance(result, datetime)
        assert result.year == 2026
        assert result.month == 1
        assert result.day == 15

    def test_parse_strips_whitespace(self, parser: DatetimeOutputParser) -> None:
        result = parser.parse("  2026-03-20 08:00:00  ")
        assert isinstance(result, datetime)
        assert result.year == 2026

    def test_parse_invalid_format_raises(self, parser: DatetimeOutputParser) -> None:
        with pytest.raises(OutputParserException):
            parser.parse("not a date")

    def test_parse_wrong_format_raises(self, parser: DatetimeOutputParser) -> None:
        with pytest.raises(OutputParserException):
            parser.parse("15/01/2026")

    def test_custom_format(self) -> None:
        parser = DatetimeOutputParser(format="%d/%m/%Y")
        result = parser.parse("25/12/2026")
        assert result.day == 25
        assert result.month == 12
        assert result.year == 2026

    def test_get_format_instructions_contains_format(self, parser: DatetimeOutputParser) -> None:
        instructions = parser.get_format_instructions()
        assert "%Y-%m-%d %H:%M:%S" in instructions

    def test_get_output_type(self, parser: DatetimeOutputParser) -> None:
        assert "datetime" in parser.get_output_type()


# ---------------------------------------------------------------------------
# EnumOutputParser
# ---------------------------------------------------------------------------


class TestEnumOutputParser:
    class Color(Enum):
        RED = "red"
        GREEN = "green"
        BLUE = "blue"

    @pytest.fixture
    def parser(self) -> EnumOutputParser:
        return EnumOutputParser(enum_class=self.Color)

    def test_parse_valid_enum_value(self, parser: EnumOutputParser) -> None:
        result = parser.parse("red")
        assert result == self.Color.RED

    def test_parse_valid_enum_name(self, parser: EnumOutputParser) -> None:
        result = parser.parse("GREEN")
        assert result == self.Color.GREEN

    def test_parse_case_insensitive_value(self, parser: EnumOutputParser) -> None:
        result = parser.parse("Blue")
        assert result == self.Color.BLUE

    def test_parse_strips_whitespace(self, parser: EnumOutputParser) -> None:
        result = parser.parse("  red  ")
        assert result == self.Color.RED

    def test_parse_invalid_raises(self, parser: EnumOutputParser) -> None:
        with pytest.raises(OutputParserException):
            parser.parse("yellow")

    def test_get_format_instructions_lists_values(self, parser: EnumOutputParser) -> None:
        instructions = parser.get_format_instructions()
        assert "red" in instructions
        assert "green" in instructions
        assert "blue" in instructions

    def test_get_output_type_has_class_name(self, parser: EnumOutputParser) -> None:
        ot = parser.get_output_type()
        assert "Color" in ot


# ---------------------------------------------------------------------------
# RegexOutputParser
# ---------------------------------------------------------------------------


class TestRegexOutputParser:
    def test_parse_single_group_extraction(self) -> None:
        parser = RegexOutputParser(pattern=r"Answer: (.+)")
        result = parser.parse("Here is the Answer: 42")
        assert result == "42"

    def test_parse_no_groups_returns_full_match(self) -> None:
        parser = RegexOutputParser(pattern=r"\d+")
        result = parser.parse("The number is 123")
        assert result == "123"

    def test_parse_with_groups(self) -> None:
        parser = RegexOutputParser(pattern=r"(\d+).*?(\d+)", return_groups=True)
        result = parser.parse("I have 3 apples and 5 oranges")
        assert isinstance(result, (list, tuple))
        assert "3" in str(result)

    def test_parse_return_groups_true(self) -> None:
        parser = RegexOutputParser(pattern=r"(\d+).*?(\d+)", return_groups=True)
        result = parser.parse("First: 10, Second: 20")
        assert isinstance(result, list)
        assert "10" in result
        assert "20" in result

    def test_parse_no_match_raises(self) -> None:
        parser = RegexOutputParser(pattern=r"\d{4}-\d{2}-\d{2}")
        with pytest.raises(OutputParserException):
            parser.parse("no date here")

    def test_case_insensitive_flag(self) -> None:
        import re

        parser = RegexOutputParser(pattern=r"answer: (.+)", flags=re.IGNORECASE)
        result = parser.parse("ANSWER: hello")
        assert result == "hello"

    def test_compiled_pattern(self) -> None:
        import re

        pattern = re.compile(r"Name: (\w+)")
        parser = RegexOutputParser(pattern=pattern)
        result = parser.parse("Name: Alice")
        assert result == "Alice"

    def test_get_format_instructions(self) -> None:
        parser = RegexOutputParser(pattern=r"(.+)")
        instr = parser.get_format_instructions()
        assert "pattern" in instr.lower()

    def test_get_output_type_single(self) -> None:
        parser = RegexOutputParser(pattern=r"(.+)")
        assert parser.get_output_type() == "str"

    def test_get_output_type_groups(self) -> None:
        parser = RegexOutputParser(pattern=r"(.+)", return_groups=True)
        assert parser.get_output_type() == "List[str]"

    def test_parse_full_match(self) -> None:
        parser = RegexOutputParser(pattern=r"\d+")
        result = parser.parse("score: 95")
        assert "95" in str(result)


# ---------------------------------------------------------------------------
# RetryOutputParser
# ---------------------------------------------------------------------------


class TestRetryOutputParser:
    @pytest.fixture
    def base_parser(self) -> JSONOutputParser:
        return JSONOutputParser()

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def parser(self, base_parser, mock_client) -> RetryOutputParser:
        return RetryOutputParser(parser=base_parser, client=mock_client, max_retries=2)

    def test_parse_delegates_to_base(self, parser: RetryOutputParser) -> None:
        result = parser.parse('{"key": "val"}')
        assert result["key"] == "val"

    def test_get_format_instructions_from_base(self, parser: RetryOutputParser) -> None:
        instr = parser.get_format_instructions()
        assert isinstance(instr, str)
        assert len(instr) > 0

    def test_get_output_type_wraps_base(self, parser: RetryOutputParser) -> None:
        ot = parser.get_output_type()
        assert "Retry" in ot

    async def test_parse_with_retry_success_first_attempt(
        self, parser: RetryOutputParser, mock_client: MagicMock
    ) -> None:
        result = await parser.parse_with_retry('{"ok": true}')
        assert result["ok"] is True
        mock_client.chat.assert_not_called()

    async def test_parse_with_retry_retries_on_failure(
        self, base_parser: JSONOutputParser, mock_client: MagicMock
    ) -> None:
        response = MagicMock()
        response.content = '{"fixed": "yes"}'
        mock_client.chat = AsyncMock(return_value=response)
        retry_parser = RetryOutputParser(parser=base_parser, client=mock_client, max_retries=2)

        result = await retry_parser.parse_with_retry("invalid json @@")
        assert result["fixed"] == "yes"
        mock_client.chat.assert_called()

    async def test_parse_with_retry_raises_after_max_retries(
        self, base_parser: JSONOutputParser, mock_client: MagicMock
    ) -> None:
        response = MagicMock()
        response.content = "still invalid"
        mock_client.chat = AsyncMock(return_value=response)
        retry_parser = RetryOutputParser(parser=base_parser, client=mock_client, max_retries=2)

        with pytest.raises(OutputParserException, match="Failed after"):
            await retry_parser.parse_with_retry("bad input @@")

    async def test_parse_with_retry_custom_prompt(
        self, base_parser: JSONOutputParser, mock_client: MagicMock
    ) -> None:
        response = MagicMock()
        response.content = '{"ok": 1}'
        mock_client.chat = AsyncMock(return_value=response)
        retry_parser = RetryOutputParser(parser=base_parser, client=mock_client, max_retries=2)
        custom = "Fix this: {completion}\nError: {error}\nInstructions: {instructions}"

        result = await retry_parser.parse_with_retry("bad @@", prompt_template=custom)
        assert result["ok"] == 1

    async def test_parse_with_retry_client_called_correct_times(
        self, base_parser: JSONOutputParser, mock_client: MagicMock
    ) -> None:
        response = MagicMock()
        response.content = "still bad"
        mock_client.chat = AsyncMock(return_value=response)
        retry_parser = RetryOutputParser(parser=base_parser, client=mock_client, max_retries=2)

        with pytest.raises(OutputParserException):
            await retry_parser.parse_with_retry("invalid @@")

        assert mock_client.chat.call_count == 2
