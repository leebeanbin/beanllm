"""
Structured Output Parsers - 날짜/시간, Enum, Boolean 파서
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Type

from beanllm.domain.parsers.base import BaseOutputParser
from beanllm.domain.parsers.exceptions import OutputParserException


class DatetimeOutputParser(BaseOutputParser):
    """
    날짜/시간 파서

    Example:
        ```python
        from beanllm.domain.parsers import DatetimeOutputParser

        parser = DatetimeOutputParser(format="%Y-%m-%d %H:%M:%S")
        dt = parser.parse("2024-01-15 10:30:00")
        ```
    """

    def __init__(self, format: str = "%Y-%m-%d %H:%M:%S"):
        """
        Args:
            format: datetime.strptime 형식 문자열
        """
        self.format = format

    def parse(self, text: str) -> datetime:
        """
        텍스트를 datetime으로 변환

        Args:
            text: 날짜/시간 문자열

        Returns:
            datetime 객체

        Raises:
            OutputParserException: 파싱 실패 시
        """
        try:
            text = text.strip()
            # 코드 블록 제거
            text = re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()

            return datetime.strptime(text, self.format)

        except ValueError as e:
            raise OutputParserException(f"Failed to parse datetime: {e}", llm_output=text)

    def get_format_instructions(self) -> str:
        return f"""Output must be a datetime string in the format: {self.format}

Example:
{datetime.now().strftime(self.format)}

Return ONLY the datetime string, nothing else."""

    def get_output_type(self) -> str:
        return "datetime"


class EnumOutputParser(BaseOutputParser):
    """
    Enum 파서

    Example:
        ```python
        from enum import Enum
        from beanllm.domain.parsers import EnumOutputParser

        class Color(Enum):
            RED = "red"
            GREEN = "green"
            BLUE = "blue"

        parser = EnumOutputParser(enum_class=Color)
        color = parser.parse("red")  # Color.RED
        ```
    """

    def __init__(self, enum_class: Type[Enum]):
        """
        Args:
            enum_class: Enum 클래스
        """
        self.enum_class = enum_class

    def parse(self, text: str) -> Enum:
        """
        텍스트를 Enum으로 변환

        Args:
            text: Enum 값 문자열

        Returns:
            Enum 인스턴스

        Raises:
            OutputParserException: 파싱 실패 시
        """
        text = text.strip().lower()

        # 값으로 찾기
        for member in self.enum_class:
            if member.value.lower() == text:
                return member

        # 이름으로 찾기
        for member in self.enum_class:
            if member.name.lower() == text:
                return member

        # 실패
        valid_values = [m.value for m in self.enum_class]
        raise OutputParserException(
            f"Invalid enum value: {text}. Valid values: {valid_values}", llm_output=text
        )

    def get_format_instructions(self) -> str:
        valid_values = [m.value for m in self.enum_class]
        return f"""Output must be one of the following values:
{", ".join(valid_values)}

Return ONLY one of these values, nothing else."""

    def get_output_type(self) -> str:
        return f"Enum[{self.enum_class.__name__}]"


class BooleanOutputParser(BaseOutputParser):
    """
    Boolean 파서

    Example:
        ```python
        from beanllm.domain.parsers import BooleanOutputParser

        parser = BooleanOutputParser()
        result = parser.parse("yes")  # True
        result = parser.parse("no")   # False
        ```
    """

    TRUE_VALUES = {"true", "yes", "y", "1", "ok", "correct"}
    FALSE_VALUES = {"false", "no", "n", "0", "not ok", "incorrect"}

    def parse(self, text: str) -> bool:
        """
        텍스트를 boolean으로 변환

        Args:
            text: boolean 값 문자열

        Returns:
            bool

        Raises:
            OutputParserException: 파싱 실패 시
        """
        text = text.strip().lower()

        if text in self.TRUE_VALUES:
            return True
        elif text in self.FALSE_VALUES:
            return False
        else:
            raise OutputParserException(f"Cannot parse as boolean: {text}", llm_output=text)

    def get_format_instructions(self) -> str:
        return """Output must be a boolean value.

Valid values for True: true, yes, y, 1
Valid values for False: false, no, n, 0

Return ONLY one of these values, nothing else."""

    def get_output_type(self) -> str:
        return "bool"
