"""
List Output Parsers - 쉼표/번호 리스트 파서
"""

from __future__ import annotations

import re
from typing import List

from beanllm.domain.parsers.base import BaseOutputParser


class CommaSeparatedListOutputParser(BaseOutputParser):
    """
    쉼표로 구분된 리스트 파서

    Example:
        ```python
        from beanllm.domain.parsers import CommaSeparatedListOutputParser

        parser = CommaSeparatedListOutputParser()
        items = parser.parse("apple, banana, cherry")
        # ["apple", "banana", "cherry"]
        ```
    """

    def parse(self, text: str) -> List[str]:
        """
        쉼표로 구분된 텍스트를 리스트로 변환

        Args:
            text: 쉼표로 구분된 텍스트

        Returns:
            문자열 리스트
        """
        # 앞뒤 공백, 코드 블록 제거
        text = text.strip()
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

        # 쉼표로 분할
        items = [item.strip() for item in text.split(",")]

        # 빈 항목 제거
        items = [item for item in items if item]

        return items

    def get_format_instructions(self) -> str:
        return """Output must be a comma-separated list.

Example:
item1, item2, item3

Return ONLY the comma-separated list, nothing else."""

    def get_output_type(self) -> str:
        return "List[str]"


class NumberedListOutputParser(BaseOutputParser):
    """
    번호가 매겨진 리스트 파서

    Example:
        ```python
        from beanllm.domain.parsers import NumberedListOutputParser

        parser = NumberedListOutputParser()
        items = parser.parse(\"\"\"
        1. First item
        2. Second item
        3. Third item
        \"\"\")
        # ["First item", "Second item", "Third item"]
        ```
    """

    def parse(self, text: str) -> List[str]:
        """
        번호가 매겨진 텍스트를 리스트로 변환

        Args:
            text: 번호가 매겨진 텍스트

        Returns:
            문자열 리스트
        """
        # 패턴: 1. item, 1) item, 1 - item
        patterns = [
            r"^\s*(\d+)\.\s*(.+)$",  # 1. item
            r"^\s*(\d+)\)\s*(.+)$",  # 1) item
            r"^\s*(\d+)\s*-\s*(.+)$",  # 1 - item
        ]

        items = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            # 패턴 매칭
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    items.append(match.group(2).strip())
                    break

        return items

    def get_format_instructions(self) -> str:
        return """Output must be a numbered list.

Example:
1. First item
2. Second item
3. Third item

Return ONLY the numbered list, nothing else."""

    def get_output_type(self) -> str:
        return "List[str]"
