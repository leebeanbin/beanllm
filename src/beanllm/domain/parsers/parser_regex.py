"""
Regex Output Parser - 정규식 기반 파서
"""

from __future__ import annotations

import re
from typing import List, Union

from beanllm.domain.parsers.base import BaseOutputParser
from beanllm.domain.parsers.exceptions import OutputParserException


class RegexOutputParser(BaseOutputParser):
    r"""
    정규식 기반 파서

    LLM 출력에서 정규식 패턴으로 매칭하여 값을 추출합니다.

    Example:
        ```python
        from beanllm.domain.parsers import RegexOutputParser

        # 단일 그룹 추출
        parser = RegexOutputParser(pattern=r"Answer: (.+)")
        result = parser.parse("Answer: 42")  # "42"

        # 여러 그룹 추출
        parser = RegexOutputParser(pattern=r"(\d+).*?(\d+)", return_groups=True)
        result = parser.parse("First: 1, Second: 2")  # ["1", "2"]
        ```
    """

    def __init__(
        self,
        pattern: Union[str, re.Pattern[str]],
        return_groups: bool = False,
        flags: int = 0,
    ):
        """
        Args:
            pattern: 정규식 패턴 (문자열 또는 컴파일된 패턴)
            return_groups: True면 캡처 그룹 리스트 반환, False면 첫 번째 그룹 또는 전체 매칭
            flags: re 모듈 플래그 (re.IGNORECASE 등)
        """
        self._pattern = re.compile(pattern, flags) if isinstance(pattern, str) else pattern
        self.return_groups = return_groups

    def parse(self, text: str) -> Union[str, List[str]]:
        """
        텍스트에서 정규식 매칭으로 값 추출

        Args:
            text: 파싱할 텍스트

        Returns:
            return_groups=False: 첫 번째 캡처 그룹이 있으면 해당 문자열, 없으면 전체 매칭
            return_groups=True: 캡처 그룹 리스트

        Raises:
            OutputParserException: 매칭 실패 시
        """
        text = text.strip()
        match = self._pattern.search(text)

        if match is None:
            raise OutputParserException(
                f"No match for pattern {self._pattern.pattern!r} in output",
                llm_output=text,
            )

        if self.return_groups:
            return list(match.groups())

        if match.lastindex is not None and match.lastindex >= 1:
            return match.group(1)
        return match.group(0)

    def get_format_instructions(self) -> str:
        return f"""Output must match the following pattern: {self._pattern.pattern}

Extract the requested information using this pattern. Return ONLY the matching part, nothing else."""

    def get_output_type(self) -> str:
        return "List[str]" if self.return_groups else "str"
