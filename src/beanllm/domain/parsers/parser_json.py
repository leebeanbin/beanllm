"""
JSON Output Parser - JSON 파서
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, cast

from beanllm.domain.parsers.base import BaseOutputParser
from beanllm.domain.parsers.exceptions import OutputParserException


class JSONOutputParser(BaseOutputParser):
    """
    JSON 파서

    LLM 출력을 Python dict로 변환

    Example:
        ```python
        from beanllm.domain.parsers import JSONOutputParser

        parser = JSONOutputParser()

        # 파싱
        data = parser.parse('{"name": "John", "age": 30}')
        print(data["name"])  # "John"
        ```
    """

    def parse(self, text: str) -> Dict[str, Any]:
        """
        JSON 텍스트를 dict로 변환

        Args:
            text: JSON 형식의 텍스트

        Returns:
            dict

        Raises:
            OutputParserException: 파싱 실패 시
        """
        try:
            # JSON 추출
            json_text = self._extract_json(text)

            # 파싱
            return cast(Dict[str, Any], json.loads(json_text))

        except json.JSONDecodeError as e:
            raise OutputParserException(f"Failed to parse JSON: {e}", llm_output=text)

    def _extract_json(self, text: str) -> str:
        """텍스트에서 JSON 추출"""
        # 코드 블록 제거
        json_match = re.search(r"```(?:json)?\s*(\{.+?\})\s*```", text, re.DOTALL)
        if json_match:
            return json_match.group(1)

        # 중괄호 찾기
        json_match = re.search(r"\{.+\}", text, re.DOTALL)
        if json_match:
            return json_match.group(0)

        return text.strip()

    def get_format_instructions(self) -> str:
        return """Output must be a valid JSON object.

Example:
```json
{
  "key1": "value1",
  "key2": "value2"
}
```

Return ONLY the JSON object, nothing else."""

    def get_output_type(self) -> str:
        return "Dict[str, Any]"
