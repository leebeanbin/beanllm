"""
Parsers Utils - 파서 편의 함수
"""

from typing import Any, Dict, List

from .parsers import (
    BooleanOutputParser,
    CommaSeparatedListOutputParser,
    JSONOutputParser,
)


def parse_json(text: str) -> Dict[str, Any]:
    """JSON 파싱 편의 함수"""
    parser = JSONOutputParser()
    return parser.parse(text)


def parse_list(text: str, separator: str = ",") -> List[str]:
    """리스트 파싱 편의 함수"""
    if separator == ",":
        parser = CommaSeparatedListOutputParser()
    else:
        # 커스텀 separator
        items = [item.strip() for item in text.split(separator)]
        return [item for item in items if item]
    return parser.parse(text)


def parse_bool(text: str) -> bool:
    """Boolean 파싱 편의 함수"""
    parser = BooleanOutputParser()
    return parser.parse(text)
