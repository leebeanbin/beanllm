"""
Parsers Base - 파서 베이스 클래스
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseOutputParser(ABC):
    """
    Output Parser 베이스 클래스

    LLM 출력을 구조화된 데이터로 변환하는 파서의 기본 인터페이스
    """

    @abstractmethod
    def parse(self, text: str) -> Any:
        """
        텍스트를 파싱

        Args:
            text: LLM 출력 텍스트

        Returns:
            파싱된 결과

        Raises:
            OutputParserException: 파싱 실패 시
        """
        pass

    def get_format_instructions(self) -> str:
        """
        LLM에게 전달할 출력 형식 지침

        Returns:
            형식 지침 문자열
        """
        return ""

    @abstractmethod
    def get_output_type(self) -> str:
        """출력 타입 설명"""
        pass
