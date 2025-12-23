"""
Prompts Base - 프롬프트 베이스 클래스
"""

from abc import ABC, abstractmethod
from typing import List


class BasePromptTemplate(ABC):
    """프롬프트 템플릿 베이스 클래스"""

    @abstractmethod
    def format(self, **kwargs) -> str:
        """템플릿 포맷팅"""
        pass

    @abstractmethod
    def get_input_variables(self) -> List[str]:
        """입력 변수 목록 반환"""
        pass

    def validate_input(self, **kwargs) -> None:
        """입력 검증"""
        required = set(self.get_input_variables())
        provided = set(kwargs.keys())

        missing = required - provided
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        extra = provided - required
        if extra:
            raise ValueError(f"Unexpected variables: {extra}")
