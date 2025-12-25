"""
Prompts Composer - 프롬프트 조합 도구
"""

from typing import List

from .base import BasePromptTemplate
from .templates import PromptTemplate


class PromptComposer:
    """
    프롬프트 조합 도구

    여러 템플릿을 조합하여 복잡한 프롬프트 생성
    """

    def __init__(self):
        self.templates: List[BasePromptTemplate] = []
        self.separator = "\n\n"

    def add_template(self, template: BasePromptTemplate) -> "PromptComposer":
        """템플릿 추가"""
        self.templates.append(template)
        return self

    def add_text(self, text: str) -> "PromptComposer":
        """고정 텍스트 추가"""
        template = PromptTemplate(template=text, input_variables=[])
        self.templates.append(template)
        return self

    def compose(self, **kwargs) -> str:
        """모든 템플릿 조합"""
        parts = []
        for template in self.templates:
            # 필요한 변수만 전달
            required_vars = template.get_input_variables()
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in required_vars}
            parts.append(template.format(**filtered_kwargs))

        return self.separator.join(parts)

    def set_separator(self, separator: str) -> "PromptComposer":
        """구분자 설정"""
        self.separator = separator
        return self
