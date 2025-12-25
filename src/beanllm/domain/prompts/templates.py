"""
Prompts Templates - 프롬프트 템플릿 구현체
"""

import re
from typing import Any, Callable, Dict, List, Optional, Union

from .base import BasePromptTemplate
from .enums import TemplateFormat
from .types import ChatMessage, PromptExample


class PromptTemplate(BasePromptTemplate):
    """
    기본 프롬프트 템플릿

    Examples:
        >>> template = PromptTemplate(
        ...     template="Translate {text} to {language}",
        ...     input_variables=["text", "language"]
        ... )
        >>> template.format(text="Hello", language="Korean")
        'Translate Hello to Korean'
    """

    def __init__(
        self,
        template: str,
        input_variables: Optional[List[str]] = None,
        template_format: TemplateFormat = TemplateFormat.F_STRING,
        validate_template: bool = True,
        partial_variables: Optional[Dict[str, Any]] = None,
    ):
        self.template = template
        self.template_format = template_format
        self.partial_variables = partial_variables or {}

        # 자동으로 input_variables 추출
        if input_variables is None:
            self.input_variables = self._extract_variables()
        else:
            self.input_variables = input_variables

        # 템플릿 검증
        if validate_template:
            self._validate_template()

    def _extract_variables(self) -> List[str]:
        """템플릿에서 변수 자동 추출"""
        if self.template_format == TemplateFormat.F_STRING:
            # {variable} 형식
            pattern = r"\{(\w+)\}"
        elif self.template_format == TemplateFormat.JINJA2:
            # {{ variable }} 형식
            pattern = r"\{\{\s*(\w+)\s*\}\}"
        else:
            # {{variable}} 형식 (Mustache)
            pattern = r"\{\{(\w+)\}\}"

        matches = re.findall(pattern, self.template)
        return list(set(matches))  # 중복 제거

    def _validate_template(self) -> None:
        """템플릿 유효성 검증"""
        # 추출된 변수와 명시된 변수가 일치하는지 확인
        extracted = set(self._extract_variables())
        declared = set(self.input_variables)

        if extracted != declared:
            raise ValueError(
                f"Template variables mismatch. Extracted: {extracted}, Declared: {declared}"
            )

    def format(self, **kwargs) -> str:
        """템플릿 포맷팅"""
        # partial_variables와 병합
        all_vars = {**self.partial_variables, **kwargs}

        # 입력 검증 (partial 제외)
        required_vars = [v for v in self.input_variables if v not in self.partial_variables]

        missing = set(required_vars) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        # 포맷팅
        if self.template_format == TemplateFormat.F_STRING:
            return self.template.format(**all_vars)
        elif self.template_format == TemplateFormat.JINJA2:
            # Jinja2 지원 (선택적)
            try:
                from jinja2 import Template

                return Template(self.template).render(**all_vars)
            except ImportError:
                # Jinja2 없으면 간단한 치환
                result = self.template
                for key, value in all_vars.items():
                    result = result.replace(f"{{{{ {key} }}}}", str(value))
                return result
        else:
            # Mustache 스타일
            result = self.template
            for key, value in all_vars.items():
                result = result.replace(f"{{{{{key}}}}}", str(value))
            return result

    def get_input_variables(self) -> List[str]:
        """입력 변수 목록 반환 (partial 제외)"""
        return [v for v in self.input_variables if v not in self.partial_variables]

    def partial(self, **kwargs) -> "PromptTemplate":
        """일부 변수를 미리 채운 새 템플릿 반환"""
        new_partial = {**self.partial_variables, **kwargs}
        return PromptTemplate(
            template=self.template,
            input_variables=self.input_variables,
            template_format=self.template_format,
            validate_template=False,
            partial_variables=new_partial,
        )


class FewShotPromptTemplate(BasePromptTemplate):
    """
    Few-shot 프롬프트 템플릿

    Examples:
        >>> examples = [
        ...     PromptExample(input="2+2", output="4"),
        ...     PromptExample(input="3+3", output="6")
        ... ]
        >>> template = FewShotPromptTemplate(
        ...     examples=examples,
        ...     example_template=PromptTemplate(
        ...         template="Q: {input}\\nA: {output}",
        ...         input_variables=["input", "output"]
        ...     ),
        ...     prefix="Solve the math problem:",
        ...     suffix="Q: {input}\\nA:",
        ...     input_variables=["input"]
        ... )
    """

    def __init__(
        self,
        examples: List[PromptExample],
        example_template: PromptTemplate,
        prefix: str = "",
        suffix: str = "",
        input_variables: Optional[List[str]] = None,
        example_separator: str = "\n\n",
        max_examples: Optional[int] = None,
        example_selector: Optional[Callable] = None,
    ):
        self.examples = examples
        self.example_template = example_template
        self.prefix = prefix
        self.suffix = suffix
        self.example_separator = example_separator
        self.max_examples = max_examples
        self.example_selector = example_selector

        # suffix에서 input_variables 추출
        if input_variables is None:
            self.input_variables = self._extract_suffix_variables()
        else:
            self.input_variables = input_variables

    def _extract_suffix_variables(self) -> List[str]:
        """suffix에서 변수 추출"""
        pattern = r"\{(\w+)\}"
        matches = re.findall(pattern, self.suffix)
        return list(set(matches))

    def format(self, **kwargs) -> str:
        """Few-shot 프롬프트 생성"""
        # 예제 선택
        if self.example_selector:
            selected_examples = self.example_selector(self.examples, kwargs)
        else:
            selected_examples = self.examples

        # max_examples 제한
        if self.max_examples:
            selected_examples = selected_examples[: self.max_examples]

        # 예제 포맷팅
        formatted_examples = []
        for example in selected_examples:
            formatted = self.example_template.format(input=example.input, output=example.output)
            formatted_examples.append(formatted)

        # 전체 프롬프트 조립
        parts = []

        if self.prefix:
            parts.append(self.prefix)

        if formatted_examples:
            parts.append(self.example_separator.join(formatted_examples))

        if self.suffix:
            parts.append(self.suffix.format(**kwargs))

        return "\n\n".join(parts)

    def get_input_variables(self) -> List[str]:
        return self.input_variables

    def add_example(self, example: PromptExample) -> None:
        """예제 추가"""
        self.examples.append(example)


class ChatPromptTemplate(BasePromptTemplate):
    """
    채팅 프롬프트 템플릿

    Examples:
        >>> template = ChatPromptTemplate.from_messages([
        ...     ("system", "You are a helpful {role}"),
        ...     ("user", "{input}")
        ... ])
        >>> messages = template.format_messages(role="assistant", input="Hello")
    """

    def __init__(
        self, messages: List[Union[ChatMessage, tuple]], input_variables: Optional[List[str]] = None
    ):
        # tuple을 ChatMessage로 변환
        self.messages = []
        for msg in messages:
            if isinstance(msg, tuple):
                role, content = msg[0], msg[1]
                name = msg[2] if len(msg) > 2 else None
                self.messages.append(ChatMessage(role=role, content=content, name=name))
            else:
                self.messages.append(msg)

        # input_variables 자동 추출
        if input_variables is None:
            self.input_variables = self._extract_variables()
        else:
            self.input_variables = input_variables

    def _extract_variables(self) -> List[str]:
        """모든 메시지에서 변수 추출"""
        variables = set()
        for msg in self.messages:
            pattern = r"\{(\w+)\}"
            matches = re.findall(pattern, msg.content)
            variables.update(matches)
        return list(variables)

    def format(self, **kwargs) -> str:
        """문자열로 포맷팅 (간단한 표현)"""
        formatted_messages = self.format_messages(**kwargs)
        return "\n\n".join(f"{msg.role.upper()}: {msg.content}" for msg in formatted_messages)

    def format_messages(self, **kwargs) -> List[ChatMessage]:
        """ChatMessage 리스트로 포맷팅"""
        formatted = []
        for msg in self.messages:
            content = msg.content.format(**kwargs)
            formatted.append(
                ChatMessage(role=msg.role, content=content, name=msg.name, metadata=msg.metadata)
            )
        return formatted

    def to_dict_messages(self, **kwargs) -> List[Dict[str, Any]]:
        """딕셔너리 리스트로 포맷팅 (API 호출용)"""
        messages = self.format_messages(**kwargs)
        return [msg.to_dict() for msg in messages]

    def get_input_variables(self) -> List[str]:
        return self.input_variables

    @classmethod
    def from_messages(cls, messages: List[Union[tuple, ChatMessage]]) -> "ChatPromptTemplate":
        """메시지 리스트로부터 생성"""
        return cls(messages=messages)

    @classmethod
    def from_template(cls, template: str, role: str = "user") -> "ChatPromptTemplate":
        """단일 템플릿으로부터 생성"""
        return cls(messages=[(role, template)])


class SystemMessageTemplate(PromptTemplate):
    """
    시스템 메시지 템플릿

    Examples:
        >>> template = SystemMessageTemplate(
        ...     template="You are a {role} that {task}",
        ...     input_variables=["role", "task"]
        ... )
    """

    def __init__(self, template: str, **kwargs):
        super().__init__(template=template, **kwargs)
        self.role = "system"

    def to_message(self, **kwargs) -> ChatMessage:
        """ChatMessage로 변환"""
        content = self.format(**kwargs)
        return ChatMessage(role="system", content=content)
