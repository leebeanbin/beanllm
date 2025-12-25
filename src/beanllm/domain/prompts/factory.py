"""
Prompts Factory - 프롬프트 생성 팩토리 함수
"""

from typing import List, Optional, Union

from .templates import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate
from .types import ChatMessage, PromptExample


def create_prompt_template(
    template: str, input_variables: Optional[List[str]] = None, **kwargs
) -> PromptTemplate:
    """간편한 PromptTemplate 생성"""
    return PromptTemplate(template=template, input_variables=input_variables, **kwargs)


def create_chat_template(messages: List[Union[tuple, ChatMessage]]) -> ChatPromptTemplate:
    """간편한 ChatPromptTemplate 생성"""
    return ChatPromptTemplate.from_messages(messages)


def create_few_shot_template(
    examples: List[PromptExample], example_format: str, prefix: str = "", suffix: str = "", **kwargs
) -> FewShotPromptTemplate:
    """간편한 FewShotPromptTemplate 생성"""
    example_template = PromptTemplate(template=example_format, input_variables=["input", "output"])

    return FewShotPromptTemplate(
        examples=examples, example_template=example_template, prefix=prefix, suffix=suffix, **kwargs
    )
