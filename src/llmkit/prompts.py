"""
llmkit.prompts - Prompt Template System
프롬프트 템플릿 시스템

이 모듈은 재사용 가능한 프롬프트 템플릿을 제공합니다.
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union


class TemplateFormat(Enum):
    """템플릿 포맷"""

    F_STRING = "f-string"  # {variable}
    JINJA2 = "jinja2"  # {{ variable }}
    MUSTACHE = "mustache"  # {{variable}}


@dataclass
class PromptExample:
    """Few-shot 예제"""

    input: str
    output: str
    metadata: Dict[str, Any] = field(default_factory=dict)


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
                f"Template variables mismatch. " f"Extracted: {extracted}, Declared: {declared}"
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


@dataclass
class ChatMessage:
    """채팅 메시지"""

    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        result = {"role": self.role, "content": self.content}
        if self.name:
            result["name"] = self.name
        return result


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


class PromptOptimizer:
    """
    프롬프트 최적화 도구

    프롬프트를 자동으로 개선합니다.
    """

    @staticmethod
    def add_instructions(prompt: str, instructions: List[str]) -> str:
        """명령어 추가"""
        instruction_text = "\n".join(f"- {inst}" for inst in instructions)
        return f"{prompt}\n\nInstructions:\n{instruction_text}"

    @staticmethod
    def add_constraints(prompt: str, constraints: List[str]) -> str:
        """제약조건 추가"""
        constraint_text = "\n".join(f"- {const}" for const in constraints)
        return f"{prompt}\n\nConstraints:\n{constraint_text}"

    @staticmethod
    def add_output_format(
        prompt: str, format_description: str, example: Optional[str] = None
    ) -> str:
        """출력 포맷 명시"""
        result = f"{prompt}\n\nOutput Format:\n{format_description}"
        if example:
            result += f"\n\nExample Output:\n{example}"
        return result

    @staticmethod
    def add_json_output(prompt: str, schema: Dict[str, Any]) -> str:
        """JSON 출력 형식 추가"""
        schema_str = json.dumps(schema, indent=2)
        return f"{prompt}\n\nPlease respond in JSON format:\n{schema_str}"

    @staticmethod
    def add_thinking_process(prompt: str) -> str:
        """사고 과정 요청 추가"""
        return (
            f"{prompt}\n\n"
            "Please think step-by-step:\n"
            "1. Analyze the problem\n"
            "2. Consider possible solutions\n"
            "3. Choose the best approach\n"
            "4. Provide your answer"
        )

    @staticmethod
    def add_role_context(prompt: str, role: str, expertise: List[str]) -> str:
        """역할 컨텍스트 추가"""
        expertise_text = ", ".join(expertise)
        role_prompt = f"You are a {role} with expertise in {expertise_text}.\n\n" f"{prompt}"
        return role_prompt


# ===== 유틸리티 함수 =====


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


# ===== 사전 정의된 템플릿 =====


class PredefinedTemplates:
    """자주 사용되는 템플릿 모음"""

    @staticmethod
    def translation() -> PromptTemplate:
        """번역 템플릿"""
        return PromptTemplate(
            template="Translate the following text from {source_lang} to {target_lang}:\n\n{text}",
            input_variables=["source_lang", "target_lang", "text"],
        )

    @staticmethod
    def summarization() -> PromptTemplate:
        """요약 템플릿"""
        return PromptTemplate(
            template="Summarize the following text in {max_sentences} sentences:\n\n{text}",
            input_variables=["text", "max_sentences"],
        )

    @staticmethod
    def question_answering() -> ChatPromptTemplate:
        """QA 템플릿"""
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that answers questions based on the given context.",
                ),
                ("user", "Context: {context}\n\nQuestion: {question}\n\nAnswer:"),
            ]
        )

    @staticmethod
    def code_generation() -> ChatPromptTemplate:
        """코드 생성 템플릿"""
        return ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert {language} programmer."),
                ("user", "Write {language} code to {task}.\n\nRequirements:\n{requirements}"),
            ]
        )

    @staticmethod
    def chain_of_thought() -> PromptTemplate:
        """Chain-of-Thought 템플릿"""
        return PromptTemplate(
            template=(
                "{question}\n\n"
                "Let's think step by step:\n"
                "1. First, let's identify what we know\n"
                "2. Next, let's determine what we need to find\n"
                "3. Then, let's work through the solution\n"
                "4. Finally, let's verify our answer"
            ),
            input_variables=["question"],
        )

    @staticmethod
    def react_agent() -> ChatPromptTemplate:
        """ReAct Agent 템플릿"""
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a helpful assistant that uses tools to answer questions.\n"
                        "Use the following format:\n\n"
                        "Thought: Consider what to do\n"
                        "Action: The action to take\n"
                        "Observation: The result of the action\n"
                        "... (repeat as needed)\n"
                        "Final Answer: The final answer"
                    ),
                ),
                ("user", "{input}\n\nAvailable tools: {tools}"),
            ]
        )


# ===== 예제 선택기 =====


class ExampleSelector:
    """Few-shot 예제 선택 전략"""

    @staticmethod
    def similarity_based(
        examples: List[PromptExample],
        input_data: Dict[str, Any],
        top_k: int = 3,
        similarity_fn: Optional[Callable] = None,
    ) -> List[PromptExample]:
        """유사도 기반 예제 선택"""
        if similarity_fn is None:
            # 기본: 간단한 문자열 유사도
            def default_similarity(ex1: str, ex2: str) -> float:
                # Jaccard similarity
                set1 = set(ex1.lower().split())
                set2 = set(ex2.lower().split())
                if not set1 or not set2:
                    return 0.0
                intersection = set1 & set2
                union = set1 | set2
                return len(intersection) / len(union)

            similarity_fn = default_similarity

        # 입력과 각 예제의 유사도 계산
        input_text = str(input_data.get("input", ""))
        scored_examples = []

        for example in examples:
            score = similarity_fn(input_text, example.input)
            scored_examples.append((score, example))

        # 점수 기준 정렬
        scored_examples.sort(reverse=True, key=lambda x: x[0])

        # top_k 반환
        return [ex for _, ex in scored_examples[:top_k]]

    @staticmethod
    def length_based(examples: List[PromptExample], max_length: int) -> List[PromptExample]:
        """길이 제한 기반 예제 선택"""
        selected = []
        current_length = 0

        for example in examples:
            example_length = len(example.input) + len(example.output)
            if current_length + example_length <= max_length:
                selected.append(example)
                current_length += example_length
            else:
                break

        return selected

    @staticmethod
    def random(examples: List[PromptExample], k: int) -> List[PromptExample]:
        """랜덤 선택"""
        import random

        return random.sample(examples, min(k, len(examples)))


# ===== 고급 기능 =====


class PromptVersioning:
    """프롬프트 버전 관리"""

    def __init__(self):
        self.versions: Dict[str, List[tuple]] = {}  # name -> [(version, template)]

    def save(self, name: str, template: BasePromptTemplate, version: str) -> None:
        """템플릿 저장"""
        if name not in self.versions:
            self.versions[name] = []
        self.versions[name].append((version, template))

    def load(self, name: str, version: Optional[str] = None) -> BasePromptTemplate:
        """템플릿 로드"""
        if name not in self.versions:
            raise ValueError(f"Template '{name}' not found")

        if version is None:
            # 최신 버전 반환
            return self.versions[name][-1][1]

        # 특정 버전 찾기
        for ver, template in self.versions[name]:
            if ver == version:
                return template

        raise ValueError(f"Version '{version}' not found for template '{name}'")

    def list_versions(self, name: str) -> List[str]:
        """템플릿의 모든 버전 나열"""
        if name not in self.versions:
            return []
        return [ver for ver, _ in self.versions[name]]


class PromptCache:
    """프롬프트 캐시 (성능 최적화)"""

    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, str] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[str]:
        """캐시에서 가져오기"""
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def set(self, key: str, value: str) -> None:
        """캐시에 저장"""
        if len(self.cache) >= self.max_size:
            # LRU-like: 첫 번째 항목 제거
            first_key = next(iter(self.cache))
            del self.cache[first_key]

        self.cache[key] = value

    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "max_size": self.max_size,
        }

    def clear(self) -> None:
        """캐시 초기화"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0


# 전역 캐시 인스턴스
_global_cache = PromptCache()


def get_cached_prompt(template: BasePromptTemplate, use_cache: bool = True, **kwargs) -> str:
    """캐시를 사용한 프롬프트 생성"""
    if not use_cache:
        return template.format(**kwargs)

    # 캐시 키 생성
    cache_key = f"{id(template)}:{json.dumps(kwargs, sort_keys=True)}"

    # 캐시 확인
    cached = _global_cache.get(cache_key)
    if cached is not None:
        return cached

    # 생성 및 캐시 저장
    result = template.format(**kwargs)
    _global_cache.set(cache_key, result)

    return result


def get_cache_stats() -> Dict[str, Any]:
    """전역 캐시 통계"""
    return _global_cache.get_stats()


def clear_cache() -> None:
    """전역 캐시 초기화"""
    _global_cache.clear()
