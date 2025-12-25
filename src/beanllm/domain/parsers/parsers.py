"""
Parsers Implementations - 파서 구현체들
"""

import json
import re
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

from .base import BaseOutputParser
from .exceptions import OutputParserException

if TYPE_CHECKING:
    from pydantic import BaseModel, ValidationError

# Pydantic optional import
try:
    from pydantic import BaseModel, ValidationError

    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = None  # type: ignore
    ValidationError = None  # type: ignore


class PydanticOutputParser(BaseOutputParser):
    """
    Pydantic 모델 기반 파서

    LLM 출력을 Pydantic 모델로 변환

    Example:
        ```python
        from beanllm.domain.parsers import PydanticOutputParser
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int
            email: str

        parser = PydanticOutputParser(pydantic_object=Person)

        # LLM에게 형식 지침 전달
        instructions = parser.get_format_instructions()
        prompt = f"Extract person info.\\n{instructions}\\n\\nText: John is 30 years old..."

        # 파싱
        person = parser.parse(llm_output)
        print(person.name)  # "John"
        print(person.age)   # 30
        ```
    """

    def __init__(self, pydantic_object: Type[BaseModel]):
        """
        Args:
            pydantic_object: Pydantic 모델 클래스

        Raises:
            ImportError: pydantic이 설치되지 않은 경우
        """
        if not HAS_PYDANTIC:
            raise ImportError(
                "pydantic is required for PydanticOutputParser. "
                "Install it with: pip install pydantic"
            )

        self.pydantic_object = pydantic_object

    def parse(self, text: str) -> BaseModel:
        """
        JSON 텍스트를 Pydantic 모델로 변환

        Args:
            text: JSON 형식의 텍스트

        Returns:
            Pydantic 모델 인스턴스

        Raises:
            OutputParserException: 파싱 실패 시
        """
        try:
            # JSON 추출 (코드 블록이나 추가 텍스트가 있을 수 있음)
            json_text = self._extract_json(text)

            # JSON 파싱
            data = json.loads(json_text)

            # Pydantic 모델 생성
            return self.pydantic_object(**data)

        except json.JSONDecodeError as e:
            raise OutputParserException(f"Failed to parse JSON: {e}", llm_output=text)
        except ValidationError as e:
            raise OutputParserException(f"Failed to validate Pydantic model: {e}", llm_output=text)
        except Exception as e:
            raise OutputParserException(f"Failed to parse output: {e}", llm_output=text)

    def _extract_json(self, text: str) -> str:
        """텍스트에서 JSON 추출"""
        # 코드 블록 제거 (```json ... ```)
        json_match = re.search(r"```(?:json)?\s*(\{.+?\})\s*```", text, re.DOTALL)
        if json_match:
            return json_match.group(1)

        # 중괄호로 둘러싸인 부분 찾기
        json_match = re.search(r"\{.+\}", text, re.DOTALL)
        if json_match:
            return json_match.group(0)

        # 그대로 반환
        return text.strip()

    def get_format_instructions(self) -> str:
        """출력 형식 지침"""
        schema = self.pydantic_object.model_json_schema()

        # 필드 정보 추출
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        fields_desc = []
        for field_name, field_info in properties.items():
            field_type = field_info.get("type", "string")
            is_required = field_name in required
            desc = field_info.get("description", "")

            req_mark = " (required)" if is_required else " (optional)"
            fields_desc.append(f"  - {field_name}: {field_type}{req_mark} - {desc}")

        fields_str = "\n".join(fields_desc)

        return f"""Output must be a valid JSON object with the following fields:
{fields_str}

Example format:
```json
{json.dumps(self._get_example_output(), indent=2)}
```

IMPORTANT: Return ONLY the JSON object, nothing else."""

    def _get_example_output(self) -> Dict[str, Any]:
        """예제 출력 생성"""
        schema = self.pydantic_object.model_json_schema()
        properties = schema.get("properties", {})

        example = {}
        for field_name, field_info in properties.items():
            field_type = field_info.get("type", "string")

            if field_type == "string":
                example[field_name] = "example_string"
            elif field_type == "integer":
                example[field_name] = 0
            elif field_type == "number":
                example[field_name] = 0.0
            elif field_type == "boolean":
                example[field_name] = True
            elif field_type == "array":
                example[field_name] = []
            elif field_type == "object":
                example[field_name] = {}
            else:
                example[field_name] = None

        return example

    def get_output_type(self) -> str:
        return f"Pydantic[{self.pydantic_object.__name__}]"


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
            return json.loads(json_text)

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


class DatetimeOutputParser(BaseOutputParser):
    """
    날짜/시간 파서

    Example:
        ```python
        from beanllm.domain.parsers import DatetimeOutputParser

        parser = DatetimeOutputParser(format="%Y-%m-%d %H:%M:%S")
        dt = parser.parse("2024-01-15 10:30:00")
        ```
    """

    def __init__(self, format: str = "%Y-%m-%d %H:%M:%S"):
        """
        Args:
            format: datetime.strptime 형식 문자열
        """
        self.format = format

    def parse(self, text: str) -> datetime:
        """
        텍스트를 datetime으로 변환

        Args:
            text: 날짜/시간 문자열

        Returns:
            datetime 객체

        Raises:
            OutputParserException: 파싱 실패 시
        """
        try:
            text = text.strip()
            # 코드 블록 제거
            text = re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()

            return datetime.strptime(text, self.format)

        except ValueError as e:
            raise OutputParserException(f"Failed to parse datetime: {e}", llm_output=text)

    def get_format_instructions(self) -> str:
        return f"""Output must be a datetime string in the format: {self.format}

Example:
{datetime.now().strftime(self.format)}

Return ONLY the datetime string, nothing else."""

    def get_output_type(self) -> str:
        return "datetime"


class EnumOutputParser(BaseOutputParser):
    """
    Enum 파서

    Example:
        ```python
        from enum import Enum
        from beanllm.domain.parsers import EnumOutputParser

        class Color(Enum):
            RED = "red"
            GREEN = "green"
            BLUE = "blue"

        parser = EnumOutputParser(enum_class=Color)
        color = parser.parse("red")  # Color.RED
        ```
    """

    def __init__(self, enum_class: Type[Enum]):
        """
        Args:
            enum_class: Enum 클래스
        """
        self.enum_class = enum_class

    def parse(self, text: str) -> Enum:
        """
        텍스트를 Enum으로 변환

        Args:
            text: Enum 값 문자열

        Returns:
            Enum 인스턴스

        Raises:
            OutputParserException: 파싱 실패 시
        """
        text = text.strip().lower()

        # 값으로 찾기
        for member in self.enum_class:
            if member.value.lower() == text:
                return member

        # 이름으로 찾기
        for member in self.enum_class:
            if member.name.lower() == text:
                return member

        # 실패
        valid_values = [m.value for m in self.enum_class]
        raise OutputParserException(
            f"Invalid enum value: {text}. Valid values: {valid_values}", llm_output=text
        )

    def get_format_instructions(self) -> str:
        valid_values = [m.value for m in self.enum_class]
        return f"""Output must be one of the following values:
{", ".join(valid_values)}

Return ONLY one of these values, nothing else."""

    def get_output_type(self) -> str:
        return f"Enum[{self.enum_class.__name__}]"


class BooleanOutputParser(BaseOutputParser):
    """
    Boolean 파서

    Example:
        ```python
        from beanllm.domain.parsers import BooleanOutputParser

        parser = BooleanOutputParser()
        result = parser.parse("yes")  # True
        result = parser.parse("no")   # False
        ```
    """

    TRUE_VALUES = {"true", "yes", "y", "1", "ok", "correct"}
    FALSE_VALUES = {"false", "no", "n", "0", "not ok", "incorrect"}

    def parse(self, text: str) -> bool:
        """
        텍스트를 boolean으로 변환

        Args:
            text: boolean 값 문자열

        Returns:
            bool

        Raises:
            OutputParserException: 파싱 실패 시
        """
        text = text.strip().lower()

        if text in self.TRUE_VALUES:
            return True
        elif text in self.FALSE_VALUES:
            return False
        else:
            raise OutputParserException(f"Cannot parse as boolean: {text}", llm_output=text)

    def get_format_instructions(self) -> str:
        return """Output must be a boolean value.

Valid values for True: true, yes, y, 1
Valid values for False: false, no, n, 0

Return ONLY one of these values, nothing else."""

    def get_output_type(self) -> str:
        return "bool"


class RetryOutputParser(BaseOutputParser):
    """
    재시도 파서

    파싱 실패 시 LLM에게 다시 요청

    Example:
        ```python
        from beanllm import Client
        from beanllm.domain.parsers import RetryOutputParser, JSONOutputParser

        client = Client(model="gpt-4o-mini")
        base_parser = JSONOutputParser()
        retry_parser = RetryOutputParser(
            parser=base_parser,
            client=client,
            max_retries=3
        )

        # 파싱 실패 시 자동으로 재시도
        result = await retry_parser.parse_with_retry("invalid json...")
        ```
    """

    def __init__(
        self,
        parser: BaseOutputParser,
        client: Any,  # Client 타입, circular import 방지
        max_retries: int = 3,
    ):
        """
        Args:
            parser: 기본 파서
            client: LLM Client
            max_retries: 최대 재시도 횟수
        """
        self.parser = parser
        self.client = client
        self.max_retries = max_retries

    def parse(self, text: str) -> Any:
        """기본 파서로 파싱"""
        return self.parser.parse(text)

    async def parse_with_retry(self, text: str, prompt_template: Optional[str] = None) -> Any:
        """
        파싱 재시도

        Args:
            text: 파싱할 텍스트
            prompt_template: 재시도 프롬프트 템플릿

        Returns:
            파싱된 결과

        Raises:
            OutputParserException: 최대 재시도 초과 시
        """
        from ...utils.logger import get_logger

        logger = get_logger(__name__)

        for attempt in range(self.max_retries + 1):
            try:
                return self.parser.parse(text)

            except OutputParserException as e:
                if attempt >= self.max_retries:
                    raise OutputParserException(
                        f"Failed after {self.max_retries} retries: {e}", llm_output=text
                    )

                # 재시도 프롬프트
                if prompt_template is None:
                    prompt_template = self._get_default_retry_prompt()

                retry_prompt = prompt_template.format(
                    completion=text,
                    error=str(e),
                    instructions=self.parser.get_format_instructions(),
                )

                # LLM 재요청
                logger.info(f"Retry attempt {attempt + 1}/{self.max_retries}")
                response = await self.client.chat([{"role": "user", "content": retry_prompt}])

                text = response.content

        # Should not reach here
        raise OutputParserException("Unexpected error in retry logic", llm_output=text)

    def _get_default_retry_prompt(self) -> str:
        return """Your previous output was invalid:

{completion}

Error: {error}

Please fix the output according to these instructions:
{instructions}"""

    def get_format_instructions(self) -> str:
        return self.parser.get_format_instructions()

    def get_output_type(self) -> str:
        return f"Retry[{self.parser.get_output_type()}]"
