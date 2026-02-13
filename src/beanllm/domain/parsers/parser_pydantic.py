"""
Pydantic Output Parser - Pydantic 모델 기반 파서
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Dict, Type

from beanllm.domain.parsers.base import BaseOutputParser
from beanllm.domain.parsers.exceptions import OutputParserException

if TYPE_CHECKING:
    from pydantic import BaseModel, ValidationError

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

        example: Dict[str, Any] = {}
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
