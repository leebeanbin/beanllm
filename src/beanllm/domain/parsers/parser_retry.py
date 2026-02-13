"""
Retry Output Parser - 재시도 파서
"""

from __future__ import annotations

from typing import Any, Optional

from beanllm.domain.parsers.base import BaseOutputParser
from beanllm.domain.parsers.exceptions import OutputParserException
from beanllm.utils.constants import DEFAULT_MAX_RETRIES


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
        max_retries: int = DEFAULT_MAX_RETRIES,
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
        from beanllm.utils.logging import get_logger

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
