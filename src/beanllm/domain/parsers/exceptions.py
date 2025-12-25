"""
Parsers Exceptions - 파서 예외
"""

from typing import Optional


class OutputParserException(Exception):
    """Output Parser 예외"""

    def __init__(self, message: str, llm_output: Optional[str] = None):
        super().__init__(message)
        self.llm_output = llm_output
