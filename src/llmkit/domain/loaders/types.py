"""
Loaders Types - 문서 데이터 타입
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Document:
    """
    문서 클래스

    참고: LangChain의 Document 구조에서 영감을 받았습니다.
    """

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 편의 속성
    @property
    def page_content(self) -> str:
        """LangChain 호환 속성"""
        return self.content

    def __str__(self) -> str:
        return f"Document(content={self.content[:100]}..., metadata={self.metadata})"
