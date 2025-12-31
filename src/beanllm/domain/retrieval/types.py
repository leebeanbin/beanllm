"""
Retrieval Types - 검색 관련 타입 정의
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RerankResult:
    """
    재순위화 결과

    Attributes:
        text: 원본 텍스트
        score: 재순위화 점수 (높을수록 관련성 높음)
        index: 원본 리스트에서의 인덱스
        metadata: 추가 메타데이터
    """

    text: str
    score: float
    index: int
    metadata: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        return f"RerankResult(index={self.index}, score={self.score:.4f}, text={self.text[:50]}...)"


@dataclass
class SearchResult:
    """
    검색 결과

    Attributes:
        text: 검색된 텍스트
        score: 검색 점수
        metadata: 메타데이터 (source, page, 등)
    """

    text: str
    score: float
    metadata: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        return f"SearchResult(score={self.score:.4f}, text={self.text[:50]}...)"
