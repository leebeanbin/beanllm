"""
Web Search Types - 검색 결과 및 응답 타입
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class SearchResult:
    """
    검색 결과 하나

    Attributes:
        title: 제목
        url: URL
        snippet: 요약
        source: 출처 (google, bing, duckduckgo 등)
        score: 관련도 점수 (0-1)
        published_date: 발행일 (선택)
        metadata: 추가 메타데이터
    """

    title: str
    url: str
    snippet: str
    source: str = "unknown"
    score: float = 0.0
    published_date: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.source}] {self.title}\n{self.url}\n{self.snippet[:100]}..."


@dataclass
class SearchResponse:
    """
    검색 응답

    Attributes:
        query: 검색 쿼리
        results: 검색 결과 리스트
        total_results: 전체 결과 수 (추정)
        search_time: 검색 소요 시간 (초)
        engine: 사용한 검색 엔진
        metadata: 추가 메타데이터
    """

    query: str
    results: List[SearchResult]
    total_results: Optional[int] = None
    search_time: float = 0.0
    engine: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)
