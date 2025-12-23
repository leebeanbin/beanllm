"""
WebSearchResponse - Web Search 응답 DTO
책임: Web Search 응답 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...domain.web_search import SearchResult


@dataclass
class WebSearchResponse:
    """
    Web Search 응답 DTO

    책임:
    - 데이터 구조 정의만
    - 변환 로직 없음
    """

    query: str
    results: List[SearchResult]
    total_results: Optional[int] = None
    search_time: float = 0.0
    engine: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
