"""
WebSearchRequest - Web Search 요청 DTO
책임: Web Search 요청 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True, kw_only=True)
class WebSearchRequest:
    """
    Web Search 요청 DTO

    책임:
    - 데이터 구조 정의만
    - 검증 없음 (Handler에서 처리)
    - 비즈니스 로직 없음 (Service에서 처리)
    """

    query: str
    engine: Optional[str] = None
    max_results: int = 10
    max_scrape: int = 3
    google_api_key: Optional[str] = None
    google_search_engine_id: Optional[str] = None
    bing_api_key: Optional[str] = None
    # 엔진별 옵션
    language: Optional[str] = None
    safe: Optional[str] = None
    market: Optional[str] = None
    safe_search: Optional[str] = None
    region: Optional[str] = None
    extra_params: dict[str, object] = field(default_factory=dict)
