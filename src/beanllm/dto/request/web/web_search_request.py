"""
WebSearchRequest - Web Search 요청 DTO
책임: Web Search 요청 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class WebSearchRequest:
    """
    Web Search 요청 DTO

    책임:
    - 데이터 구조 정의만
    - 검증 없음 (Handler에서 처리)
    - 비즈니스 로직 없음 (Service에서 처리)
    """

    query: str
    engine: Optional[str] = None  # "google", "bing", "duckduckgo"
    max_results: int = 10
    max_scrape: int = 3  # search_and_scrape용
    google_api_key: Optional[str] = None
    google_search_engine_id: Optional[str] = None
    bing_api_key: Optional[str] = None
    # 엔진별 옵션
    language: Optional[str] = None  # Google용
    safe: Optional[str] = None  # Google용
    market: Optional[str] = None  # Bing용
    safe_search: Optional[str] = None  # Bing/DuckDuckGo용
    region: Optional[str] = None  # DuckDuckGo용
    extra_params: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """기본값 설정"""
        if self.extra_params is None:
            self.extra_params = {}
