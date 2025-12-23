"""
IWebSearchService - Web Search 서비스 인터페이스
SOLID 원칙:
- ISP: Web Search 관련 메서드만 포함
- DIP: 인터페이스에 의존
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ..dto.request.web_search_request import WebSearchRequest
from ..dto.response.web_search_response import WebSearchResponse


class IWebSearchService(ABC):
    """
    Web Search 서비스 인터페이스

    책임:
    - Web Search 비즈니스 로직 정의만
    - 검증, 에러 처리 없음 (Handler에서 처리)

    SOLID:
    - ISP: Web Search 관련 메서드만 (작은 인터페이스)
    - DIP: 구현체가 아닌 인터페이스에 의존
    """

    @abstractmethod
    async def search(self, request: WebSearchRequest) -> WebSearchResponse:
        """
        웹 검색 실행

        Args:
            request: Web Search 요청 DTO

        Returns:
            WebSearchResponse: Web Search 응답 DTO
        """
        pass

    @abstractmethod
    async def search_and_scrape(self, request: WebSearchRequest) -> List[Dict[str, Any]]:
        """
        검색 후 상위 결과 스크래핑

        Args:
            request: Web Search 요청 DTO

        Returns:
            스크래핑된 콘텐츠 리스트
        """
        pass
