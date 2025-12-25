"""
IGraphService - Graph 서비스 인터페이스
SOLID 원칙:
- ISP: Graph 관련 메서드만 포함
- DIP: 인터페이스에 의존
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..dto.request.graph_request import GraphRequest
from ..dto.response.graph_response import GraphResponse


class IGraphService(ABC):
    """
    Graph 서비스 인터페이스

    책임:
    - Graph 비즈니스 로직 정의만
    - 검증, 에러 처리 없음 (Handler에서 처리)

    SOLID:
    - ISP: Graph 관련 메서드만 (작은 인터페이스)
    - DIP: 구현체가 아닌 인터페이스에 의존
    """

    @abstractmethod
    async def run_graph(self, request: GraphRequest) -> GraphResponse:
        """
        Graph 실행

        Args:
            request: Graph 요청 DTO

        Returns:
            GraphResponse: Graph 응답 DTO

        책임:
            - Graph 비즈니스 로직만
            - 검증 없음 (Handler에서 처리)
            - 에러 처리 없음 (Handler에서 처리)
        """
        pass
