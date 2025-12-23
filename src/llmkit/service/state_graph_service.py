"""
IStateGraphService - StateGraph 서비스 인터페이스
SOLID 원칙:
- ISP: StateGraph 관련 메서드만 포함
- DIP: 인터페이스에 의존
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator

from ..dto.request.state_graph_request import StateGraphRequest
from ..dto.response.state_graph_response import StateGraphResponse


class IStateGraphService(ABC):
    """
    StateGraph 서비스 인터페이스

    책임:
    - StateGraph 비즈니스 로직 정의만
    - 검증, 에러 처리 없음 (Handler에서 처리)

    SOLID:
    - ISP: StateGraph 관련 메서드만 (작은 인터페이스)
    - DIP: 구현체가 아닌 인터페이스에 의존
    """

    @abstractmethod
    async def invoke(self, request: StateGraphRequest) -> StateGraphResponse:
        """
        StateGraph 실행

        Args:
            request: StateGraph 요청 DTO

        Returns:
            StateGraphResponse: StateGraph 응답 DTO
        """
        pass

    @abstractmethod
    def stream(self, request: StateGraphRequest) -> Iterator[tuple[str, Dict[str, Any]]]:
        """
        StateGraph 스트리밍 실행

        Args:
            request: StateGraph 요청 DTO

        Yields:
            (node_name, state) 튜플
        """
        pass
