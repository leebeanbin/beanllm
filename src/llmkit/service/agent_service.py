"""
IAgentService - 에이전트 서비스 인터페이스
SOLID 원칙:
- ISP: 에이전트 관련 메서드만 포함
- DIP: 인터페이스에 의존
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..dto.request.agent_request import AgentRequest
from ..dto.response.agent_response import AgentResponse


class IAgentService(ABC):
    """
    에이전트 서비스 인터페이스

    책임:
    - 에이전트 비즈니스 로직 정의만
    - 검증, 에러 처리 없음 (Handler에서 처리)

    SOLID:
    - ISP: 에이전트 관련 메서드만 (작은 인터페이스)
    - DIP: 구현체가 아닌 인터페이스에 의존
    """

    @abstractmethod
    async def run(self, request: AgentRequest) -> AgentResponse:
        """
        에이전트 실행

        Args:
            request: 에이전트 요청 DTO

        Returns:
            AgentResponse: 에이전트 응답 DTO

        책임:
            - 에이전트 비즈니스 로직만 (ReAct 패턴 실행 등)
            - 검증 없음 (Handler에서 처리)
            - 에러 처리 없음 (Handler에서 처리)
        """
        pass
