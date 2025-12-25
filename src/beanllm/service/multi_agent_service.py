"""
IMultiAgentService - Multi-Agent 서비스 인터페이스
SOLID 원칙:
- ISP: Multi-Agent 관련 메서드만 포함
- DIP: 인터페이스에 의존
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..dto.request.multi_agent_request import MultiAgentRequest
from ..dto.response.multi_agent_response import MultiAgentResponse


class IMultiAgentService(ABC):
    """
    Multi-Agent 서비스 인터페이스

    책임:
    - Multi-Agent 비즈니스 로직 정의만
    - 검증, 에러 처리 없음 (Handler에서 처리)

    SOLID:
    - ISP: Multi-Agent 관련 메서드만 (작은 인터페이스)
    - DIP: 구현체가 아닌 인터페이스에 의존
    """

    @abstractmethod
    async def execute_sequential(self, request: MultiAgentRequest) -> MultiAgentResponse:
        """
        순차 실행

        Args:
            request: Multi-Agent 요청 DTO

        Returns:
            MultiAgentResponse: Multi-Agent 응답 DTO
        """
        pass

    @abstractmethod
    async def execute_parallel(self, request: MultiAgentRequest) -> MultiAgentResponse:
        """
        병렬 실행

        Args:
            request: Multi-Agent 요청 DTO

        Returns:
            MultiAgentResponse: Multi-Agent 응답 DTO
        """
        pass

    @abstractmethod
    async def execute_hierarchical(self, request: MultiAgentRequest) -> MultiAgentResponse:
        """
        계층적 실행

        Args:
            request: Multi-Agent 요청 DTO

        Returns:
            MultiAgentResponse: Multi-Agent 응답 DTO
        """
        pass

    @abstractmethod
    async def execute_debate(self, request: MultiAgentRequest) -> MultiAgentResponse:
        """
        토론 실행

        Args:
            request: Multi-Agent 요청 DTO

        Returns:
            MultiAgentResponse: Multi-Agent 응답 DTO
        """
        pass

    async def execute(self, request: MultiAgentRequest) -> MultiAgentResponse:
        """
        통합 실행 메서드 (Strategy 패턴)

        Args:
            request: Multi-Agent 요청 DTO

        Returns:
            MultiAgentResponse: Multi-Agent 응답 DTO
        """
        strategy = request.strategy
        if strategy == "sequential":
            return await self.execute_sequential(request)
        elif strategy == "parallel":
            return await self.execute_parallel(request)
        elif strategy == "hierarchical":
            return await self.execute_hierarchical(request)
        elif strategy == "debate":
            return await self.execute_debate(request)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
