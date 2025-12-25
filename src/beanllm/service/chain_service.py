"""
IChainService - Chain 서비스 인터페이스
SOLID 원칙:
- ISP: Chain 관련 메서드만 포함
- DIP: 인터페이스에 의존
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..dto.request.chain_request import ChainRequest
from ..dto.response.chain_response import ChainResponse


class IChainService(ABC):
    """
    Chain 서비스 인터페이스

    책임:
    - Chain 비즈니스 로직 정의만
    - 검증, 에러 처리 없음 (Handler에서 처리)

    SOLID:
    - ISP: Chain 관련 메서드만 (작은 인터페이스)
    - DIP: 구현체가 아닌 인터페이스에 의존
    """

    @abstractmethod
    async def run_chain(self, request: ChainRequest) -> ChainResponse:
        """
        기본 Chain 실행

        Args:
            request: Chain 요청 DTO

        Returns:
            ChainResponse: Chain 응답 DTO

        책임:
            - Chain 비즈니스 로직만
            - 검증 없음 (Handler에서 처리)
            - 에러 처리 없음 (Handler에서 처리)
        """
        pass

    @abstractmethod
    async def run_prompt_chain(self, request: ChainRequest) -> ChainResponse:
        """
        Prompt Chain 실행

        Args:
            request: Chain 요청 DTO

        Returns:
            ChainResponse: Chain 응답 DTO
        """
        pass

    @abstractmethod
    async def run_sequential_chain(self, request: ChainRequest) -> ChainResponse:
        """
        Sequential Chain 실행

        Args:
            request: Chain 요청 DTO

        Returns:
            ChainResponse: Chain 응답 DTO
        """
        pass

    @abstractmethod
    async def run_parallel_chain(self, request: ChainRequest) -> ChainResponse:
        """
        Parallel Chain 실행

        Args:
            request: Chain 요청 DTO

        Returns:
            ChainResponse: Chain 응답 DTO
        """
        pass

    async def execute(self, request: ChainRequest) -> ChainResponse:
        """
        통합 실행 메서드 (Strategy 패턴)

        Args:
            request: Chain 요청 DTO

        Returns:
            ChainResponse: Chain 응답 DTO
        """
        chain_type = request.chain_type
        if chain_type == "basic":
            return await self.run_chain(request)
        elif chain_type == "prompt":
            return await self.run_prompt_chain(request)
        elif chain_type == "sequential":
            return await self.run_sequential_chain(request)
        elif chain_type == "parallel":
            return await self.run_parallel_chain(request)
        else:
            raise ValueError(f"Unknown chain type: {chain_type}")
