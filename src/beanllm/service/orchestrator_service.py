"""
IOrchestratorService - Multi-Agent 오케스트레이터 서비스 인터페이스
SOLID 원칙:
- ISP: 오케스트레이션 관련 메서드만 포함
- DIP: 인터페이스에 의존
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from beanllm.dto.request.advanced.orchestrator_request import (
    CreateWorkflowRequest,
    ExecuteWorkflowRequest,
    MonitorWorkflowRequest,
)
from beanllm.dto.response.advanced.orchestrator_response import (
    AnalyticsResponse,
    CreateWorkflowResponse,
    ExecuteWorkflowResponse,
    MonitorWorkflowResponse,
)


class IOrchestratorService(ABC):
    """
    Multi-Agent 오케스트레이터 서비스 인터페이스

    책임:
    - 워크플로우 생성, 실행, 모니터링 비즈니스 로직 정의
    - 시각적 워크플로우 빌더, 실시간 모니터링, 분석

    SOLID:
    - ISP: 오케스트레이션 관련 메서드만
    - DIP: 구현체가 아닌 인터페이스에 의존
    """

    @abstractmethod
    async def create_workflow(self, request: CreateWorkflowRequest) -> CreateWorkflowResponse:
        """
        워크플로우 생성

        Args:
            request: 워크플로우 생성 요청 DTO

        Returns:
            CreateWorkflowResponse: 생성된 워크플로우 정보
        """
        pass

    @abstractmethod
    async def execute_workflow(self, request: ExecuteWorkflowRequest) -> ExecuteWorkflowResponse:
        """
        워크플로우 실행

        Args:
            request: 워크플로우 실행 요청 DTO

        Returns:
            ExecuteWorkflowResponse: 실행 결과
        """
        pass

    @abstractmethod
    async def monitor_workflow(self, request: MonitorWorkflowRequest) -> MonitorWorkflowResponse:
        """
        워크플로우 실시간 모니터링

        Args:
            request: 모니터링 요청 DTO

        Returns:
            MonitorWorkflowResponse: 실시간 모니터링 데이터
        """
        pass

    @abstractmethod
    async def get_analytics(self, workflow_id: str) -> AnalyticsResponse:
        """
        워크플로우 분석 (utilization, bottleneck, cost)

        Args:
            workflow_id: 워크플로우 ID

        Returns:
            AnalyticsResponse: 분석 결과
        """
        pass

    @abstractmethod
    async def visualize_workflow(self, workflow_id: str) -> str:
        """
        워크플로우 시각화 (ASCII diagram)

        Args:
            workflow_id: 워크플로우 ID

        Returns:
            str: ASCII 다이어그램
        """
        pass

    @abstractmethod
    async def get_templates(self) -> Dict[str, Any]:
        """
        사전 정의된 워크플로우 템플릿 목록

        Returns:
            Dict: 템플릿 목록
        """
        pass
