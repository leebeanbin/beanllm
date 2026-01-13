"""
OrchestratorHandler - Multi-Agent 오케스트레이터 Handler
SOLID 원칙:
- SRP: 검증 및 에러 처리만 담당
- DIP: 인터페이스에 의존
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

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
from beanllm.utils.logging import get_logger

if TYPE_CHECKING:
    from beanllm.service.orchestrator_service import IOrchestratorService

logger = get_logger(__name__)


class OrchestratorHandler:
    """
    Multi-Agent 오케스트레이터 Handler

    책임:
    - 요청 검증
    - 에러 처리
    - 응답 포매팅
    """

    def __init__(self, service: "IOrchestratorService") -> None:
        """
        Args:
            service: Orchestrator 서비스
        """
        self._service = service

    async def handle_create_workflow(
        self, request: CreateWorkflowRequest
    ) -> CreateWorkflowResponse:
        """
        워크플로우 생성 처리

        Args:
            request: 워크플로우 생성 요청

        Returns:
            CreateWorkflowResponse: 생성 결과

        Raises:
            ValueError: Validation 실패 시
            RuntimeError: Service 에러 시
        """
        # Validation
        if not request.workflow_name:
            raise ValueError("workflow_name is required")

        if request.strategy == "custom":
            if not request.nodes:
                raise ValueError("nodes are required for custom workflow")
            if not request.edges:
                raise ValueError("edges are required for custom workflow")

        # Service 호출
        try:
            response = await self._service.create_workflow(request)
            return response
        except ValueError as e:
            logger.error(f"Validation error in create_workflow: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in create_workflow: {e}")
            raise RuntimeError(f"Failed to create workflow: {e}") from e

    async def handle_execute_workflow(
        self, request: ExecuteWorkflowRequest
    ) -> ExecuteWorkflowResponse:
        """
        워크플로우 실행 처리

        Args:
            request: 실행 요청

        Returns:
            ExecuteWorkflowResponse: 실행 결과

        Raises:
            ValueError: Validation 실패 시
            RuntimeError: Service 에러 시
        """
        # Validation
        if not request.workflow_id:
            raise ValueError("workflow_id is required")

        if not request.input_data:
            raise ValueError("input_data is required")

        # Service 호출
        try:
            response = await self._service.execute_workflow(request)
            return response
        except ValueError as e:
            logger.error(f"Validation error in execute_workflow: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in execute_workflow: {e}")
            raise RuntimeError(f"Failed to execute workflow: {e}") from e

    async def handle_monitor_workflow(
        self, request: MonitorWorkflowRequest
    ) -> MonitorWorkflowResponse:
        """
        워크플로우 모니터링 처리

        Args:
            request: 모니터링 요청

        Returns:
            MonitorWorkflowResponse: 모니터링 데이터

        Raises:
            ValueError: Validation 실패 시
            RuntimeError: Service 에러 시
        """
        # Validation
        if not request.workflow_id:
            raise ValueError("workflow_id is required")

        if not request.execution_id:
            raise ValueError("execution_id is required")

        # Service 호출
        try:
            response = await self._service.monitor_workflow(request)
            return response
        except ValueError as e:
            logger.error(f"Validation error in monitor_workflow: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in monitor_workflow: {e}")
            raise RuntimeError(f"Failed to monitor workflow: {e}") from e

    async def handle_get_analytics(self, workflow_id: str) -> AnalyticsResponse:
        """
        분석 결과 조회 처리

        Args:
            workflow_id: 워크플로우 ID

        Returns:
            AnalyticsResponse: 분석 결과

        Raises:
            ValueError: Validation 실패 시
            RuntimeError: Service 에러 시
        """
        # Validation
        if not workflow_id:
            raise ValueError("workflow_id is required")

        # Service 호출
        try:
            response = await self._service.get_analytics(workflow_id)
            return response
        except ValueError as e:
            logger.error(f"Validation error in get_analytics: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in get_analytics: {e}")
            raise RuntimeError(f"Failed to get analytics: {e}") from e

    async def handle_visualize_workflow(self, workflow_id: str) -> str:
        """
        워크플로우 시각화 처리

        Args:
            workflow_id: 워크플로우 ID

        Returns:
            str: ASCII 다이어그램

        Raises:
            ValueError: Validation 실패 시
            RuntimeError: Service 에러 시
        """
        # Validation
        if not workflow_id:
            raise ValueError("workflow_id is required")

        # Service 호출
        try:
            diagram = await self._service.visualize_workflow(workflow_id)
            return diagram
        except ValueError as e:
            logger.error(f"Validation error in visualize_workflow: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in visualize_workflow: {e}")
            raise RuntimeError(f"Failed to visualize workflow: {e}") from e

    async def handle_get_templates(self) -> Dict[str, Any]:
        """
        템플릿 목록 조회 처리

        Returns:
            Dict: 템플릿 목록

        Raises:
            RuntimeError: Service 에러 시
        """
        # Service 호출
        try:
            templates = await self._service.get_templates()
            return templates
        except Exception as e:
            logger.error(f"Error in get_templates: {e}")
            raise RuntimeError(f"Failed to get templates: {e}") from e
