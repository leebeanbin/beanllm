"""
Orchestrator Response DTOs - 오케스트레이터 응답 데이터 전송 객체
책임: 응답 데이터만 전달
"""

from __future__ import annotations

from typing import Optional

from pydantic import ConfigDict

from beanllm.dto.response.base_response import BaseResponse


class CreateWorkflowResponse(BaseResponse):
    """
    워크플로우 생성 응답 DTO

    책임:
    - 응답 데이터 구조 정의만
    - 변환 로직 없음 (Service에서 처리)
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    workflow_id: str
    workflow_name: str
    num_nodes: int
    num_edges: int
    strategy: str
    visualization: str
    created_at: str
    metadata: dict[str, object] = {}


class ExecuteWorkflowResponse(BaseResponse):
    """
    워크플로우 실행 응답 DTO
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    execution_id: str
    workflow_id: str
    status: str
    result: Optional[object] = None
    node_results: Optional[list[dict[str, object]]] = None
    execution_time: Optional[float] = None
    checkpoint_id: Optional[str] = None
    error: Optional[str] = None
    metadata: dict[str, object] = {}


class MonitorWorkflowResponse(BaseResponse):
    """
    워크플로우 모니터링 응답 DTO
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    execution_id: str
    workflow_id: str
    current_node: Optional[str] = None
    progress: float = 0.0
    nodes_completed: list[str] = []
    nodes_pending: list[str] = []
    messages: list[dict[str, object]] = []
    elapsed_time: Optional[float] = None
    estimated_remaining: Optional[float] = None
    metadata: dict[str, object] = {}


class AnalyticsResponse(BaseResponse):
    """
    워크플로우 분석 응답 DTO
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    workflow_id: str
    total_executions: int
    avg_execution_time: float
    success_rate: float
    bottlenecks: list[dict[str, object]]
    agent_utilization: dict[str, float]
    cost_breakdown: dict[str, float]
    recommendations: list[str] = []
    metadata: dict[str, object] = {}
