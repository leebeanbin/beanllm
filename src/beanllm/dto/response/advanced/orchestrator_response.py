"""
Orchestrator Response DTOs - 오케스트레이터 응답 데이터 전송 객체
책임: 응답 데이터만 전달
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import ConfigDict

from beanllm.dto.response.base_response import BaseResponse


class CreateWorkflowResponse(BaseResponse):
    """
    워크플로우 생성 응답 DTO

    책임:
    - 응답 데이터 구조 정의만
    - 변환 로직 없음 (Service에서 처리)
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    workflow_id: str
    workflow_name: str
    num_nodes: int
    num_edges: int
    strategy: str
    visualization: str  # ASCII diagram
    created_at: str
    metadata: Dict[str, Any] = {}


class ExecuteWorkflowResponse(BaseResponse):
    """
    워크플로우 실행 응답 DTO
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    execution_id: str
    workflow_id: str
    status: str  # "running", "completed", "failed"
    result: Optional[Any] = None
    node_results: Optional[List[Dict[str, Any]]] = None
    execution_time: Optional[float] = None
    checkpoint_id: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


class MonitorWorkflowResponse(BaseResponse):
    """
    워크플로우 모니터링 응답 DTO
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    execution_id: str
    workflow_id: str
    current_node: Optional[str] = None
    progress: float = 0.0
    nodes_completed: List[str] = []
    nodes_pending: List[str] = []
    messages: List[Dict[str, Any]] = []
    elapsed_time: Optional[float] = None
    estimated_remaining: Optional[float] = None
    metadata: Dict[str, Any] = {}


class AnalyticsResponse(BaseResponse):
    """
    워크플로우 분석 응답 DTO
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    workflow_id: str
    total_executions: int
    avg_execution_time: float
    success_rate: float
    bottlenecks: List[Dict[str, Any]]
    agent_utilization: Dict[str, float]
    cost_breakdown: Dict[str, float]
    recommendations: List[str] = []
    metadata: Dict[str, Any] = {}
