"""
Orchestrator Response DTOs - 오케스트레이터 응답 데이터 전송 객체
책임: 응답 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class CreateWorkflowResponse:
    """
    워크플로우 생성 응답 DTO

    책임:
    - 응답 데이터 구조 정의만
    - 변환 로직 없음 (Service에서 처리)
    """

    workflow_id: str
    workflow_name: str
    num_nodes: int
    num_edges: int
    strategy: str
    visualization: str  # ASCII diagram
    created_at: str
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ExecuteWorkflowResponse:
    """
    워크플로우 실행 응답 DTO
    """

    execution_id: str
    workflow_id: str
    status: str  # "running", "completed", "failed"
    result: Optional[Any] = None
    node_results: Optional[List[Dict[str, Any]]] = None
    execution_time: Optional[float] = None  # seconds
    checkpoint_id: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MonitorWorkflowResponse:
    """
    워크플로우 모니터링 응답 DTO
    """

    execution_id: str
    workflow_id: str
    current_node: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0
    nodes_completed: Optional[List[str]] = None
    nodes_pending: Optional[List[str]] = None
    messages: Optional[List[Dict[str, Any]]] = None  # Agent messages
    elapsed_time: Optional[float] = None
    estimated_remaining: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.nodes_completed is None:
            self.nodes_completed = []
        if self.nodes_pending is None:
            self.nodes_pending = []
        if self.messages is None:
            self.messages = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AnalyticsResponse:
    """
    워크플로우 분석 응답 DTO
    """

    workflow_id: str
    total_executions: int
    avg_execution_time: float
    success_rate: float
    bottlenecks: List[Dict[str, Any]]  # [{"node": "name", "avg_time": ...}]
    agent_utilization: Dict[str, float]  # {"agent_name": utilization_ratio}
    cost_breakdown: Dict[str, float]  # {"llm": cost, "embedding": cost, ...}
    recommendations: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []
        if self.metadata is None:
            self.metadata = {}
