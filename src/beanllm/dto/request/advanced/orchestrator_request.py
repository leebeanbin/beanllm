"""
Orchestrator Request DTOs - 오케스트레이터 요청 데이터 전송 객체
책임: 요청 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(slots=True, kw_only=True)
class CreateWorkflowRequest:
    """
    워크플로우 생성 요청 DTO

    책임:
    - 데이터 구조 정의만
    - 검증 없음 (Handler에서 처리)
    """

    workflow_name: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    strategy: str = "sequential"
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, kw_only=True)
class ExecuteWorkflowRequest:
    """
    워크플로우 실행 요청 DTO
    """

    workflow_id: str
    input_data: Dict[str, Any]
    stream: bool = False
    checkpoint: bool = True


@dataclass(slots=True, kw_only=True)
class MonitorWorkflowRequest:
    """
    워크플로우 모니터링 요청 DTO
    """

    workflow_id: str
    execution_id: str
    real_time: bool = True
