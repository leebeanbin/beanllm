"""
WorkflowMonitor 타입 정의

순환 import 방지를 위해 순수 데이터 타입을 별도 모듈로 분리합니다.
EventType, MonitorEvent, NodeStatus, NodeExecutionState 등을 정의합니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class NodeStatus(Enum):
    """노드 실행 상태"""

    PENDING = "pending"  # 대기 중
    RUNNING = "running"  # 실행 중
    COMPLETED = "completed"  # 완료
    FAILED = "failed"  # 실패
    SKIPPED = "skipped"  # 건너뜀


class EventType(Enum):
    """모니터링 이벤트 타입"""

    WORKFLOW_START = "workflow_start"
    WORKFLOW_END = "workflow_end"
    NODE_START = "node_start"
    NODE_END = "node_end"
    NODE_ERROR = "node_error"
    EDGE_TRAVERSED = "edge_traversed"
    STATE_CHANGED = "state_changed"


@dataclass
class MonitorEvent:
    """모니터링 이벤트"""

    event_type: EventType
    timestamp: datetime
    workflow_id: str
    node_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "workflow_id": self.workflow_id,
            "node_id": self.node_id,
            "data": self.data,
            "metadata": self.metadata,
        }


@dataclass
class NodeExecutionState:
    """노드 실행 상태"""

    node_id: str
    status: NodeStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    output: Any = None
    error: Optional[str] = None
    attempts: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "node_id": self.node_id,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "output": str(self.output) if self.output else None,
            "error": self.error,
            "attempts": self.attempts,
            "metadata": self.metadata,
        }
