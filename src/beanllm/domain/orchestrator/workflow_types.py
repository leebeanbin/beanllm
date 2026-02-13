"""
Workflow Types - 워크플로우 타입 정의

NodeType, EdgeCondition Enum과
WorkflowNode, WorkflowEdge, ExecutionResult 데이터 클래스를 정의.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional


class NodeType(Enum):
    """워크플로우 노드 타입"""

    AGENT = "agent"  # 단일 Agent 실행
    TOOL = "tool"  # Tool 실행
    DECISION = "decision"  # 조건부 분기
    PARALLEL = "parallel"  # 병렬 실행
    SEQUENTIAL = "sequential"  # 순차 실행 그룹
    HIERARCHICAL = "hierarchical"  # 계층적 실행
    DEBATE = "debate"  # 토론
    MERGE = "merge"  # 결과 병합
    START = "start"  # 시작 노드
    END = "end"  # 종료 노드


class EdgeCondition(Enum):
    """엣지 조건"""

    ALWAYS = "always"  # 항상 실행
    ON_SUCCESS = "on_success"  # 성공 시
    ON_FAILURE = "on_failure"  # 실패 시
    CONDITIONAL = "conditional"  # 조건부 (함수 평가)


@dataclass
class WorkflowNode:
    """
    워크플로우 노드

    각 노드는 워크플로우의 한 단계를 나타냅니다.
    """

    node_id: str
    node_type: NodeType
    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    position: tuple[int, int] = (0, 0)  # (x, y) for visualization
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "name": self.name,
            "config": self.config,
            "position": self.position,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowNode":
        """딕셔너리에서 생성"""
        return cls(
            node_id=data["node_id"],
            node_type=NodeType(data["node_type"]),
            name=data["name"],
            config=data.get("config", {}),
            position=tuple(data.get("position", (0, 0))),
            metadata=data.get("metadata", {}),
        )


@dataclass
class WorkflowEdge:
    """
    워크플로우 엣지 (노드 간 연결)
    """

    edge_id: str
    source: str  # source node_id
    target: str  # target node_id
    condition: EdgeCondition = EdgeCondition.ALWAYS
    condition_func: Optional[Callable[[Any], bool]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def should_execute(self, context: Dict[str, Any]) -> bool:
        """
        엣지를 따라 실행할지 결정

        Args:
            context: 실행 컨텍스트 (이전 노드의 결과 등)

        Returns:
            bool: 실행 여부
        """
        if self.condition == EdgeCondition.ALWAYS:
            return True

        elif self.condition == EdgeCondition.ON_SUCCESS:
            return bool(context.get("success", True))

        elif self.condition == EdgeCondition.ON_FAILURE:
            return not bool(context.get("success", True))

        elif self.condition == EdgeCondition.CONDITIONAL:
            if self.condition_func:
                return self.condition_func(context)
            return True

        return True

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "edge_id": self.edge_id,
            "source": self.source,
            "target": self.target,
            "condition": self.condition.value,
            "metadata": self.metadata,
        }


@dataclass
class ExecutionResult:
    """노드 실행 결과"""

    node_id: str
    success: bool
    output: Any
    error: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "node_id": self.node_id,
            "success": self.success,
            "output": str(self.output),
            "error": self.error,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }
