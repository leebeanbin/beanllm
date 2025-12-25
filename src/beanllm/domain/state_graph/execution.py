"""
Graph Execution - 실행 기록
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TypeVar

StateType = TypeVar("StateType", bound=Dict[str, Any])


@dataclass
class NodeExecution:
    """노드 실행 기록"""

    node_name: str
    input_state: Dict[str, Any]
    output_state: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[Exception] = None


@dataclass
class GraphExecution:
    """그래프 실행 기록"""

    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    nodes_executed: List[NodeExecution] = field(default_factory=list)
    final_state: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None


class END:
    """종료 노드 마커"""

    pass
