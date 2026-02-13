"""
Status Tracker - 상태 추적 모듈
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from beanllm.domain.orchestrator.monitor_types import (
    EventType,
    MonitorEvent,
    NodeExecutionState,
    NodeStatus,
)
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class StatusTracker:
    """상태 추적기"""

    def __init__(self, workflow_id: str, total_nodes: int = 0) -> None:
        """
        Args:
            workflow_id: 워크플로우 ID
            total_nodes: 총 노드 수
        """
        self.workflow_id = workflow_id
        self.total_nodes = total_nodes

        # State tracking
        self.node_states: Dict[str, NodeExecutionState] = {}
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # Real-time stats
        self.stats = {
            "nodes_completed": 0,
            "nodes_failed": 0,
            "nodes_running": 0,
            "nodes_pending": 0,
        }

    def get_status(self) -> Dict[str, Any]:
        """
        현재 상태 조회

        Returns:
            Dict: 상태 정보
        """
        total_finished = self.stats["nodes_completed"] + self.stats["nodes_failed"]
        progress_percent = (
            (total_finished / self.total_nodes * 100) if self.total_nodes > 0 else 0.0
        )

        current_time = datetime.now()
        elapsed_ms = 0.0
        if self.start_time:
            elapsed_ms = (current_time - self.start_time).total_seconds() * 1000

        return {
            "workflow_id": self.workflow_id,
            "is_running": self.start_time is not None and self.end_time is None,
            "progress_percent": progress_percent,
            "elapsed_ms": elapsed_ms,
            "stats": self.stats.copy(),
            "total_nodes": self.total_nodes,
        }

    def get_node_state(self, node_id: str) -> Optional[NodeExecutionState]:
        """특정 노드 상태 조회"""
        return self.node_states.get(node_id)

    def get_all_node_states(self) -> Dict[str, NodeExecutionState]:
        """모든 노드 상태 조회"""
        return self.node_states.copy()

    def node_started(
        self,
        node_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MonitorEvent:
        """
        노드 시작

        Args:
            node_id: 노드 ID
            metadata: 메타데이터

        Returns:
            MonitorEvent: 생성된 이벤트
        """
        start_time = datetime.now()

        # Update state
        if node_id not in self.node_states:
            self.node_states[node_id] = NodeExecutionState(
                node_id=node_id,
                status=NodeStatus.PENDING,
            )

        state = self.node_states[node_id]
        state.status = NodeStatus.RUNNING
        state.start_time = start_time
        state.attempts += 1

        if metadata:
            state.metadata.update(metadata)

        # Update stats
        self.stats["nodes_running"] += 1
        if state.attempts == 1:
            self.stats["nodes_pending"] -= 1

        # Create event
        event = MonitorEvent(
            event_type=EventType.NODE_START,
            timestamp=start_time,
            workflow_id=self.workflow_id,
            node_id=node_id,
            data={"attempt": state.attempts},
            metadata=metadata or {},
        )

        logger.debug(f"Node {node_id} started (attempt {state.attempts})")
        return event

    def node_completed(
        self,
        node_id: str,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MonitorEvent:
        """
        노드 완료

        Args:
            node_id: 노드 ID
            output: 출력 결과
            metadata: 메타데이터

        Returns:
            MonitorEvent: 생성된 이벤트
        """
        end_time = datetime.now()

        if node_id not in self.node_states:
            logger.warning(f"Node {node_id} completed but not started")
            # Create a minimal state
            self.node_states[node_id] = NodeExecutionState(
                node_id=node_id,
                status=NodeStatus.COMPLETED,
            )

        state = self.node_states[node_id]
        state.status = NodeStatus.COMPLETED
        state.end_time = end_time
        state.output = output

        if state.start_time:
            state.duration_ms = (end_time - state.start_time).total_seconds() * 1000

        if metadata:
            state.metadata.update(metadata)

        # Update stats
        self.stats["nodes_running"] -= 1
        self.stats["nodes_completed"] += 1

        # Create event
        event = MonitorEvent(
            event_type=EventType.NODE_END,
            timestamp=end_time,
            workflow_id=self.workflow_id,
            node_id=node_id,
            data={
                "success": True,
                "duration_ms": state.duration_ms,
                "output": str(output) if output else None,
            },
            metadata=metadata or {},
        )

        logger.debug(f"Node {node_id} completed in {state.duration_ms:.2f}ms")
        return event

    def node_failed(
        self,
        node_id: str,
        error: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MonitorEvent:
        """
        노드 실패

        Args:
            node_id: 노드 ID
            error: 에러 메시지
            metadata: 메타데이터

        Returns:
            MonitorEvent: 생성된 이벤트
        """
        end_time = datetime.now()

        if node_id not in self.node_states:
            logger.warning(f"Node {node_id} failed but not started")
            # Create a minimal state
            self.node_states[node_id] = NodeExecutionState(
                node_id=node_id,
                status=NodeStatus.FAILED,
            )

        state = self.node_states[node_id]
        state.status = NodeStatus.FAILED
        state.end_time = end_time
        state.error = error

        if state.start_time:
            state.duration_ms = (end_time - state.start_time).total_seconds() * 1000

        if metadata:
            state.metadata.update(metadata)

        # Update stats
        self.stats["nodes_running"] -= 1
        self.stats["nodes_failed"] += 1

        # Create event
        event = MonitorEvent(
            event_type=EventType.NODE_ERROR,
            timestamp=end_time,
            workflow_id=self.workflow_id,
            node_id=node_id,
            data={
                "error": error,
                "duration_ms": state.duration_ms,
            },
            metadata=metadata or {},
        )

        logger.error(f"Node {node_id} failed: {error}")
        return event

    def node_skipped(
        self,
        node_id: str,
        reason: str = "Condition not met",
    ) -> MonitorEvent:
        """
        노드 건너뜀

        Args:
            node_id: 노드 ID
            reason: 이유

        Returns:
            MonitorEvent: 생성된 이벤트
        """
        if node_id not in self.node_states:
            self.node_states[node_id] = NodeExecutionState(
                node_id=node_id,
                status=NodeStatus.SKIPPED,
            )
        else:
            self.node_states[node_id].status = NodeStatus.SKIPPED

        # Create event
        event = MonitorEvent(
            event_type=EventType.STATE_CHANGED,
            timestamp=datetime.now(),
            workflow_id=self.workflow_id,
            node_id=node_id,
            data={"status": "skipped", "reason": reason},
        )

        logger.debug(f"Node {node_id} skipped: {reason}")
        return event

    def reset(self) -> None:
        """상태 추적기 리셋"""
        self.node_states.clear()
        self.start_time = None
        self.end_time = None
        self.stats = {
            "nodes_completed": 0,
            "nodes_failed": 0,
            "nodes_running": 0,
            "nodes_pending": 0,
        }
