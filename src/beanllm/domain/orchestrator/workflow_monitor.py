"""
WorkflowMonitor - 실시간 워크플로우 모니터링
SOLID 원칙:
- SRP: 모니터링 및 상태 추적만 담당
- OCP: 새로운 이벤트 타입 추가 가능
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


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


class WorkflowMonitor:
    """
    워크플로우 실시간 모니터링

    책임:
    - 워크플로우 실행 상태 추적
    - 이벤트 발생 및 리스너 관리
    - 실시간 진행 상황 제공
    - 성능 메트릭 수집

    Example:
        ```python
        monitor = WorkflowMonitor(workflow_id="wf123")

        # 이벤트 리스너 등록
        monitor.add_listener(EventType.NODE_START, on_node_start)
        monitor.add_listener(EventType.NODE_END, on_node_end)

        # 워크플로우 시작
        await monitor.start()

        # 노드 시작
        await monitor.node_started(node_id="node1")

        # 노드 완료
        await monitor.node_completed(node_id="node1", output="result")

        # 현재 상태 조회
        status = monitor.get_status()
        print(f"Progress: {status['progress_percent']}%")
        ```
    """

    def __init__(
        self,
        workflow_id: str,
        total_nodes: int = 0,
    ) -> None:
        """
        Args:
            workflow_id: 워크플로우 ID
            total_nodes: 총 노드 수
        """
        self.workflow_id = workflow_id
        self.total_nodes = total_nodes

        # State tracking
        self.node_states: Dict[str, NodeExecutionState] = {}
        self.event_history: List[MonitorEvent] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # Event listeners
        self.listeners: Dict[EventType, List[Callable]] = {
            event_type: [] for event_type in EventType
        }

        # Real-time stats
        self.stats = {
            "nodes_completed": 0,
            "nodes_failed": 0,
            "nodes_running": 0,
            "nodes_pending": 0,
        }

    def add_listener(
        self,
        event_type: EventType,
        callback: Callable[[MonitorEvent], None],
    ) -> None:
        """
        이벤트 리스너 추가

        Args:
            event_type: 이벤트 타입
            callback: 콜백 함수
        """
        self.listeners[event_type].append(callback)
        logger.debug(f"Added listener for {event_type.value}")

    def remove_listener(
        self,
        event_type: EventType,
        callback: Callable[[MonitorEvent], None],
    ) -> None:
        """이벤트 리스너 제거"""
        if callback in self.listeners[event_type]:
            self.listeners[event_type].remove(callback)

    async def _emit_event(self, event: MonitorEvent) -> None:
        """
        이벤트 발생

        Args:
            event: 발생할 이벤트
        """
        # Store in history
        self.event_history.append(event)

        # Call listeners
        for callback in self.listeners[event.event_type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in event listener: {e}")

    async def start(self) -> None:
        """워크플로우 시작"""
        self.start_time = datetime.now()

        event = MonitorEvent(
            event_type=EventType.WORKFLOW_START,
            timestamp=self.start_time,
            workflow_id=self.workflow_id,
            data={"total_nodes": self.total_nodes},
        )

        await self._emit_event(event)
        logger.info(f"Workflow {self.workflow_id} monitoring started")

    async def end(self, success: bool = True) -> None:
        """워크플로우 종료"""
        self.end_time = datetime.now()

        duration_ms = 0.0
        if self.start_time:
            duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

        event = MonitorEvent(
            event_type=EventType.WORKFLOW_END,
            timestamp=self.end_time,
            workflow_id=self.workflow_id,
            data={
                "success": success,
                "duration_ms": duration_ms,
                "stats": self.stats.copy(),
            },
        )

        await self._emit_event(event)
        logger.info(f"Workflow {self.workflow_id} monitoring ended")

    async def node_started(
        self,
        node_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        노드 시작

        Args:
            node_id: 노드 ID
            metadata: 메타데이터
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

        # Emit event
        event = MonitorEvent(
            event_type=EventType.NODE_START,
            timestamp=start_time,
            workflow_id=self.workflow_id,
            node_id=node_id,
            data={"attempt": state.attempts},
            metadata=metadata or {},
        )

        await self._emit_event(event)
        logger.debug(f"Node {node_id} started (attempt {state.attempts})")

    async def node_completed(
        self,
        node_id: str,
        output: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        노드 완료

        Args:
            node_id: 노드 ID
            output: 출력 결과
            metadata: 메타데이터
        """
        end_time = datetime.now()

        if node_id not in self.node_states:
            logger.warning(f"Node {node_id} completed but not started")
            return

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

        # Emit event
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

        await self._emit_event(event)
        logger.debug(f"Node {node_id} completed in {state.duration_ms:.2f}ms")

    async def node_failed(
        self,
        node_id: str,
        error: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        노드 실패

        Args:
            node_id: 노드 ID
            error: 에러 메시지
            metadata: 메타데이터
        """
        end_time = datetime.now()

        if node_id not in self.node_states:
            logger.warning(f"Node {node_id} failed but not started")
            return

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

        # Emit event
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

        await self._emit_event(event)
        logger.error(f"Node {node_id} failed: {error}")

    async def node_skipped(
        self,
        node_id: str,
        reason: str = "Condition not met",
    ) -> None:
        """
        노드 건너뜀

        Args:
            node_id: 노드 ID
            reason: 이유
        """
        if node_id not in self.node_states:
            self.node_states[node_id] = NodeExecutionState(
                node_id=node_id,
                status=NodeStatus.SKIPPED,
            )
        else:
            self.node_states[node_id].status = NodeStatus.SKIPPED

        # Emit event
        event = MonitorEvent(
            event_type=EventType.STATE_CHANGED,
            timestamp=datetime.now(),
            workflow_id=self.workflow_id,
            node_id=node_id,
            data={"status": "skipped", "reason": reason},
        )

        await self._emit_event(event)
        logger.debug(f"Node {node_id} skipped: {reason}")

    async def edge_traversed(
        self,
        source_id: str,
        target_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        엣지 이동

        Args:
            source_id: 소스 노드 ID
            target_id: 타겟 노드 ID
            metadata: 메타데이터
        """
        event = MonitorEvent(
            event_type=EventType.EDGE_TRAVERSED,
            timestamp=datetime.now(),
            workflow_id=self.workflow_id,
            data={
                "source": source_id,
                "target": target_id,
            },
            metadata=metadata or {},
        )

        await self._emit_event(event)

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
            "total_events": len(self.event_history),
        }

    def get_node_state(self, node_id: str) -> Optional[NodeExecutionState]:
        """특정 노드 상태 조회"""
        return self.node_states.get(node_id)

    def get_all_node_states(self) -> Dict[str, NodeExecutionState]:
        """모든 노드 상태 조회"""
        return self.node_states.copy()

    def get_recent_events(self, limit: int = 10) -> List[MonitorEvent]:
        """최근 이벤트 조회"""
        return self.event_history[-limit:]

    def get_timeline(self) -> List[Dict[str, Any]]:
        """
        실행 타임라인 생성

        Returns:
            List[Dict]: 타임라인 항목 리스트
        """
        timeline = []

        for event in self.event_history:
            timeline.append(
                {
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type.value,
                    "node_id": event.node_id,
                    "data": event.data,
                }
            )

        return timeline

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        성능 요약

        Returns:
            Dict: 성능 메트릭
        """
        if not self.node_states:
            return {}

        completed_states = [
            s for s in self.node_states.values() if s.status == NodeStatus.COMPLETED
        ]

        if not completed_states:
            return {"completed_nodes": 0}

        durations = [s.duration_ms for s in completed_states if s.duration_ms > 0]

        avg_duration = sum(durations) / len(durations) if durations else 0.0
        min_duration = min(durations) if durations else 0.0
        max_duration = max(durations) if durations else 0.0

        # Slowest nodes
        slowest = sorted(completed_states, key=lambda s: s.duration_ms, reverse=True)[:5]

        return {
            "completed_nodes": len(completed_states),
            "avg_duration_ms": avg_duration,
            "min_duration_ms": min_duration,
            "max_duration_ms": max_duration,
            "slowest_nodes": [
                {"node_id": s.node_id, "duration_ms": s.duration_ms} for s in slowest
            ],
        }

    def export_report(self) -> Dict[str, Any]:
        """
        완전한 리포트 내보내기

        Returns:
            Dict: 전체 리포트
        """
        return {
            "workflow_id": self.workflow_id,
            "status": self.get_status(),
            "node_states": {nid: state.to_dict() for nid, state in self.node_states.items()},
            "timeline": self.get_timeline(),
            "performance": self.get_performance_summary(),
            "event_count": len(self.event_history),
        }

    def reset(self) -> None:
        """모니터 리셋"""
        self.node_states.clear()
        self.event_history.clear()
        self.start_time = None
        self.end_time = None
        self.stats = {
            "nodes_completed": 0,
            "nodes_failed": 0,
            "nodes_running": 0,
            "nodes_pending": 0,
        }
        logger.info(f"Monitor {self.workflow_id} reset")
