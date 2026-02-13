"""
WorkflowMonitor - 실시간 워크플로우 모니터링
SOLID 원칙:
- SRP: 모니터링 및 상태 추적만 담당
- OCP: 새로운 이벤트 타입 추가 가능
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from beanllm.domain.orchestrator.monitor_types import (
    EventType,
    MonitorEvent,
    NodeExecutionState,
    NodeStatus,
)
from beanllm.domain.orchestrator.monitoring import EventHandler, PerformanceMetrics, StatusTracker
from beanllm.utils.logging import get_logger

# Re-export for backward compatibility
__all__ = [
    "EventType",
    "MonitorEvent",
    "NodeExecutionState",
    "NodeStatus",
    "WorkflowMonitor",
]

logger = get_logger(__name__)


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

        # Compose modules
        self._event_handler = EventHandler()
        self._status_tracker = StatusTracker(workflow_id, total_nodes)
        self._performance_metrics = PerformanceMetrics()

        # Expose for backward compatibility (direct references)
        self.node_states = self._status_tracker.node_states
        self.event_history = self._event_handler.event_history
        self.listeners = self._event_handler.listeners
        self.stats = self._status_tracker.stats

    @property
    def start_time(self) -> Optional[datetime]:
        """워크플로우 시작 시간"""
        return self._status_tracker.start_time

    @property
    def end_time(self) -> Optional[datetime]:
        """워크플로우 종료 시간"""
        return self._status_tracker.end_time

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
        self._event_handler.add_listener(event_type, callback)

    def remove_listener(
        self,
        event_type: EventType,
        callback: Callable[[MonitorEvent], None],
    ) -> None:
        """이벤트 리스너 제거"""
        self._event_handler.remove_listener(event_type, callback)

    async def _emit_event(self, event: MonitorEvent) -> None:
        """
        이벤트 발생

        Args:
            event: 발생할 이벤트
        """
        await self._event_handler.emit_event(event)

    async def start(self) -> None:
        """워크플로우 시작"""
        now = datetime.now()
        self._status_tracker.start_time = now

        event = MonitorEvent(
            event_type=EventType.WORKFLOW_START,
            timestamp=now,
            workflow_id=self.workflow_id,
            data={"total_nodes": self.total_nodes},
        )

        await self._emit_event(event)
        logger.info(f"Workflow {self.workflow_id} monitoring started")

    async def end(self, success: bool = True) -> None:
        """워크플로우 종료"""
        now = datetime.now()
        self._status_tracker.end_time = now

        duration_ms = 0.0
        if self.start_time:
            duration_ms = (now - self.start_time).total_seconds() * 1000

        event = MonitorEvent(
            event_type=EventType.WORKFLOW_END,
            timestamp=now,
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
        event = self._status_tracker.node_started(node_id, metadata)
        await self._emit_event(event)

    async def node_completed(
        self,
        node_id: str,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        노드 완료

        Args:
            node_id: 노드 ID
            output: 출력 결과
            metadata: 메타데이터
        """
        event = self._status_tracker.node_completed(node_id, output, metadata)
        await self._emit_event(event)

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
        event = self._status_tracker.node_failed(node_id, error, metadata)
        await self._emit_event(event)

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
        event = self._status_tracker.node_skipped(node_id, reason)
        await self._emit_event(event)

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
        status = self._status_tracker.get_status()
        status["total_events"] = len(self.event_history)
        return status

    def get_node_state(self, node_id: str) -> Optional[NodeExecutionState]:
        """특정 노드 상태 조회"""
        return self._status_tracker.get_node_state(node_id)

    def get_all_node_states(self) -> Dict[str, NodeExecutionState]:
        """모든 노드 상태 조회"""
        return self._status_tracker.get_all_node_states()

    def get_recent_events(self, limit: int = 10) -> List[MonitorEvent]:
        """최근 이벤트 조회"""
        return self._event_handler.get_recent_events(limit)

    def get_timeline(self) -> List[Dict[str, Any]]:
        """
        실행 타임라인 생성

        Returns:
            List[Dict]: 타임라인 항목 리스트
        """
        return self._event_handler.get_timeline()

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        성능 요약

        Returns:
            Dict: 성능 메트릭
        """
        return self._performance_metrics.get_performance_summary(self.node_states)

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
        self._event_handler.reset()
        self._status_tracker.reset()
        logger.info(f"Monitor {self.workflow_id} reset")
