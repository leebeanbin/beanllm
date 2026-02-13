"""
Event Handler - 이벤트 처리 모듈
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List

from beanllm.domain.orchestrator.monitor_types import EventType, MonitorEvent
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class EventHandler:
    """이벤트 핸들러"""

    def __init__(self) -> None:
        """Initialize event handler"""
        # Event listeners
        self.listeners: Dict[EventType, List[Callable]] = {
            event_type: [] for event_type in EventType
        }
        self.event_history: List[MonitorEvent] = []

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

    async def emit_event(self, event: MonitorEvent) -> None:
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

    def reset(self) -> None:
        """이벤트 핸들러 리셋"""
        self.event_history.clear()
        self.listeners = {event_type: [] for event_type in EventType}
