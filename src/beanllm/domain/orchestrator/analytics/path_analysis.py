"""
Path Analysis - 실행 경로 분석 모듈
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from beanllm.domain.orchestrator.monitor_types import (
    MonitorEvent,
    NodeExecutionState,
    NodeStatus,
)


@dataclass
class PathAnalysis:
    """실행 경로 분석"""

    path: List[str]  # Node IDs
    frequency: int
    avg_duration_ms: float
    success_rate: float


class PathAnalyzer:
    """실행 경로 분석기"""

    @staticmethod
    def analyze_execution_paths(
        events: List[MonitorEvent],
        node_states: Dict[str, NodeExecutionState],
    ) -> List[PathAnalysis]:
        """
        실행 경로 분석

        Args:
            events: 이벤트 리스트
            node_states: 노드 상태 딕셔너리

        Returns:
            List[PathAnalysis]: 경로 분석 결과
        """
        # Extract execution order from events
        node_order = []
        for event in events:
            if event.node_id and event.event_type.value == "node_start":
                node_order.append(event.node_id)

        if not node_order:
            return []

        # For now, we have one path (future: handle conditional branches)
        total_duration = sum(
            state.duration_ms
            for state in node_states.values()
            if state.status == NodeStatus.COMPLETED
        )

        success_count = sum(
            1 for state in node_states.values() if state.status == NodeStatus.COMPLETED
        )
        total_count = len(node_states)

        path_analysis = PathAnalysis(
            path=node_order,
            frequency=1,
            avg_duration_ms=total_duration,
            success_rate=success_count / total_count if total_count > 0 else 0.0,
        )

        return [path_analysis]
