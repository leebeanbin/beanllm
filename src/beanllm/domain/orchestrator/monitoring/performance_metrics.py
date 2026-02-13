"""
Performance Metrics - 성능 메트릭 모듈
"""

from __future__ import annotations

import heapq
from typing import Any, Dict, List

from beanllm.domain.orchestrator.monitor_types import NodeExecutionState, NodeStatus


class PerformanceMetrics:
    """성능 메트릭 계산기"""

    @staticmethod
    def get_performance_summary(
        node_states: Dict[str, NodeExecutionState],
    ) -> Dict[str, Any]:
        """
        성능 요약

        Args:
            node_states: 노드 상태 딕셔너리

        Returns:
            Dict: 성능 메트릭
        """
        if not node_states:
            return {}

        completed_states = [s for s in node_states.values() if s.status == NodeStatus.COMPLETED]

        if not completed_states:
            return {"completed_nodes": 0}

        durations = [s.duration_ms for s in completed_states if s.duration_ms > 0]

        avg_duration = sum(durations) / len(durations) if durations else 0.0
        min_duration = min(durations) if durations else 0.0
        max_duration = max(durations) if durations else 0.0

        # Slowest nodes
        slowest = heapq.nlargest(5, completed_states, key=lambda s: s.duration_ms)

        return {
            "completed_nodes": len(completed_states),
            "avg_duration_ms": avg_duration,
            "min_duration_ms": min_duration,
            "max_duration_ms": max_duration,
            "slowest_nodes": [
                {"node_id": s.node_id, "duration_ms": s.duration_ms} for s in slowest
            ],
        }
