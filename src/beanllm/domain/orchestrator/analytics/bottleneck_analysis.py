"""
Bottleneck Analysis - 병목 지점 분석 모듈
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from beanllm.domain.orchestrator.monitor_types import NodeExecutionState, NodeStatus


@dataclass
class BottleneckAnalysis:
    """병목 지점 분석"""

    node_id: str
    duration_ms: float
    percentage_of_total: float
    is_bottleneck: bool
    recommendation: str


class BottleneckAnalyzer:
    """병목 지점 분석기"""

    @staticmethod
    def find_bottlenecks(
        node_states: Dict[str, NodeExecutionState],
        threshold_percentile: float = 0.8,
    ) -> List[BottleneckAnalysis]:
        """
        병목 지점 찾기

        Args:
            node_states: 노드 상태 딕셔너리
            threshold_percentile: 병목으로 간주할 백분위 (0.8 = 상위 20%)

        Returns:
            List[BottleneckAnalysis]: 병목 분석 결과
        """
        # Calculate total duration
        total_duration = sum(
            state.duration_ms
            for state in node_states.values()
            if state.status == NodeStatus.COMPLETED
        )

        if total_duration == 0:
            return []

        # Analyze each node
        bottlenecks = []

        for node_id, state in node_states.items():
            if state.status != NodeStatus.COMPLETED:
                continue

            percentage = (state.duration_ms / total_duration) * 100

            # Check if bottleneck
            is_bottleneck = percentage >= (threshold_percentile * 100)

            # Generate recommendation
            recommendation = ""
            if is_bottleneck:
                if percentage > 50:
                    recommendation = (
                        "Critical bottleneck. Consider parallelization or optimization."
                    )
                elif percentage > 30:
                    recommendation = "Major bottleneck. Review implementation efficiency."
                else:
                    recommendation = "Minor bottleneck. Monitor for optimization opportunities."

            bottlenecks.append(
                BottleneckAnalysis(
                    node_id=node_id,
                    duration_ms=state.duration_ms,
                    percentage_of_total=percentage,
                    is_bottleneck=is_bottleneck,
                    recommendation=recommendation,
                )
            )

        # Sort by duration (descending)
        bottlenecks.sort(key=lambda x: x.duration_ms, reverse=True)

        return bottlenecks
