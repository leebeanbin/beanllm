"""
Agent Utilization Analysis - Agent 활용도 분석 모듈
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class UtilizationStats:
    """Agent 활용도 통계"""

    agent_id: str
    total_executions: int
    total_duration_ms: float
    avg_duration_ms: float
    success_rate: float
    nodes_used: List[str]


class AgentUtilizationAnalyzer:
    """Agent 활용도 분석기"""

    @staticmethod
    def analyze_utilization(
        agent_metrics: Dict[str, Dict[str, Any]],
    ) -> Dict[str, UtilizationStats]:
        """
        Agent 활용도 분석

        Args:
            agent_metrics: Agent 메트릭 딕셔너리

        Returns:
            Dict[str, UtilizationStats]: Agent별 활용도 통계
        """
        utilization = {}

        for agent_id, metrics in agent_metrics.items():
            executions = metrics["executions"]
            if executions == 0:
                continue

            successes = metrics["successes"]
            total_duration = metrics["total_duration_ms"]
            nodes = metrics["nodes"]

            utilization[agent_id] = UtilizationStats(
                agent_id=agent_id,
                total_executions=executions,
                total_duration_ms=total_duration,
                avg_duration_ms=total_duration / executions,
                success_rate=successes / executions,
                nodes_used=list(nodes),
            )

        return utilization
