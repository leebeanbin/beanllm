"""
WorkflowAnalytics - 워크플로우 성능 분석
SOLID 원칙:
- SRP: 성능 분석 및 메트릭 계산만 담당
- OCP: 새로운 분석 메트릭 추가 가능
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from beanllm.utils.logging import get_logger

from .analytics import (
    AgentUtilizationAnalyzer,
    BottleneckAnalysis,
    BottleneckAnalyzer,
    CostAnalyzer,
    PathAnalysis,
    PathAnalyzer,
    StatisticsAnalyzer,
    UtilizationStats,
)
from .workflow_monitor import MonitorEvent, NodeExecutionState, NodeStatus

logger = get_logger(__name__)

# Re-export for backward compatibility
__all__ = [
    "BottleneckAnalysis",
    "UtilizationStats",
    "PathAnalysis",
    "WorkflowAnalytics",
]


class WorkflowAnalytics:
    """
    워크플로우 성능 분석

    책임:
    - 실행 데이터 분석
    - 병목 지점 식별
    - Agent 활용도 분석
    - 비용 분석
    - 최적화 추천

    Example:
        ```python
        analytics = WorkflowAnalytics()

        # Add execution data
        analytics.add_execution(
            workflow_id="wf123",
            node_states=monitor.get_all_node_states(),
            events=monitor.event_history
        )

        # Analyze bottlenecks
        bottlenecks = analytics.find_bottlenecks(workflow_id="wf123")
        for bn in bottlenecks:
            print(f"Bottleneck: {bn.node_id} ({bn.duration_ms}ms)")

        # Agent utilization
        utilization = analytics.analyze_agent_utilization()
        for agent_id, stats in utilization.items():
            print(f"{agent_id}: {stats.success_rate:.1%} success rate")
        ```
    """

    def __init__(self) -> None:
        """Initialize analytics"""
        # Execution data storage
        self.executions: Dict[str, Dict[str, Any]] = {}  # workflow_id -> data

        # Aggregated data
        self.node_metrics: Dict[str, List[float]] = defaultdict(list)  # node_id -> durations
        self.agent_metrics: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "executions": 0,
                "successes": 0,
                "total_duration_ms": 0.0,
                "nodes": set(),
            }
        )

    def add_execution(
        self,
        workflow_id: str,
        node_states: Dict[str, NodeExecutionState],
        events: List[MonitorEvent],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        실행 데이터 추가

        Args:
            workflow_id: 워크플로우 ID
            node_states: 노드 상태 딕셔너리
            events: 이벤트 리스트
            metadata: 메타데이터
        """
        self.executions[workflow_id] = {
            "node_states": node_states,
            "events": events,
            "metadata": metadata or {},
            "added_at": datetime.now(),
        }

        # Update aggregated metrics
        for node_id, state in node_states.items():
            if state.status == NodeStatus.COMPLETED and state.duration_ms > 0:
                self.node_metrics[node_id].append(state.duration_ms)

                # Extract agent_id from metadata if available
                agent_id = state.metadata.get("agent_id")
                if agent_id:
                    self.agent_metrics[agent_id]["executions"] += 1
                    self.agent_metrics[agent_id]["successes"] += 1
                    self.agent_metrics[agent_id]["total_duration_ms"] += state.duration_ms
                    self.agent_metrics[agent_id]["nodes"].add(node_id)

            elif state.status == NodeStatus.FAILED:
                agent_id = state.metadata.get("agent_id")
                if agent_id:
                    self.agent_metrics[agent_id]["executions"] += 1

        logger.debug(f"Added execution data for workflow {workflow_id}")

    def find_bottlenecks(
        self,
        workflow_id: str,
        threshold_percentile: float = 0.8,
    ) -> List[BottleneckAnalysis]:
        """
        병목 지점 찾기

        Args:
            workflow_id: 워크플로우 ID
            threshold_percentile: 병목으로 간주할 백분위 (0.8 = 상위 20%)

        Returns:
            List[BottleneckAnalysis]: 병목 분석 결과
        """
        if workflow_id not in self.executions:
            return []

        node_states = self.executions[workflow_id]["node_states"]
        return BottleneckAnalyzer.find_bottlenecks(node_states, threshold_percentile)

    def analyze_agent_utilization(self) -> Dict[str, UtilizationStats]:
        """
        Agent 활용도 분석

        Returns:
            Dict[str, UtilizationStats]: Agent별 활용도 통계
        """
        return AgentUtilizationAnalyzer.analyze_utilization(self.agent_metrics)

    def analyze_execution_paths(
        self,
        workflow_id: str,
    ) -> List[PathAnalysis]:
        """
        실행 경로 분석

        Args:
            workflow_id: 워크플로우 ID

        Returns:
            List[PathAnalysis]: 경로 분석 결과
        """
        if workflow_id not in self.executions:
            return []

        events = self.executions[workflow_id]["events"]
        node_states = self.executions[workflow_id]["node_states"]
        return PathAnalyzer.analyze_execution_paths(events, node_states)

    def get_node_statistics(self, node_id: str) -> Dict[str, Any]:
        """
        특정 노드 통계

        Args:
            node_id: 노드 ID

        Returns:
            Dict: 노드 통계
        """
        return StatisticsAnalyzer.get_node_statistics(node_id, self.node_metrics)

    def compare_executions(
        self,
        workflow_id_a: str,
        workflow_id_b: str,
    ) -> Dict[str, Any]:
        """
        두 실행 비교

        Args:
            workflow_id_a: 첫 번째 워크플로우 ID
            workflow_id_b: 두 번째 워크플로우 ID

        Returns:
            Dict: 비교 결과
        """
        if workflow_id_a not in self.executions or workflow_id_b not in self.executions:
            return {"error": "One or both workflow IDs not found"}

        states_a = self.executions[workflow_id_a]["node_states"]
        states_b = self.executions[workflow_id_b]["node_states"]
        return StatisticsAnalyzer.compare_executions(
            states_a, states_b, workflow_id_a, workflow_id_b
        )

    def generate_optimization_recommendations(
        self,
        workflow_id: str,
    ) -> List[str]:
        """
        최적화 추천 생성

        Args:
            workflow_id: 워크플로우 ID

        Returns:
            List[str]: 추천 목록
        """
        recommendations = []

        # Find bottlenecks
        bottlenecks = self.find_bottlenecks(workflow_id)
        critical_bottlenecks = [bn for bn in bottlenecks if bn.percentage_of_total > 30]

        if critical_bottlenecks:
            recommendations.append(
                f"🔴 Critical: {len(critical_bottlenecks)} nodes consuming >30% of total time"
            )
            for bn in critical_bottlenecks[:3]:  # Top 3
                recommendations.append(f"   - Optimize {bn.node_id}: {bn.recommendation}")

        # Check for parallelization opportunities
        node_states = self.executions[workflow_id]["node_states"]
        sequential_count = sum(1 for s in node_states.values() if s.status == NodeStatus.COMPLETED)

        if sequential_count > 3:
            recommendations.append(
                "💡 Consider parallelizing independent nodes to reduce total execution time"
            )

        # Agent utilization
        utilization = self.analyze_agent_utilization()
        underutilized = [
            agent_id for agent_id, stats in utilization.items() if stats.total_executions < 2
        ]

        if underutilized:
            recommendations.append(
                f"⚠️  {len(underutilized)} agents are underutilized. Consider consolidation."
            )

        # Success rate
        success_rate = (
            sum(1 for s in node_states.values() if s.status == NodeStatus.COMPLETED)
            / len(node_states)
            if node_states
            else 0.0
        )

        if success_rate < 0.95:
            recommendations.append(
                f"⚠️  Success rate is {success_rate:.1%}. Review error handling and retry logic."
            )

        if not recommendations:
            recommendations.append("✅ Workflow is well-optimized!")

        return recommendations

    def calculate_cost_estimate(
        self,
        workflow_id: str,
        cost_per_second: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        비용 추정

        Args:
            workflow_id: 워크플로우 ID
            cost_per_second: Agent별 초당 비용 (None이면 기본값 사용)

        Returns:
            Dict: 비용 추정
        """
        if workflow_id not in self.executions:
            return {"error": "Workflow not found"}

        node_states = self.executions[workflow_id]["node_states"]
        return CostAnalyzer.calculate_cost_estimate(node_states, workflow_id, cost_per_second)

    def export_analytics_report(self, workflow_id: str) -> Dict[str, Any]:
        """
        완전한 분석 리포트 내보내기

        Args:
            workflow_id: 워크플로우 ID

        Returns:
            Dict: 전체 분석 리포트
        """
        if workflow_id not in self.executions:
            return {"error": "Workflow not found"}

        return {
            "workflow_id": workflow_id,
            "bottlenecks": [
                {
                    "node_id": bn.node_id,
                    "duration_ms": bn.duration_ms,
                    "percentage": bn.percentage_of_total,
                    "is_bottleneck": bn.is_bottleneck,
                    "recommendation": bn.recommendation,
                }
                for bn in self.find_bottlenecks(workflow_id)
            ],
            "agent_utilization": {
                agent_id: {
                    "executions": stats.total_executions,
                    "avg_duration_ms": stats.avg_duration_ms,
                    "success_rate": stats.success_rate,
                    "nodes_used": stats.nodes_used,
                }
                for agent_id, stats in self.analyze_agent_utilization().items()
            },
            "execution_paths": [
                {
                    "path": path.path,
                    "frequency": path.frequency,
                    "avg_duration_ms": path.avg_duration_ms,
                    "success_rate": path.success_rate,
                }
                for path in self.analyze_execution_paths(workflow_id)
            ],
            "recommendations": self.generate_optimization_recommendations(workflow_id),
            "cost_estimate": self.calculate_cost_estimate(workflow_id),
        }

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        전체 요약 통계

        Returns:
            Dict: 요약 통계
        """
        return StatisticsAnalyzer.get_summary_statistics(
            self.executions, self.node_metrics, self.agent_metrics
        )
