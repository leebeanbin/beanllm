"""
WorkflowAnalytics - 워크플로우 성능 분석
SOLID 원칙:
- SRP: 성능 분석 및 메트릭 계산만 담당
- OCP: 새로운 분석 메트릭 추가 가능
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from beanllm.utils.logging import get_logger

from .workflow_monitor import MonitorEvent, NodeExecutionState, NodeStatus

logger = get_logger(__name__)


@dataclass
class BottleneckAnalysis:
    """병목 지점 분석"""

    node_id: str
    duration_ms: float
    percentage_of_total: float
    is_bottleneck: bool
    recommendation: str


@dataclass
class UtilizationStats:
    """Agent 활용도 통계"""

    agent_id: str
    total_executions: int
    total_duration_ms: float
    avg_duration_ms: float
    success_rate: float
    nodes_used: List[str]


@dataclass
class PathAnalysis:
    """실행 경로 분석"""

    path: List[str]  # Node IDs
    frequency: int
    avg_duration_ms: float
    success_rate: float


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
                    recommendation = "Critical bottleneck. Consider parallelization or optimization."
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

    def analyze_agent_utilization(self) -> Dict[str, UtilizationStats]:
        """
        Agent 활용도 분석

        Returns:
            Dict[str, UtilizationStats]: Agent별 활용도 통계
        """
        utilization = {}

        for agent_id, metrics in self.agent_metrics.items():
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

    def get_node_statistics(self, node_id: str) -> Dict[str, Any]:
        """
        특정 노드 통계

        Args:
            node_id: 노드 ID

        Returns:
            Dict: 노드 통계
        """
        durations = self.node_metrics.get(node_id, [])

        if not durations:
            return {"node_id": node_id, "executions": 0}

        return {
            "node_id": node_id,
            "executions": len(durations),
            "avg_duration_ms": sum(durations) / len(durations),
            "min_duration_ms": min(durations),
            "max_duration_ms": max(durations),
            "total_duration_ms": sum(durations),
        }

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

        # Total durations
        duration_a = sum(
            s.duration_ms for s in states_a.values() if s.status == NodeStatus.COMPLETED
        )
        duration_b = sum(
            s.duration_ms for s in states_b.values() if s.status == NodeStatus.COMPLETED
        )

        # Success rates
        success_a = sum(1 for s in states_a.values() if s.status == NodeStatus.COMPLETED)
        success_b = sum(1 for s in states_b.values() if s.status == NodeStatus.COMPLETED)

        total_a = len(states_a)
        total_b = len(states_b)

        return {
            "workflow_a": {
                "workflow_id": workflow_id_a,
                "total_duration_ms": duration_a,
                "success_rate": success_a / total_a if total_a > 0 else 0.0,
                "node_count": total_a,
            },
            "workflow_b": {
                "workflow_id": workflow_id_b,
                "total_duration_ms": duration_b,
                "success_rate": success_b / total_b if total_b > 0 else 0.0,
                "node_count": total_b,
            },
            "comparison": {
                "duration_diff_ms": duration_b - duration_a,
                "duration_diff_percent": (
                    ((duration_b - duration_a) / duration_a * 100) if duration_a > 0 else 0.0
                ),
                "faster": workflow_id_a if duration_a < duration_b else workflow_id_b,
            },
        }

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
        sequential_count = sum(
            1 for s in node_states.values() if s.status == NodeStatus.COMPLETED
        )

        if sequential_count > 3:
            recommendations.append(
                "💡 Consider parallelizing independent nodes to reduce total execution time"
            )

        # Agent utilization
        utilization = self.analyze_agent_utilization()
        underutilized = [
            agent_id
            for agent_id, stats in utilization.items()
            if stats.total_executions < 2
        ]

        if underutilized:
            recommendations.append(
                f"⚠️  {len(underutilized)} agents are underutilized. Consider consolidation."
            )

        # Success rate
        success_rate = sum(
            1 for s in node_states.values() if s.status == NodeStatus.COMPLETED
        ) / len(node_states)

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

        # Default costs (USD per second)
        default_costs = {
            "gpt-4": 0.03 / 1000,  # $0.03 per 1K tokens ~ rough estimate
            "gpt-4o": 0.005 / 1000,
            "gpt-4o-mini": 0.0005 / 1000,
            "default": 0.001 / 1000,
        }

        cost_per_second = cost_per_second or default_costs

        total_cost = 0.0
        node_costs = {}

        for node_id, state in node_states.items():
            if state.status != NodeStatus.COMPLETED:
                continue

            agent_id = state.metadata.get("agent_id", "default")
            model = state.metadata.get("model", "default")

            # Get cost rate
            cost_rate = cost_per_second.get(model, cost_per_second.get("default", 0.0))

            # Calculate cost
            duration_seconds = state.duration_ms / 1000
            node_cost = duration_seconds * cost_rate

            node_costs[node_id] = node_cost
            total_cost += node_cost

        return {
            "workflow_id": workflow_id,
            "total_cost_usd": total_cost,
            "node_costs": node_costs,
            "currency": "USD",
            "note": "Costs are rough estimates based on execution time",
        }

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
        total_executions = len(self.executions)

        if total_executions == 0:
            return {"total_executions": 0}

        # Aggregate metrics
        all_durations = []
        all_success_rates = []

        for exec_data in self.executions.values():
            node_states = exec_data["node_states"]

            duration = sum(
                s.duration_ms for s in node_states.values() if s.status == NodeStatus.COMPLETED
            )
            all_durations.append(duration)

            success_count = sum(
                1 for s in node_states.values() if s.status == NodeStatus.COMPLETED
            )
            total_count = len(node_states)
            success_rate = success_count / total_count if total_count > 0 else 0.0
            all_success_rates.append(success_rate)

        return {
            "total_executions": total_executions,
            "avg_duration_ms": sum(all_durations) / len(all_durations) if all_durations else 0.0,
            "min_duration_ms": min(all_durations) if all_durations else 0.0,
            "max_duration_ms": max(all_durations) if all_durations else 0.0,
            "avg_success_rate": (
                sum(all_success_rates) / len(all_success_rates) if all_success_rates else 0.0
            ),
            "total_agents_used": len(self.agent_metrics),
            "total_nodes_analyzed": len(self.node_metrics),
        }
