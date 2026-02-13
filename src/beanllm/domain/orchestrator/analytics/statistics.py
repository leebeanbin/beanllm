"""
Statistics - 통계 분석 모듈
"""

from __future__ import annotations

from typing import Any, Dict, List

from beanllm.domain.orchestrator.monitor_types import NodeExecutionState, NodeStatus


class StatisticsAnalyzer:
    """통계 분석기"""

    @staticmethod
    def get_node_statistics(
        node_id: str,
        node_metrics: Dict[str, List[float]],
    ) -> Dict[str, Any]:
        """
        특정 노드 통계

        Args:
            node_id: 노드 ID
            node_metrics: 노드 메트릭 딕셔너리

        Returns:
            Dict: 노드 통계
        """
        durations = node_metrics.get(node_id, [])

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

    @staticmethod
    def compare_executions(
        states_a: Dict[str, NodeExecutionState],
        states_b: Dict[str, NodeExecutionState],
        workflow_id_a: str,
        workflow_id_b: str,
    ) -> Dict[str, Any]:
        """
        두 실행 비교

        Args:
            states_a: 첫 번째 워크플로우 노드 상태
            states_b: 두 번째 워크플로우 노드 상태
            workflow_id_a: 첫 번째 워크플로우 ID
            workflow_id_b: 두 번째 워크플로우 ID

        Returns:
            Dict: 비교 결과
        """
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

    @staticmethod
    def get_summary_statistics(
        executions: Dict[str, Dict[str, Any]],
        node_metrics: Dict[str, List[float]],
        agent_metrics: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        전체 요약 통계

        Args:
            executions: 실행 데이터 딕셔너리
            node_metrics: 노드 메트릭 딕셔너리
            agent_metrics: Agent 메트릭 딕셔너리

        Returns:
            Dict: 요약 통계
        """
        total_executions = len(executions)

        if total_executions == 0:
            return {"total_executions": 0}

        # Aggregate metrics
        all_durations = []
        all_success_rates = []

        for exec_data in executions.values():
            node_states = exec_data["node_states"]

            duration = sum(
                s.duration_ms for s in node_states.values() if s.status == NodeStatus.COMPLETED
            )
            all_durations.append(duration)

            success_count = sum(1 for s in node_states.values() if s.status == NodeStatus.COMPLETED)
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
            "total_agents_used": len(agent_metrics),
            "total_nodes_analyzed": len(node_metrics),
        }
