"""
Cost Analysis - 비용 분석 모듈
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from beanllm.domain.orchestrator.monitor_types import NodeExecutionState, NodeStatus


class CostAnalyzer:
    """비용 분석기"""

    @staticmethod
    def calculate_cost_estimate(
        node_states: Dict[str, NodeExecutionState],
        workflow_id: str,
        cost_per_second: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        비용 추정

        Args:
            node_states: 노드 상태 딕셔너리
            workflow_id: 워크플로우 ID
            cost_per_second: Agent별 초당 비용 (None이면 기본값 사용)

        Returns:
            Dict: 비용 추정
        """
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
