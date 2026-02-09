"""
MultiAgentRequest - Multi-Agent 요청 DTO
책임: Multi-Agent 요청 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True, kw_only=True)
class MultiAgentRequest:
    """
    Multi-Agent 요청 DTO

    책임:
    - 데이터 구조 정의만
    - 검증 없음 (Handler에서 처리)
    - 비즈니스 로직 없음 (Service에서 처리)
    """

    strategy: str  # "sequential", "parallel", "hierarchical", "debate"
    task: str
    agents: List[Any] = field(default_factory=list)
    agent_order: List[str] = field(default_factory=list)
    agent_ids: List[str] = field(default_factory=list)
    manager_id: Optional[str] = None
    worker_ids: List[str] = field(default_factory=list)
    aggregation: str = "vote"
    rounds: int = 3
    judge_id: Optional[str] = None
    judge_agent: Optional[Any] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)
