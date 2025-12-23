"""
MultiAgentRequest - Multi-Agent 요청 DTO
책임: Multi-Agent 요청 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
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
    agents: Optional[List[Any]] = None  # Agent 리스트
    agent_order: Optional[List[str]] = None  # 순차 실행용 순서
    agent_ids: Optional[List[str]] = None  # 병렬/토론 실행용 agent IDs
    manager_id: Optional[str] = None  # 계층적 실행용 매니저 ID
    worker_ids: Optional[List[str]] = None  # 계층적 실행용 워커 IDs
    aggregation: str = "vote"  # 병렬 실행용 집계 방법
    rounds: int = 3  # 토론 실행용 라운드 수
    judge_id: Optional[str] = None  # 토론 실행용 판정자 ID
    judge_agent: Optional[Any] = None  # 토론 실행용 판정자 Agent (직접 전달)
    extra_params: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """기본값 설정"""
        if self.agents is None:
            self.agents = []
        if self.agent_order is None:
            self.agent_order = []
        if self.agent_ids is None:
            self.agent_ids = []
        if self.worker_ids is None:
            self.worker_ids = []
        if self.extra_params is None:
            self.extra_params = {}
