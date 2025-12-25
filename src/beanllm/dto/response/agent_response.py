"""
AgentResponse - 에이전트 응답 DTO
책임: 에이전트 응답 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class AgentResponse:
    """
    에이전트 응답 DTO

    책임:
    - 응답 데이터 구조 정의만
    - 변환 로직 없음 (Service에서 처리)
    """

    answer: str
    steps: List[Any]  # AgentStep 타입
    total_steps: int
    success: bool = True
    error: Optional[str] = None
