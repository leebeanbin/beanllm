"""
AgentRequest - 에이전트 요청 DTO
책임: 에이전트 요청 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True, kw_only=True)
class AgentRequest:
    """
    에이전트 요청 DTO

    책임:
    - 데이터 구조 정의만
    - 검증 없음 (Handler에서 처리)
    - 비즈니스 로직 없음 (Service에서 처리)
    """

    task: str
    model: str
    tools: List[Any] = field(default_factory=list)
    tool_registry: Optional[Any] = None  # ToolRegistry 인스턴스
    max_steps: int = 10
    temperature: Optional[float] = None
    system_prompt: Optional[str] = None
    memory: Optional[Any] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)
