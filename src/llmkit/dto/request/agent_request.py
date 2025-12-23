"""
AgentRequest - 에이전트 요청 DTO
책임: 에이전트 요청 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
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
    tools: Optional[List[Any]] = None
    tool_registry: Optional[Any] = None  # ToolRegistry 인스턴스
    max_steps: int = 10
    temperature: Optional[float] = None
    system_prompt: Optional[str] = None
    memory: Optional[Any] = None
    extra_params: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """기본값 설정"""
        if self.tools is None:
            self.tools = []
        if self.extra_params is None:
            self.extra_params = {}
