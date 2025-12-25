"""
MultiAgentResponse - Multi-Agent 응답 DTO
책임: Multi-Agent 응답 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MultiAgentResponse:
    """
    Multi-Agent 응답 DTO

    책임:
    - 데이터 구조 정의만
    - 변환 로직 없음
    """

    final_result: Any
    strategy: str
    intermediate_results: Optional[List[Any]] = None
    all_steps: Optional[List[Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
