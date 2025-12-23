"""
GraphResponse - Graph 응답 DTO
책임: Graph 응답 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GraphResponse:
    """
    Graph 응답 DTO

    책임:
    - 데이터 구조 정의만
    - 변환 로직 없음
    """

    final_state: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    cache_stats: Optional[Dict[str, Any]] = None
    visited_nodes: List[str] = field(default_factory=list)
    iterations: int = 0
