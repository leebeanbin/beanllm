"""
GraphRequest - Graph 요청 DTO
책임: Graph 요청 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass(slots=True, kw_only=True)
class GraphRequest:
    """
    Graph 요청 DTO

    책임:
    - 데이터 구조 정의만
    - 검증 없음 (Handler에서 처리)
    - 비즈니스 로직 없음 (Service에서 처리)
    """

    initial_state: Dict[str, Any]
    nodes: List[Any] = field(default_factory=list)
    edges: Dict[str, List[str]] = field(default_factory=dict)
    conditional_edges: Dict[str, Callable] = field(default_factory=dict)
    entry_point: Optional[str] = None
    enable_cache: bool = True
    verbose: bool = False
    max_iterations: int = 100
    extra_params: Dict[str, Any] = field(default_factory=dict)
