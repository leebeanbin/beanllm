"""
GraphRequest - Graph 요청 DTO
책임: Graph 요청 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass(slots=True, kw_only=True)
class GraphRequest:
    """
    Graph 요청 DTO

    책임:
    - 데이터 구조 정의만
    - 검증 없음 (Handler에서 처리)
    - 비즈니스 로직 없음 (Service에서 처리)
    """

    initial_state: dict[str, object]
    nodes: list[object] = field(default_factory=list)
    edges: dict[str, list[str]] = field(default_factory=dict)
    conditional_edges: dict[str, Callable[..., object]] = field(default_factory=dict)
    entry_point: Optional[str] = None
    enable_cache: bool = True
    verbose: bool = False
    max_iterations: int = 100
    extra_params: dict[str, object] = field(default_factory=dict)
