"""
GraphRequest - Graph 요청 DTO
책임: Graph 요청 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class GraphRequest:
    """
    Graph 요청 DTO

    책임:
    - 데이터 구조 정의만
    - 검증 없음 (Handler에서 처리)
    - 비즈니스 로직 없음 (Service에서 처리)
    """

    initial_state: Dict[str, Any]
    nodes: Optional[List[Any]] = None  # BaseNode 리스트
    edges: Optional[Dict[str, List[str]]] = None  # node_name -> [next_nodes]
    conditional_edges: Optional[Dict[str, Callable]] = None  # node_name -> condition_func
    entry_point: Optional[str] = None
    enable_cache: bool = True
    verbose: bool = False
    max_iterations: int = 100
    extra_params: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """기본값 설정"""
        if self.nodes is None:
            self.nodes = []
        if self.edges is None:
            self.edges = {}
        if self.conditional_edges is None:
            self.conditional_edges = {}
        if self.extra_params is None:
            self.extra_params = {}
