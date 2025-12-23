"""
StateGraphRequest - StateGraph 요청 DTO
책임: StateGraph 요청 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, Union

from ...domain.state_graph import END


@dataclass
class StateGraphRequest:
    """
    StateGraph 요청 DTO

    책임:
    - 데이터 구조 정의만
    - 검증 없음 (Handler에서 처리)
    - 비즈니스 로직 없음 (Service에서 처리)
    """

    initial_state: Dict[str, Any]
    state_schema: Optional[Type] = None
    nodes: Optional[Dict[str, Callable]] = None  # node_name -> node_func
    edges: Optional[Dict[str, Union[str, Type[END]]]] = None  # from_node -> to_node
    conditional_edges: Optional[Dict[str, tuple]] = (
        None  # from_node -> (condition_func, edge_mapping)
    )
    entry_point: Optional[str] = None
    execution_id: Optional[str] = None
    resume_from: Optional[str] = None
    max_iterations: int = 100
    enable_checkpointing: bool = False
    checkpoint_dir: Optional[Path] = None
    debug: bool = False
    extra_params: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """기본값 설정"""
        if self.nodes is None:
            self.nodes = {}
        if self.edges is None:
            self.edges = {}
        if self.conditional_edges is None:
            self.conditional_edges = {}
        if self.extra_params is None:
            self.extra_params = {}
