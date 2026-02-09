"""
StateGraphRequest - StateGraph 요청 DTO
책임: StateGraph 요청 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, Union

from beanllm.domain.state_graph import END


@dataclass(slots=True, kw_only=True)
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
    nodes: Dict[str, Callable] = field(default_factory=dict)
    edges: Dict[str, Union[str, Type[END]]] = field(default_factory=dict)
    conditional_edges: Dict[str, tuple] = field(default_factory=dict)
    entry_point: Optional[str] = None
    execution_id: Optional[str] = None
    resume_from: Optional[str] = None
    max_iterations: int = 100
    enable_checkpointing: bool = False
    checkpoint_dir: Optional[Path] = None
    debug: bool = False
    extra_params: Dict[str, Any] = field(default_factory=dict)
