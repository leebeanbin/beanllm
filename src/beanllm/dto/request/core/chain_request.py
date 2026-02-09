"""
ChainRequest - Chain 요청 DTO
책임: Chain 요청 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True, kw_only=True)
class ChainRequest:
    """
    Chain 요청 DTO

    책임:
    - 데이터 구조 정의만
    - 검증 없음 (Handler에서 처리)
    - 비즈니스 로직 없음 (Service에서 처리)
    """

    chain_type: str  # "basic", "prompt", "sequential", "parallel"
    user_input: Optional[str] = None  # 기본 Chain용
    template: Optional[str] = None  # PromptChain용
    template_vars: Dict[str, Any] = field(default_factory=dict)
    chains: List[Any] = field(default_factory=list)
    model: str = "gpt-4o-mini"
    memory_type: Optional[str] = None
    memory_config: Dict[str, Any] = field(default_factory=dict)
    tools: List[Any] = field(default_factory=list)
    verbose: bool = False
    extra_params: Dict[str, Any] = field(default_factory=dict)
