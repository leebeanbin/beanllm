"""
ChainRequest - Chain 요청 DTO
책임: Chain 요청 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True, kw_only=True)
class ChainRequest:
    """
    Chain 요청 DTO

    책임:
    - 데이터 구조 정의만
    - 검증 없음 (Handler에서 처리)
    - 비즈니스 로직 없음 (Service에서 처리)
    """

    chain_type: str
    user_input: Optional[str] = None
    template: Optional[str] = None
    template_vars: dict[str, object] = field(default_factory=dict)
    chains: list[object] = field(default_factory=list)
    model: str = "gpt-4o-mini"
    memory_type: Optional[str] = None
    memory_config: dict[str, object] = field(default_factory=dict)
    tools: list[object] = field(default_factory=list)
    verbose: bool = False
    extra_params: dict[str, object] = field(default_factory=dict)
