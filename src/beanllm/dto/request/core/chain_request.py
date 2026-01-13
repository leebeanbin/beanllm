"""
ChainRequest - Chain 요청 DTO
책임: Chain 요청 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
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
    template_vars: Optional[Dict[str, Any]] = None  # PromptChain용
    chains: Optional[List[Any]] = None  # SequentialChain, ParallelChain용
    model: str = "gpt-4o-mini"
    memory_type: Optional[str] = None
    memory_config: Optional[Dict[str, Any]] = None
    tools: Optional[List[Any]] = None
    verbose: bool = False
    extra_params: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """기본값 설정"""
        if self.template_vars is None:
            self.template_vars = {}
        if self.chains is None:
            self.chains = []
        if self.tools is None:
            self.tools = []
        if self.memory_config is None:
            self.memory_config = {}
        if self.extra_params is None:
            self.extra_params = {}
