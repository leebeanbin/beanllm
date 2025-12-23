"""
ChatRequest - 채팅 요청 DTO
책임: 채팅 요청 데이터만 전달 (검증, 비즈니스 로직 없음)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ChatRequest:
    """
    채팅 요청 DTO

    책임:
    - 데이터 구조 정의만
    - 검증 없음 (Handler에서 처리)
    - 비즈니스 로직 없음 (Service에서 처리)
    """

    messages: List[Dict[str, str]]
    model: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    system: Optional[str] = None
    stream: bool = False
    extra_params: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """기본값 설정"""
        if self.extra_params is None:
            self.extra_params = {}
