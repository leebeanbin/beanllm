"""
ChainResponse - Chain 응답 DTO
책임: Chain 응답 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChainResponse:
    """
    Chain 응답 DTO

    책임:
    - 데이터 구조 정의만
    - 변환 로직 없음
    """

    output: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None
