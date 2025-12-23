"""
Hybrid Types - 하이브리드 모델 데이터 타입
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class HybridModelInfo:
    """통합 모델 정보"""

    model_id: str
    provider: str
    display_name: str

    # 메타데이터
    supports_streaming: bool = True
    supports_temperature: bool = True
    supports_max_tokens: bool = True
    uses_max_completion_tokens: bool = False
    max_tokens: Optional[int] = None

    # 추가 정보
    tier: Optional[str] = None
    speed: Optional[str] = None

    # 소스 정보
    source: str = "unknown"  # "local", "api", "inferred"
    inference_confidence: float = 0.0
    matched_patterns: List[str] = None

    # 시간 정보
    discovered_at: Optional[str] = None
    last_seen: Optional[str] = None

    def __post_init__(self):
        if self.matched_patterns is None:
            self.matched_patterns = []
