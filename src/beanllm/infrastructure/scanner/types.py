"""
Scanner Types - 스캐너 데이터 타입
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ScannedModel:
    """API에서 스캔된 모델 정보"""

    model_id: str
    provider: str
    created_at: Optional[str] = None
    raw_data: Optional[Dict] = None
