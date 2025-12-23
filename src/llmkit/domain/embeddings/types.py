"""
Embeddings Types - 임베딩 데이터 타입
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class EmbeddingResult:
    """Embedding 결과"""

    embeddings: List[List[float]]  # 임베딩 벡터들
    model: str  # 사용된 모델
    usage: Dict[str, int]  # 토큰 사용량 등
