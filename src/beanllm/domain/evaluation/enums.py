"""
Evaluation Enums - 평가 관련 열거형
"""

from enum import Enum


class MetricType(Enum):
    """메트릭 타입"""

    SIMILARITY = "similarity"  # 텍스트 유사도
    SEMANTIC = "semantic"  # 의미론적 유사도
    QUALITY = "quality"  # 품질 평가
    RAG = "rag"  # RAG 전용
    CUSTOM = "custom"  # 사용자 정의
