"""
Parameter Management - 파라미터 공간 및 결과 정의

최적화에 사용되는 파라미터 타입, 공간, 결과 데이터 구조.
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ParameterType(Enum):
    """파라미터 타입"""

    INTEGER = "integer"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


@dataclass
class ParameterSpace:
    """
    파라미터 공간 정의

    Example:
        ```python
        top_k = ParameterSpace(
            name="top_k",
            type=ParameterType.INTEGER,
            low=1,
            high=20
        )
        threshold = ParameterSpace(
            name="score_threshold",
            type=ParameterType.FLOAT,
            low=0.0,
            high=1.0
        )
        ```
    """

    name: str
    type: ParameterType
    low: Optional[float] = None
    high: Optional[float] = None
    categories: Optional[List[Any]] = None
    default: Optional[Any] = None

    def __post_init__(self) -> None:
        """검증"""
        if self.type in [ParameterType.INTEGER, ParameterType.FLOAT]:
            if self.low is None or self.high is None:
                raise ValueError(f"{self.name}: low and high are required for {self.type}")
        elif self.type == ParameterType.CATEGORICAL:
            if not self.categories:
                raise ValueError(f"{self.name}: categories are required for CATEGORICAL")

    def sample(self) -> Any:
        """랜덤 샘플링"""
        if self.type == ParameterType.INTEGER:
            assert self.low is not None and self.high is not None
            return random.randint(int(self.low), int(self.high))
        elif self.type == ParameterType.FLOAT:
            assert self.low is not None and self.high is not None
            return random.uniform(self.low, self.high)
        elif self.type == ParameterType.CATEGORICAL:
            assert self.categories is not None
            return random.choice(self.categories)
        elif self.type == ParameterType.BOOLEAN:
            return random.choice([True, False])
        return None


@dataclass
class OptimizationResult:
    """
    최적화 결과

    Attributes:
        best_params: 최적 파라미터
        best_score: 최고 점수
        total_trials: 총 시행 횟수
        history: 최적화 히스토리
        method: 사용된 최적화 방법
        metadata: 추가 메타데이터
    """

    best_params: Dict[str, Any]
    best_score: float
    total_trials: int
    history: List[Dict[str, Any]] = field(default_factory=list)
    method: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_top_n(self, n: int = 5) -> List[Dict[str, Any]]:
        """상위 N개 결과 반환"""
        return heapq.nlargest(n, self.history, key=lambda x: x["score"])
