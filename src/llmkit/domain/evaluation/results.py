"""
Evaluation Results - 평가 결과 데이터 구조
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvaluationResult:
    """평가 결과"""

    metric_name: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    explanation: Optional[str] = None

    def __repr__(self) -> str:
        return f"{self.metric_name}: {self.score:.4f}"


@dataclass
class BatchEvaluationResult:
    """배치 평가 결과"""

    results: List[EvaluationResult]
    average_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_metric(self, metric_name: str) -> Optional[EvaluationResult]:
        """특정 메트릭 결과 가져오기"""
        for result in self.results:
            if result.metric_name == metric_name:
                return result
        return None

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "results": [
                {
                    "metric": r.metric_name,
                    "score": r.score,
                    "metadata": r.metadata,
                    "explanation": r.explanation,
                }
                for r in self.results
            ],
            "average_score": self.average_score,
            "metadata": self.metadata,
        }
