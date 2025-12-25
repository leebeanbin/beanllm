"""
Evaluation Response DTOs
"""

from __future__ import annotations

from typing import List

from ...domain.evaluation.results import BatchEvaluationResult


class EvaluationResponse:
    """평가 응답 DTO"""

    def __init__(self, result: BatchEvaluationResult):
        self.result = result

    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        return self.result.to_dict()


class BatchEvaluationResponse:
    """배치 평가 응답 DTO"""

    def __init__(self, results: List[BatchEvaluationResult]):
        self.results = results

    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        return {
            "results": [r.to_dict() for r in self.results],
            "count": len(self.results),
        }
