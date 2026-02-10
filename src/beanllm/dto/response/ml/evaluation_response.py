"""
Evaluation Response DTOs
"""

from __future__ import annotations

from pydantic import ConfigDict

from beanllm.domain.evaluation.results import BatchEvaluationResult
from beanllm.dto.response.base_response import BaseResponse


class EvaluationResponse(BaseResponse):
    """평가 응답 DTO"""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, arbitrary_types_allowed=True
    )

    result: BatchEvaluationResult

    def to_dict(self) -> dict[str, object]:
        """딕셔너리로 변환"""
        return self.result.to_dict()


class BatchEvaluationResponse(BaseResponse):
    """배치 평가 응답 DTO"""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, arbitrary_types_allowed=True
    )

    results: list[BatchEvaluationResult]

    def to_dict(self) -> dict[str, object]:
        """딕셔너리로 변환"""
        return {
            "results": [r.to_dict() for r in self.results],
            "count": len(self.results),
        }
