"""
Evaluation Response DTOs
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import ConfigDict

from beanllm.domain.evaluation.results import BatchEvaluationResult
from beanllm.dto.response.base_response import BaseResponse


class EvaluationResponse(BaseResponse):
    """평가 응답 DTO"""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    result: BatchEvaluationResult

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return self.result.to_dict()


class BatchEvaluationResponse(BaseResponse):
    """배치 평가 응답 DTO"""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    results: List[BatchEvaluationResult]

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "results": [r.to_dict() for r in self.results],
            "count": len(self.results),
        }
