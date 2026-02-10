"""
Finetuning Response DTOs
"""

from __future__ import annotations

from typing import Optional

from pydantic import ConfigDict

from beanllm.domain.finetuning.types import FineTuningJob, FineTuningMetrics
from beanllm.dto.response.base_response import BaseResponse


class PrepareDataResponse(BaseResponse):
    """데이터 준비 응답 DTO"""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, arbitrary_types_allowed=True
    )

    file_id: str


class CreateJobResponse(BaseResponse):
    """작업 생성 응답 DTO"""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, arbitrary_types_allowed=True
    )

    job: FineTuningJob


class GetJobResponse(BaseResponse):
    """작업 조회 응답 DTO"""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, arbitrary_types_allowed=True
    )

    job: FineTuningJob


class ListJobsResponse(BaseResponse):
    """작업 목록 조회 응답 DTO"""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, arbitrary_types_allowed=True
    )

    jobs: list[FineTuningJob]


class CancelJobResponse(BaseResponse):
    """작업 취소 응답 DTO"""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, arbitrary_types_allowed=True
    )

    job: FineTuningJob


class GetMetricsResponse(BaseResponse):
    """메트릭 조회 응답 DTO"""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, arbitrary_types_allowed=True
    )

    metrics: list[FineTuningMetrics]


class GetTrainingProgressResponse(BaseResponse):
    """훈련 진행상황 응답 DTO"""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, arbitrary_types_allowed=True
    )

    job: FineTuningJob
    metrics: list[FineTuningMetrics]
    latest_metric: Optional[FineTuningMetrics] = None

    def to_dict(self) -> dict[str, object]:
        """딕셔너리로 변환"""
        return {
            "job": {
                "job_id": self.job.job_id,
                "status": self.job.status.value,
                "model": self.job.model,
            },
            "metrics": [
                {
                    "step": m.step,
                    "train_loss": m.train_loss,
                    "valid_loss": m.valid_loss,
                }
                for m in self.metrics
            ],
            "latest_metric": (
                {
                    "step": self.latest_metric.step,
                    "train_loss": self.latest_metric.train_loss,
                    "valid_loss": self.latest_metric.valid_loss,
                }
                if self.latest_metric
                else None
            ),
        }


class StartTrainingResponse(BaseResponse):
    """훈련 시작 응답 DTO"""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, arbitrary_types_allowed=True
    )

    job: FineTuningJob
