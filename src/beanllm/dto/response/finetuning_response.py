"""
Finetuning Response DTOs
"""

from __future__ import annotations

from typing import Any, Dict, List

from ...domain.finetuning.types import FineTuningJob, FineTuningMetrics


class PrepareDataResponse:
    """데이터 준비 응답 DTO"""

    def __init__(self, file_id: str):
        self.file_id = file_id


class CreateJobResponse:
    """작업 생성 응답 DTO"""

    def __init__(self, job: FineTuningJob):
        self.job = job


class GetJobResponse:
    """작업 조회 응답 DTO"""

    def __init__(self, job: FineTuningJob):
        self.job = job


class ListJobsResponse:
    """작업 목록 조회 응답 DTO"""

    def __init__(self, jobs: List[FineTuningJob]):
        self.jobs = jobs


class CancelJobResponse:
    """작업 취소 응답 DTO"""

    def __init__(self, job: FineTuningJob):
        self.job = job


class GetMetricsResponse:
    """메트릭 조회 응답 DTO"""

    def __init__(self, metrics: List[FineTuningMetrics]):
        self.metrics = metrics


class GetTrainingProgressResponse:
    """훈련 진행상황 응답 DTO"""

    def __init__(self, job: FineTuningJob, metrics: List[FineTuningMetrics]):
        self.job = job
        self.metrics = metrics
        self.latest_metric = metrics[-1] if metrics else None

    def to_dict(self) -> Dict[str, Any]:
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


class StartTrainingResponse:
    """훈련 시작 응답 DTO"""

    def __init__(self, job: FineTuningJob):
        self.job = job
