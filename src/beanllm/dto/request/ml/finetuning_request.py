"""
Finetuning Request DTOs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from beanllm.domain.finetuning.types import FineTuningConfig, FineTuningJob, TrainingExample


@dataclass(slots=True, kw_only=True)
class PrepareDataRequest:
    """데이터 준비 요청 DTO"""

    examples: list["TrainingExample"]
    output_path: str
    validate: bool = True


@dataclass(slots=True, kw_only=True)
class CreateJobRequest:
    """작업 생성 요청 DTO"""

    config: "FineTuningConfig"


@dataclass(slots=True, kw_only=True)
class GetJobRequest:
    """작업 조회 요청 DTO"""

    job_id: str


@dataclass(slots=True, kw_only=True)
class ListJobsRequest:
    """작업 목록 조회 요청 DTO"""

    limit: int = 20


@dataclass(slots=True, kw_only=True)
class CancelJobRequest:
    """작업 취소 요청 DTO"""

    job_id: str


@dataclass(slots=True, kw_only=True)
class GetMetricsRequest:
    """메트릭 조회 요청 DTO"""

    job_id: str


@dataclass(slots=True, kw_only=True)
class StartTrainingRequest:
    """훈련 시작 요청 DTO"""

    model: str
    training_file: str
    validation_file: Optional[str] = None
    extra_params: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, kw_only=True)
class WaitForCompletionRequest:
    """완료 대기 요청 DTO"""

    job_id: str
    poll_interval: int = 60
    timeout: Optional[int] = None
    callback: Optional[Callable[["FineTuningJob"], None]] = None


@dataclass(slots=True, kw_only=True)
class QuickFinetuneRequest:
    """빠른 파인튜닝 요청 DTO"""

    training_data: list["TrainingExample"]
    model: str = "gpt-3.5-turbo"
    validation_split: float = 0.1
    n_epochs: int = 3
    wait: bool = True
    extra_params: dict[str, object] = field(default_factory=dict)
