"""
Finetuning Request DTOs
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List, Optional

if TYPE_CHECKING:
    from ...domain.finetuning.types import FineTuningConfig, FineTuningJob, TrainingExample


class PrepareDataRequest:
    """데이터 준비 요청 DTO"""

    def __init__(
        self,
        examples: List["TrainingExample"],
        output_path: str,
        validate: bool = True,
    ):
        self.examples = examples
        self.output_path = output_path
        self.validate = validate


class CreateJobRequest:
    """작업 생성 요청 DTO"""

    def __init__(self, config: "FineTuningConfig"):
        self.config = config


class GetJobRequest:
    """작업 조회 요청 DTO"""

    def __init__(self, job_id: str):
        self.job_id = job_id


class ListJobsRequest:
    """작업 목록 조회 요청 DTO"""

    def __init__(self, limit: int = 20):
        self.limit = limit


class CancelJobRequest:
    """작업 취소 요청 DTO"""

    def __init__(self, job_id: str):
        self.job_id = job_id


class GetMetricsRequest:
    """메트릭 조회 요청 DTO"""

    def __init__(self, job_id: str):
        self.job_id = job_id


class StartTrainingRequest:
    """훈련 시작 요청 DTO"""

    def __init__(
        self,
        model: str,
        training_file: str,
        validation_file: Optional[str] = None,
        **kwargs: Any,
    ):
        self.model = model
        self.training_file = training_file
        self.validation_file = validation_file
        self.kwargs = kwargs


class WaitForCompletionRequest:
    """완료 대기 요청 DTO"""

    def __init__(
        self,
        job_id: str,
        poll_interval: int = 60,
        timeout: Optional[int] = None,
        callback: Optional[Callable[["FineTuningJob"], None]] = None,
    ):
        self.job_id = job_id
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.callback = callback


class QuickFinetuneRequest:
    """빠른 파인튜닝 요청 DTO"""

    def __init__(
        self,
        training_data: List["TrainingExample"],
        model: str = "gpt-3.5-turbo",
        validation_split: float = 0.1,
        n_epochs: int = 3,
        wait: bool = True,
        **kwargs: Any,
    ):
        self.training_data = training_data
        self.model = model
        self.validation_split = validation_split
        self.n_epochs = n_epochs
        self.wait = wait
        self.kwargs = kwargs
