"""
Finetuning Handler - 파인튜닝 요청 처리
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional

from ..decorators.error_handler import handle_errors
from ..decorators.validation import validate_input
from ..dto.request.finetuning_request import (
    CancelJobRequest,
    CreateJobRequest,
    GetJobRequest,
    GetMetricsRequest,
    ListJobsRequest,
    PrepareDataRequest,
    QuickFinetuneRequest,
    StartTrainingRequest,
    WaitForCompletionRequest,
)
from ..dto.response.finetuning_response import (
    CancelJobResponse,
    CreateJobResponse,
    GetJobResponse,
    GetMetricsResponse,
    ListJobsResponse,
    PrepareDataResponse,
    StartTrainingResponse,
)
from ..service.finetuning_service import IFinetuningService
from .base_handler import BaseHandler

if TYPE_CHECKING:
    from ..domain.finetuning.types import FineTuningConfig, FineTuningJob, TrainingExample


class FinetuningHandler(BaseHandler):
    """파인튜닝 요청 핸들러"""

    def __init__(self, finetuning_service: IFinetuningService):
        """
        Args:
            finetuning_service: 파인튜닝 서비스
        """
        super().__init__(finetuning_service)
        self._finetuning_service = (
            finetuning_service  # BaseHandler._service와 동일하지만 명시적으로 유지
        )

    @handle_errors(error_message="Prepare data failed")
    @validate_input(
        required_params=["examples", "output_path"],
        param_types={"examples": list, "output_path": str, "validate": bool},
    )
    async def handle_prepare_data(
        self,
        examples: List["TrainingExample"],
        output_path: str,
        validate: bool = True,
    ) -> "PrepareDataResponse":
        """데이터 준비 처리"""
        request = PrepareDataRequest(examples=examples, output_path=output_path, validate=validate)
        return await self._call_service("prepare_data", request)

    @handle_errors(error_message="Create job failed")
    @validate_input(
        required_params=["config"],
        param_types={"config": object},
    )
    async def handle_create_job(self, config: "FineTuningConfig") -> "CreateJobResponse":
        """작업 생성 처리"""
        request = CreateJobRequest(config=config)
        return await self._call_service("create_job", request)

    @handle_errors(error_message="Get job failed")
    @validate_input(
        required_params=["job_id"],
        param_types={"job_id": str},
    )
    async def handle_get_job(self, job_id: str) -> "GetJobResponse":
        """작업 조회 처리"""
        request = GetJobRequest(job_id=job_id)
        return await self._finetuning_service.get_job(request)

    @handle_errors(error_message="List jobs failed")
    @validate_input(
        required_params=[],
        param_types={"limit": int, "status": str},
    )
    async def handle_list_jobs(self, limit: int = 20) -> "ListJobsResponse":
        """작업 목록 조회 처리"""
        request = ListJobsRequest(limit=limit)
        return await self._call_service("list_jobs", request)

    @handle_errors(error_message="Cancel job failed")
    @validate_input(
        required_params=["job_id"],
        param_types={"job_id": str},
    )
    async def handle_cancel_job(self, job_id: str) -> "CancelJobResponse":
        """작업 취소 처리"""
        request = CancelJobRequest(job_id=job_id)
        return await self._call_service("cancel_job", request)

    @handle_errors(error_message="Get metrics failed")
    @validate_input(
        required_params=["job_id"],
        param_types={"job_id": str},
    )
    async def handle_get_metrics(self, job_id: str) -> "GetMetricsResponse":
        """메트릭 조회 처리"""
        request = GetMetricsRequest(job_id=job_id)
        return await self._call_service("get_metrics", request)

    @handle_errors(error_message="Start training failed")
    @validate_input(
        required_params=["model", "training_file"],
        param_types={"model": str, "training_file": str, "validation_file": str},
    )
    async def handle_start_training(
        self,
        model: str,
        training_file: str,
        validation_file: Optional[str] = None,
        **kwargs,
    ) -> "StartTrainingResponse":
        """훈련 시작 처리"""
        request = StartTrainingRequest(
            model=model,
            training_file=training_file,
            validation_file=validation_file,
            **kwargs,
        )
        return await self._finetuning_service.start_training(request)

    @handle_errors(error_message="Wait for completion failed")
    @validate_input(
        required_params=["job_id"],
        param_types={"job_id": str, "poll_interval": int, "timeout": int},
        param_ranges={"poll_interval": (1, None), "timeout": (1, None)},
    )
    async def handle_wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 60,
        timeout: Optional[int] = None,
        callback: Optional[Callable[["FineTuningJob"], None]] = None,
    ) -> "GetJobResponse":
        """완료 대기 처리"""
        request = WaitForCompletionRequest(
            job_id=job_id,
            poll_interval=poll_interval,
            timeout=timeout,
            callback=callback,
        )
        return await self._call_service("wait_for_completion", request)

    @handle_errors(error_message="Quick finetune failed")
    @validate_input(
        required_params=["training_data", "model"],
        param_types={
            "training_data": list,
            "model": str,
            "validation_split": float,
            "n_epochs": int,
        },
        param_ranges={"validation_split": (0.0, 1.0), "n_epochs": (1, None)},
    )
    async def handle_quick_finetune(
        self,
        training_data: List["TrainingExample"],
        model: str = "gpt-3.5-turbo",
        validation_split: float = 0.1,
        n_epochs: int = 3,
        wait: bool = True,
        **kwargs,
    ) -> "CreateJobResponse":
        """빠른 파인튜닝 처리"""
        request = QuickFinetuneRequest(
            training_data=training_data,
            model=model,
            validation_split=validation_split,
            n_epochs=n_epochs,
            wait=wait,
            **kwargs,
        )
        return await self._call_service("quick_finetune", request)
