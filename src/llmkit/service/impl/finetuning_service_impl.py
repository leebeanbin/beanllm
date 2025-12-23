"""
Finetuning Service Implementation
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ...domain.finetuning.providers import BaseFineTuningProvider, OpenAIFineTuningProvider
from ...domain.finetuning.types import FineTuningJob
from ...domain.finetuning.utils import DatasetBuilder, FineTuningManager
from ...dto.request.finetuning_request import (
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
from ...dto.response.finetuning_response import (
    CancelJobResponse,
    CreateJobResponse,
    GetJobResponse,
    GetMetricsResponse,
    ListJobsResponse,
    PrepareDataResponse,
    StartTrainingResponse,
)
from ..finetuning_service import IFinetuningService

if TYPE_CHECKING:
    pass


class FinetuningServiceImpl(IFinetuningService):
    """파인튜닝 서비스 구현체"""

    def __init__(self, provider: Optional[BaseFineTuningProvider] = None):
        """
        Args:
            provider: 파인튜닝 프로바이더 (없으면 OpenAIFineTuningProvider 사용)
        """
        self._provider = provider or OpenAIFineTuningProvider()
        self._manager = FineTuningManager(self._provider)

    async def prepare_data(self, request: "PrepareDataRequest") -> "PrepareDataResponse":
        """데이터 준비 및 업로드"""
        file_id = self._manager.prepare_and_upload(
            examples=request.examples,
            output_path=request.output_path,
            validate=request.validate,
        )
        return PrepareDataResponse(file_id=file_id)

    async def create_job(self, request: "CreateJobRequest") -> "CreateJobResponse":
        """파인튜닝 작업 생성"""
        job = self._provider.create_job(request.config)
        return CreateJobResponse(job=job)

    async def get_job(self, request: "GetJobRequest") -> "GetJobResponse":
        """작업 상태 조회"""
        job = self._provider.get_job(request.job_id)
        return GetJobResponse(job=job)

    async def list_jobs(self, request: "ListJobsRequest") -> "ListJobsResponse":
        """작업 목록 조회"""
        jobs = self._provider.list_jobs(limit=request.limit)
        return ListJobsResponse(jobs=jobs)

    async def cancel_job(self, request: "CancelJobRequest") -> "CancelJobResponse":
        """작업 취소"""
        job = self._provider.cancel_job(request.job_id)
        return CancelJobResponse(job=job)

    async def get_metrics(self, request: "GetMetricsRequest") -> "GetMetricsResponse":
        """훈련 메트릭 조회"""
        metrics = self._provider.get_metrics(request.job_id)
        return GetMetricsResponse(metrics=metrics)

    async def start_training(self, request: "StartTrainingRequest") -> "StartTrainingResponse":
        """훈련 시작"""
        job = self._manager.start_training(
            model=request.model,
            training_file=request.training_file,
            validation_file=request.validation_file,
            **request.kwargs,
        )
        return StartTrainingResponse(job=job)

    async def wait_for_completion(self, request: "WaitForCompletionRequest") -> "GetJobResponse":
        """작업 완료 대기"""
        job = self._manager.wait_for_completion(
            job_id=request.job_id,
            poll_interval=request.poll_interval,
            timeout=request.timeout,
            callback=request.callback,
        )
        return GetJobResponse(job=job)

    async def quick_finetune(self, request: "QuickFinetuneRequest") -> "CreateJobResponse":
        """빠른 파인튜닝 시작"""
        # 데이터 분할
        train_examples, val_examples = DatasetBuilder.split_dataset(
            request.training_data, train_ratio=1 - request.validation_split
        )

        # 데이터 업로드
        train_file = self._manager.prepare_and_upload(train_examples, "train.jsonl")

        val_file = None
        if val_examples:
            val_file = self._manager.prepare_and_upload(val_examples, "val.jsonl")

        # 훈련 시작
        job = self._manager.start_training(
            model=request.model,
            training_file=train_file,
            validation_file=val_file,
            n_epochs=request.n_epochs,
            **request.kwargs,
        )

        # 대기
        if request.wait:

            def progress_callback(j: FineTuningJob) -> None:
                print(f"Status: {j.status.value}, Model: {j.fine_tuned_model or 'N/A'}")

            job = self._manager.wait_for_completion(job.job_id, callback=progress_callback)

        return CreateJobResponse(job=job)
