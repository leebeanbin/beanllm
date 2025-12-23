"""
Finetuning Service Interface
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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


class IFinetuningService(ABC):
    """파인튜닝 서비스 인터페이스"""

    @abstractmethod
    async def prepare_data(self, request: "PrepareDataRequest") -> "PrepareDataResponse":
        """데이터 준비 및 업로드"""
        pass

    @abstractmethod
    async def create_job(self, request: "CreateJobRequest") -> "CreateJobResponse":
        """파인튜닝 작업 생성"""
        pass

    @abstractmethod
    async def get_job(self, request: "GetJobRequest") -> "GetJobResponse":
        """작업 상태 조회"""
        pass

    @abstractmethod
    async def list_jobs(self, request: "ListJobsRequest") -> "ListJobsResponse":
        """작업 목록 조회"""
        pass

    @abstractmethod
    async def cancel_job(self, request: "CancelJobRequest") -> "CancelJobResponse":
        """작업 취소"""
        pass

    @abstractmethod
    async def get_metrics(self, request: "GetMetricsRequest") -> "GetMetricsResponse":
        """훈련 메트릭 조회"""
        pass

    @abstractmethod
    async def start_training(self, request: "StartTrainingRequest") -> "StartTrainingResponse":
        """훈련 시작"""
        pass

    @abstractmethod
    async def wait_for_completion(self, request: "WaitForCompletionRequest") -> "GetJobResponse":
        """작업 완료 대기"""
        pass

    @abstractmethod
    async def quick_finetune(self, request: "QuickFinetuneRequest") -> "CreateJobResponse":
        """빠른 파인튜닝 시작"""
        pass
