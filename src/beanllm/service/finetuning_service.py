"""
Finetuning Service Interface

파인튜닝 관련 비즈니스 로직의 계약을 정의합니다.
Handler 레이어는 이 인터페이스에만 의존하며,
실제 구현체는 service/impl/ml/ 에 위치합니다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from beanllm.dto.request.ml.finetuning_request import (
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
    from beanllm.dto.response.ml.finetuning_response import (
        CancelJobResponse,
        CreateJobResponse,
        GetJobResponse,
        GetMetricsResponse,
        ListJobsResponse,
        PrepareDataResponse,
        StartTrainingResponse,
    )


class IFinetuningService(ABC):
    """파인튜닝 서비스 인터페이스

    LLM 모델 파인튜닝의 전체 라이프사이클을 관리합니다.
    데이터 준비, 작업 생성/조회/취소, 훈련 시작 및 완료 대기를 포함합니다.

    Example:
        >>> service: IFinetuningService = factory.create_finetuning_service()
        >>> response = await service.prepare_data(
        ...     PrepareDataRequest(examples=examples, output_path="train.jsonl")
        ... )
        >>> job = await service.start_training(
        ...     StartTrainingRequest(model="gpt-3.5-turbo", training_file=response.file_id)
        ... )
    """

    @abstractmethod
    async def prepare_data(self, request: "PrepareDataRequest") -> "PrepareDataResponse":
        """훈련 데이터를 JSONL 형식으로 준비하고 업로드합니다.

        Args:
            request: 훈련 예제 목록, 출력 경로, 검증 여부를 포함하는 요청

        Returns:
            PrepareDataResponse: 업로드된 파일 ID 및 검증 결과

        Raises:
            ValueError: 데이터 검증 실패 시
        """

    @abstractmethod
    async def create_job(self, request: "CreateJobRequest") -> "CreateJobResponse":
        """파인튜닝 작업을 생성합니다.

        Args:
            request: 모델명, 훈련 파일 ID, 하이퍼파라미터를 포함하는 요청

        Returns:
            CreateJobResponse: 생성된 작업 ID 및 상태

        Raises:
            RuntimeError: 작업 생성 실패 시
        """

    @abstractmethod
    async def get_job(self, request: "GetJobRequest") -> "GetJobResponse":
        """작업 상태를 조회합니다.

        Args:
            request: 조회할 작업 ID를 포함하는 요청

        Returns:
            GetJobResponse: 작업 상태 및 메타데이터

        Raises:
            ValueError: 존재하지 않는 작업 ID
        """

    @abstractmethod
    async def list_jobs(self, request: "ListJobsRequest") -> "ListJobsResponse":
        """파인튜닝 작업 목록을 조회합니다.

        Args:
            request: 필터 및 페이지네이션 옵션을 포함하는 요청

        Returns:
            ListJobsResponse: 작업 목록 및 페이지네이션 정보
        """

    @abstractmethod
    async def cancel_job(self, request: "CancelJobRequest") -> "CancelJobResponse":
        """진행 중인 파인튜닝 작업을 취소합니다.

        Args:
            request: 취소할 작업 ID를 포함하는 요청

        Returns:
            CancelJobResponse: 취소 결과 및 최종 상태

        Raises:
            ValueError: 이미 완료/취소된 작업
        """

    @abstractmethod
    async def get_metrics(self, request: "GetMetricsRequest") -> "GetMetricsResponse":
        """훈련 메트릭(loss, accuracy 등)을 조회합니다.

        Args:
            request: 작업 ID를 포함하는 요청

        Returns:
            GetMetricsResponse: 훈련 스텝별 메트릭 목록
        """

    @abstractmethod
    async def start_training(self, request: "StartTrainingRequest") -> "StartTrainingResponse":
        """데이터 준비 없이 즉시 훈련을 시작합니다.

        Args:
            request: 모델명, 훈련 파일, 검증 파일 등을 포함하는 요청

        Returns:
            StartTrainingResponse: 생성된 작업 정보

        Raises:
            RuntimeError: 훈련 시작 실패 시
        """

    @abstractmethod
    async def wait_for_completion(self, request: "WaitForCompletionRequest") -> "GetJobResponse":
        """작업이 완료될 때까지 폴링하며 대기합니다.

        Args:
            request: 작업 ID, 폴링 간격, 타임아웃을 포함하는 요청

        Returns:
            GetJobResponse: 완료된 작업 상태

        Raises:
            TimeoutError: 타임아웃 초과 시
            RuntimeError: 작업 실패 시
        """

    @abstractmethod
    async def quick_finetune(self, request: "QuickFinetuneRequest") -> "CreateJobResponse":
        """데이터 준비부터 훈련 시작까지 한 번에 실행합니다.

        Args:
            request: 훈련 데이터, 모델, 에폭 수 등을 포함하는 요청

        Returns:
            CreateJobResponse: 생성된 작업 정보

        Raises:
            ValueError: 데이터 검증 실패 시
            RuntimeError: 작업 생성 실패 시
        """
