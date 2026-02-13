"""
Finetuning Service Interfaces

파인튜닝 작업의 전체 수명주기를 관리하는 서비스 인터페이스.
ISP(Interface Segregation Principle)에 따라 역할별로 분리합니다:
- IFinetuningDataService: 데이터 준비
- IFinetuningJobService: 작업 CRUD (생성/조회/목록/취소)
- IFinetuningTrainingService: 훈련 실행 및 모니터링
- IFinetuningService: 위 3개를 통합 + quick_finetune 편의 메서드
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


class IFinetuningDataService(ABC):
    """
    파인튜닝 데이터 준비 인터페이스

    훈련 데이터의 준비 및 업로드를 담당합니다.

    Example:
        >>> data_service: IFinetuningDataService = ...
        >>> resp = await data_service.prepare_data(PrepareDataRequest(...))
    """

    @abstractmethod
    async def prepare_data(self, request: "PrepareDataRequest") -> "PrepareDataResponse":
        """
        훈련 데이터를 준비하고 프로바이더에 업로드합니다.

        Args:
            request: 훈련 예제, 출력 경로, 검증 여부가 포함된 요청

        Returns:
            PrepareDataResponse: 업로드된 파일 ID가 포함된 응답

        Raises:
            ValueError: 훈련 데이터가 유효하지 않은 경우
            RuntimeError: 파일 업로드 실패 시
        """
        pass


class IFinetuningJobService(ABC):
    """
    파인튜닝 작업 관리 인터페이스

    작업의 생성, 조회, 목록, 취소를 담당합니다.

    Example:
        >>> job_service: IFinetuningJobService = ...
        >>> job = await job_service.create_job(CreateJobRequest(...))
        >>> status = await job_service.get_job(GetJobRequest(job_id=job.job_id))
    """

    @abstractmethod
    async def create_job(self, request: "CreateJobRequest") -> "CreateJobResponse":
        """
        파인튜닝 작업을 생성합니다.

        Args:
            request: 모델명, 훈련 파일 ID 등이 포함된 요청

        Returns:
            CreateJobResponse: 생성된 작업 정보

        Raises:
            ValueError: 요청 파라미터가 유효하지 않은 경우
        """
        pass

    @abstractmethod
    async def get_job(self, request: "GetJobRequest") -> "GetJobResponse":
        """
        파인튜닝 작업 상태를 조회합니다.

        Args:
            request: 조회할 작업 ID가 포함된 요청

        Returns:
            GetJobResponse: 작업 상태, 진행률 등의 정보

        Raises:
            ValueError: 작업 ID가 존재하지 않는 경우
        """
        pass

    @abstractmethod
    async def list_jobs(self, request: "ListJobsRequest") -> "ListJobsResponse":
        """
        파인튜닝 작업 목록을 조회합니다.

        Args:
            request: 필터 조건이 포함된 요청

        Returns:
            ListJobsResponse: 작업 목록
        """
        pass

    @abstractmethod
    async def cancel_job(self, request: "CancelJobRequest") -> "CancelJobResponse":
        """
        진행 중인 파인튜닝 작업을 취소합니다.

        Args:
            request: 취소할 작업 ID가 포함된 요청

        Returns:
            CancelJobResponse: 취소 결과

        Raises:
            ValueError: 작업이 이미 완료되었거나 존재하지 않는 경우
        """
        pass


class IFinetuningTrainingService(ABC):
    """
    파인튜닝 훈련 실행 및 모니터링 인터페이스

    훈련 시작, 완료 대기, 메트릭 조회를 담당합니다.

    Example:
        >>> training_service: IFinetuningTrainingService = ...
        >>> resp = await training_service.start_training(StartTrainingRequest(...))
        >>> metrics = await training_service.get_metrics(GetMetricsRequest(job_id=...))
    """

    @abstractmethod
    async def start_training(self, request: "StartTrainingRequest") -> "StartTrainingResponse":
        """
        파인튜닝 훈련을 시작합니다.

        Args:
            request: 모델, 훈련 파일, 하이퍼파라미터가 포함된 요청

        Returns:
            StartTrainingResponse: 훈련 작업 정보

        Raises:
            ValueError: 요청 파라미터가 유효하지 않은 경우
            RuntimeError: 훈련 시작 실패 시
        """
        pass

    @abstractmethod
    async def wait_for_completion(self, request: "WaitForCompletionRequest") -> "GetJobResponse":
        """
        파인튜닝 작업이 완료될 때까지 대기합니다.

        Args:
            request: 작업 ID, 폴링 간격, 타임아웃, 콜백이 포함된 요청

        Returns:
            GetJobResponse: 완료된 작업 정보

        Raises:
            TimeoutError: 타임아웃 초과 시
            RuntimeError: 작업이 실패한 경우
        """
        pass

    @abstractmethod
    async def get_metrics(self, request: "GetMetricsRequest") -> "GetMetricsResponse":
        """
        훈련 메트릭(loss, accuracy 등)을 조회합니다.

        Args:
            request: 작업 ID가 포함된 요청

        Returns:
            GetMetricsResponse: 훈련 메트릭 목록

        Raises:
            ValueError: 작업 ID가 존재하지 않는 경우
        """
        pass


class IFinetuningService(
    IFinetuningDataService,
    IFinetuningJobService,
    IFinetuningTrainingService,
):
    """
    파인튜닝 통합 서비스 인터페이스

    데이터 준비, 작업 관리, 훈련의 전체 워크플로우를 하나로 통합합니다.
    필요에 따라 세분화된 인터페이스(IFinetuningDataService 등)만 사용할 수도 있습니다.

    Example:
        >>> service: IFinetuningService = factory.create_finetuning_service()
        >>> data_resp = await service.prepare_data(PrepareDataRequest(...))
        >>> job_resp = await service.start_training(StartTrainingRequest(...))
        >>> await service.wait_for_completion(WaitForCompletionRequest(job_id=...))
    """

    @abstractmethod
    async def quick_finetune(self, request: "QuickFinetuneRequest") -> "CreateJobResponse":
        """
        빠른 파인튜닝을 시작합니다 (데이터 준비 → 훈련 → 완료 대기를 한번에).

        Args:
            request: 훈련 데이터, 모델, 하이퍼파라미터가 포함된 요청

        Returns:
            CreateJobResponse: 완료된 작업 정보

        Raises:
            ValueError: 요청 파라미터가 유효하지 않은 경우
            RuntimeError: 파인튜닝 프로세스 실패 시
        """
        pass
