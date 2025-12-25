"""
FinetuningService 테스트 - Finetuning 서비스 구현체 테스트
"""

import pytest
from unittest.mock import Mock

from beanllm.dto.request.finetuning_request import (
    PrepareDataRequest,
    CreateJobRequest,
    GetJobRequest,
    ListJobsRequest,
    CancelJobRequest,
    GetMetricsRequest,
    StartTrainingRequest,
    WaitForCompletionRequest,
    QuickFinetuneRequest,
)
from beanllm.dto.response.finetuning_response import (
    PrepareDataResponse,
    CreateJobResponse,
    GetJobResponse,
    ListJobsResponse,
    CancelJobResponse,
    GetMetricsResponse,
    StartTrainingResponse,
)
from beanllm.domain.finetuning.types import FineTuningJob, FineTuningConfig
from beanllm.domain.finetuning.enums import FineTuningStatus
from beanllm.service.impl.finetuning_service_impl import FinetuningServiceImpl


class TestFinetuningService:
    """FinetuningService 테스트"""

    @pytest.fixture
    def mock_provider(self):
        """Mock FineTuningProvider"""
        provider = Mock()

        # Mock job - FineTuningJob은 dataclass이므로 실제 인스턴스 생성
        from beanllm.domain.finetuning.types import FineTuningJob, FineTuningStatus
        import time

        mock_job = FineTuningJob(
            job_id="job_123",
            status=FineTuningStatus.CREATED,
            model="gpt-3.5-turbo",
            created_at=int(time.time()),
            fine_tuned_model=None,
        )

        provider.create_job = Mock(return_value=mock_job)
        provider.get_job = Mock(return_value=mock_job)
        provider.list_jobs = Mock(return_value=[mock_job])
        provider.cancel_job = Mock(return_value=mock_job)
        provider.get_metrics = Mock(return_value=[])

        return provider

    @pytest.fixture
    def mock_manager(self):
        """Mock FineTuningManager"""
        from beanllm.domain.finetuning.types import FineTuningJob, FineTuningStatus
        import time

        manager = Mock()
        manager.prepare_and_upload = Mock(return_value="file_123")

        # FineTuningJob 인스턴스 생성
        training_job = FineTuningJob(
            job_id="job_123",
            status=FineTuningStatus.CREATED,
            model="gpt-3.5-turbo",
            created_at=int(time.time()),
        )
        completed_job = FineTuningJob(
            job_id="job_123",
            status=FineTuningStatus.SUCCEEDED,
            model="gpt-3.5-turbo",
            created_at=int(time.time()),
        )

        manager.start_training = Mock(return_value=training_job)
        manager.wait_for_completion = Mock(return_value=completed_job)
        return manager

    @pytest.fixture
    def finetuning_service(self, mock_provider, mock_manager):
        """FinetuningService 인스턴스"""
        service = FinetuningServiceImpl(provider=mock_provider)
        service._manager = mock_manager
        return service

    @pytest.mark.asyncio
    async def test_prepare_data(self, finetuning_service):
        """데이터 준비 테스트"""
        from beanllm.domain.finetuning.types import TrainingExample

        examples = [
            TrainingExample(
                messages=[
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ]
            )
        ]
        request = PrepareDataRequest(
            examples=examples,
            output_path="train.jsonl",
            validate=True,
        )

        response = await finetuning_service.prepare_data(request)

        assert response is not None
        assert isinstance(response, PrepareDataResponse)
        assert response.file_id == "file_123"

    @pytest.mark.asyncio
    async def test_create_job(self, finetuning_service):
        """작업 생성 테스트"""
        config = FineTuningConfig(
            model="gpt-3.5-turbo",
            training_file="file_123",
        )
        request = CreateJobRequest(config=config)

        response = await finetuning_service.create_job(request)

        assert response is not None
        assert isinstance(response, CreateJobResponse)
        assert response.job.job_id == "job_123"

    @pytest.mark.asyncio
    async def test_get_job(self, finetuning_service):
        """작업 상태 조회 테스트"""
        request = GetJobRequest(job_id="job_123")

        response = await finetuning_service.get_job(request)

        assert response is not None
        assert isinstance(response, GetJobResponse)
        assert response.job.job_id == "job_123"

    @pytest.mark.asyncio
    async def test_list_jobs(self, finetuning_service):
        """작업 목록 조회 테스트"""
        request = ListJobsRequest(limit=10)

        response = await finetuning_service.list_jobs(request)

        assert response is not None
        assert isinstance(response, ListJobsResponse)
        assert len(response.jobs) == 1

    @pytest.mark.asyncio
    async def test_cancel_job(self, finetuning_service):
        """작업 취소 테스트"""
        request = CancelJobRequest(job_id="job_123")

        response = await finetuning_service.cancel_job(request)

        assert response is not None
        assert isinstance(response, CancelJobResponse)
        assert response.job.job_id == "job_123"

    @pytest.mark.asyncio
    async def test_get_metrics(self, finetuning_service):
        """훈련 메트릭 조회 테스트"""
        request = GetMetricsRequest(job_id="job_123")

        response = await finetuning_service.get_metrics(request)

        assert response is not None
        assert isinstance(response, GetMetricsResponse)
        assert response.metrics == []

    @pytest.mark.asyncio
    async def test_start_training(self, finetuning_service):
        """훈련 시작 테스트"""
        request = StartTrainingRequest(
            model="gpt-3.5-turbo",
            training_file="file_123",
            validation_file="file_456",
        )

        response = await finetuning_service.start_training(request)

        assert response is not None
        assert isinstance(response, StartTrainingResponse)
        assert response.job.job_id == "job_123"

    @pytest.mark.asyncio
    async def test_wait_for_completion(self, finetuning_service):
        """작업 완료 대기 테스트"""
        request = WaitForCompletionRequest(
            job_id="job_123",
            poll_interval=10,
            timeout=300,
        )

        response = await finetuning_service.wait_for_completion(request)

        assert response is not None
        assert isinstance(response, GetJobResponse)
        assert response.job.job_id == "job_123"

    @pytest.mark.asyncio
    async def test_quick_finetune(self, finetuning_service):
        """빠른 파인튜닝 테스트"""
        from beanllm.domain.finetuning.types import TrainingExample

        training_data = [
            TrainingExample(
                messages=[
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ]
            ),
            TrainingExample(
                messages=[
                    {"role": "user", "content": "How are you?"},
                    {"role": "assistant", "content": "I'm fine"},
                ]
            ),
        ]
        request = QuickFinetuneRequest(
            training_data=training_data,
            model="gpt-3.5-turbo",
            validation_split=0.1,
            n_epochs=3,
            wait=False,  # 대기하지 않음
        )

        response = await finetuning_service.quick_finetune(request)

        assert response is not None
        assert isinstance(response, CreateJobResponse)
        assert response.job.job_id == "job_123"

    @pytest.mark.asyncio
    async def test_quick_finetune_with_wait(self, finetuning_service):
        """대기 포함 빠른 파인튜닝 테스트"""
        from beanllm.domain.finetuning.types import TrainingExample

        training_data = [
            TrainingExample(
                messages=[
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ]
            ),
        ]
        request = QuickFinetuneRequest(
            training_data=training_data,
            model="gpt-3.5-turbo",
            validation_split=0.0,  # 검증 데이터 없음
            n_epochs=1,
            wait=True,
        )

        response = await finetuning_service.quick_finetune(request)

        assert response is not None
        assert isinstance(response, CreateJobResponse)
        assert response.job.job_id == "job_123"


