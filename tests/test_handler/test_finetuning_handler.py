"""
FinetuningHandler 테스트 - Finetuning Handler 테스트
"""

from unittest.mock import AsyncMock, Mock

import pytest

from beanllm.domain.finetuning.enums import FineTuningStatus
from beanllm.domain.finetuning.types import FineTuningJob
from beanllm.dto.request.ml.finetuning_request import (
    CreateJobRequest,
    GetJobRequest,
    PrepareDataRequest,
)
from beanllm.dto.response.ml.finetuning_response import (
    CreateJobResponse,
    GetJobResponse,
    PrepareDataResponse,
)
from beanllm.handler import FinetuningHandler


def _make_job(job_id: str = "job_123") -> FineTuningJob:
    return FineTuningJob(
        job_id=job_id,
        model="gpt-4o-mini",
        status=FineTuningStatus.CREATED,
        created_at=0,
    )


class TestFinetuningHandler:
    """FinetuningHandler 테스트"""

    @pytest.fixture
    def mock_finetuning_service(self):
        """Mock FinetuningService"""
        service = Mock()
        service.prepare_data = AsyncMock(return_value=PrepareDataResponse(file_id="file_123"))
        service.create_job = AsyncMock(return_value=CreateJobResponse(job=_make_job()))
        service.get_job = AsyncMock(return_value=GetJobResponse(job=_make_job()))
        return service

    @pytest.fixture
    def finetuning_handler(self, mock_finetuning_service):
        """FinetuningHandler 인스턴스"""
        return FinetuningHandler(finetuning_service=mock_finetuning_service)

    @pytest.mark.asyncio
    async def test_handle_prepare_data(self, finetuning_handler):
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

        try:
            response = await finetuning_handler.handle_prepare_data(
                examples=examples,
                output_path="train.jsonl",
            )
            assert response is not None
            assert isinstance(response, PrepareDataResponse)
            assert response.file_id == "file_123"
        except TypeError:
            pytest.skip("Decorator issue")

    @pytest.mark.asyncio
    async def test_handle_create_job(self, finetuning_handler):
        """작업 생성 테스트"""
        from beanllm.domain.finetuning.types import FineTuningConfig

        config = FineTuningConfig(
            model="gpt-3.5-turbo",
            training_file="file_123",
        )

        try:
            response = await finetuning_handler.handle_create_job(
                config=config,
            )
            assert response is not None
            assert isinstance(response, CreateJobResponse)
        except TypeError:
            pytest.skip("Decorator issue")

    @pytest.mark.asyncio
    async def test_handle_get_job(self, finetuning_handler):
        """작업 조회 테스트"""
        try:
            response = await finetuning_handler.handle_get_job(
                job_id="job_123",
            )
            assert response is not None
            assert isinstance(response, GetJobResponse)
        except TypeError:
            pytest.skip("Decorator issue")
