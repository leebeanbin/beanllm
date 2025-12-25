"""
FineTuning Facade 테스트
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

try:
    from llmkit.facade.finetuning_facade import FineTuningManagerFacade
    from llmkit.domain.finetuning.providers import OpenAIFineTuningProvider
    from llmkit.domain.finetuning.types import FineTuningJob, TrainingExample
    from llmkit.dto.response.finetuning_response import (
        PrepareDataResponse,
        StartTrainingResponse,
        GetJobResponse,
        GetMetricsResponse,
    )

    FACADE_AVAILABLE = True
except ImportError:
    FACADE_AVAILABLE = False


@pytest.mark.skipif(not FACADE_AVAILABLE, reason="FineTuningManagerFacade not available")
class TestFineTuningManagerFacade:
    @pytest.fixture
    def provider(self):
        return Mock(spec=OpenAIFineTuningProvider)

    @pytest.fixture
    def manager(self, provider):
        # Facade가 직접 Handler를 생성하므로 Handler를 Mock으로 교체
        with patch("llmkit.facade.finetuning_facade.FinetuningHandler") as mock_handler_class:
            mock_handler = MagicMock()

            # prepare_data mock
            from llmkit.domain.finetuning.enums import FineTuningStatus

            mock_job = FineTuningJob(
                job_id="job_123",
                model="gpt-3.5-turbo",
                status=FineTuningStatus.CREATED,
                created_at=1234567890,
            )

            mock_prepare_response = PrepareDataResponse(file_id="file_123")

            async def mock_handle_prepare_data(*args, **kwargs):
                return mock_prepare_response

            mock_handler.handle_prepare_data = MagicMock(side_effect=mock_handle_prepare_data)

            # start_training mock
            mock_start_response = StartTrainingResponse(job=mock_job)

            async def mock_handle_start_training(*args, **kwargs):
                return mock_start_response

            mock_handler.handle_start_training = MagicMock(side_effect=mock_handle_start_training)

            # wait_for_completion mock
            mock_wait_response = GetJobResponse(job=mock_job)

            async def mock_handle_wait_for_completion(*args, **kwargs):
                return mock_wait_response

            mock_handler.handle_wait_for_completion = MagicMock(
                side_effect=mock_handle_wait_for_completion
            )

            # get_job mock
            mock_get_response = GetJobResponse(job=mock_job)

            async def mock_handle_get_job(*args, **kwargs):
                return mock_get_response

            mock_handler.handle_get_job = MagicMock(side_effect=mock_handle_get_job)

            # get_metrics mock - metrics를 리스트로 설정
            from llmkit.domain.finetuning.types import FineTuningMetrics
            mock_metrics = [FineTuningMetrics(step=1, train_loss=0.5, valid_loss=0.6)]
            mock_metrics_response = GetMetricsResponse(metrics=mock_metrics)

            async def mock_handle_get_metrics(*args, **kwargs):
                return mock_metrics_response

            mock_handler.handle_get_metrics = MagicMock(side_effect=mock_handle_get_metrics)

            # Handler 클래스가 인스턴스화될 때 mock_handler 반환
            mock_handler_class.return_value = mock_handler

            manager = FineTuningManagerFacade(provider=provider)
            # 실제 생성된 Handler를 Mock으로 교체
            manager._finetuning_handler = mock_handler
            return manager

    def test_prepare_and_upload(self, manager):
        examples = [
            TrainingExample(
                messages=[
                    {"role": "user", "content": "test"},
                    {"role": "assistant", "content": "response"},
                ]
            )
        ]
        file_id = manager.prepare_and_upload(examples, "output.jsonl")
        assert file_id == "file_123"
        assert manager._finetuning_handler.handle_prepare_data.called

    def test_start_training(self, manager):
        job = manager.start_training("gpt-3.5-turbo", "file_123")
        assert isinstance(job, FineTuningJob)
        assert job.job_id == "job_123"
        assert manager._finetuning_handler.handle_start_training.called

    def test_wait_for_completion(self, manager):
        job = manager.wait_for_completion("job_123")
        assert isinstance(job, FineTuningJob)
        assert job.job_id == "job_123"
        assert manager._finetuning_handler.handle_wait_for_completion.called

    def test_get_training_progress(self, manager):
        progress = manager.get_training_progress("job_123")
        assert isinstance(progress, dict)
        assert "job" in progress
        assert "metrics" in progress
        assert manager._finetuning_handler.handle_get_job.called
        assert manager._finetuning_handler.handle_get_metrics.called

