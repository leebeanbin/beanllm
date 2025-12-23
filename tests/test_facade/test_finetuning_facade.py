"""
FineTuning Facade 테스트
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

try:
    from llmkit.facade.finetuning_facade import FineTuningManagerFacade
    from llmkit.domain.finetuning.providers import OpenAIFineTuningProvider
    from llmkit.domain.finetuning.types import FineTuningJob, TrainingExample

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
        with patch("llmkit.facade.finetuning_facade.HandlerFactory") as mock_factory:
            mock_handler = MagicMock()

            # prepare_data mock
            mock_prepare_response = Mock()
            mock_prepare_response.file_id = "file_123"

            async def mock_handle_prepare_data(*args, **kwargs):
                return mock_prepare_response

            mock_handler.handle_prepare_data = MagicMock(side_effect=mock_handle_prepare_data)

            # start_training mock
            from llmkit.domain.finetuning.enums import FineTuningStatus

            mock_job = FineTuningJob(
                job_id="job_123",
                model="gpt-3.5-turbo",
                status=FineTuningStatus.CREATED,
                created_at=1234567890,
            )
            mock_start_response = Mock()
            mock_start_response.job = mock_job

            async def mock_handle_start_training(*args, **kwargs):
                return mock_start_response

            mock_handler.handle_start_training = MagicMock(side_effect=mock_handle_start_training)

            # wait_for_completion mock
            mock_wait_response = Mock()
            mock_wait_response.job = mock_job

            async def mock_handle_wait_for_completion(*args, **kwargs):
                return mock_wait_response

            mock_handler.handle_wait_for_completion = MagicMock(
                side_effect=mock_handle_wait_for_completion
            )

            # get_job mock
            mock_get_response = Mock()
            mock_get_response.job = mock_job

            async def mock_handle_get_job(*args, **kwargs):
                return mock_get_response

            mock_handler.handle_get_job = MagicMock(side_effect=mock_handle_get_job)

            # get_metrics mock
            mock_metrics_response = Mock()
            mock_metrics_response.metrics = []

            async def mock_handle_get_metrics(*args, **kwargs):
                return mock_metrics_response

            mock_handler.handle_get_metrics = MagicMock(side_effect=mock_handle_get_metrics)

            mock_handler_factory = Mock()
            mock_handler_factory.create_finetuning_handler.return_value = mock_handler
            mock_factory.return_value = mock_handler_factory

            manager = FineTuningManagerFacade(provider=provider)
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

