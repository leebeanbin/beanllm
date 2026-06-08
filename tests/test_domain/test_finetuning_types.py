"""Tests for domain/finetuning: types, enums, providers."""

import json
from unittest.mock import MagicMock, patch

import pytest

from beanllm.domain.finetuning.enums import FineTuningStatus, ModelProvider
from beanllm.domain.finetuning.providers import (
    BaseFineTuningProvider,
    OpenAIFineTuningProvider,
)
from beanllm.domain.finetuning.types import (
    FineTuningConfig,
    FineTuningJob,
    FineTuningMetrics,
    TrainingExample,
)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestFineTuningStatus:
    def test_values_exist(self):
        assert FineTuningStatus.CREATED
        assert FineTuningStatus.RUNNING
        assert FineTuningStatus.SUCCEEDED
        assert FineTuningStatus.FAILED
        assert FineTuningStatus.CANCELLED

    def test_succeeded_is_not_failed(self):
        assert FineTuningStatus.SUCCEEDED != FineTuningStatus.FAILED


class TestModelProvider:
    def test_openai_exists(self):
        assert ModelProvider.OPENAI

    def test_anthropic_exists(self):
        assert ModelProvider.ANTHROPIC


# ---------------------------------------------------------------------------
# TrainingExample
# ---------------------------------------------------------------------------


class TestTrainingExample:
    def _make_messages(self):
        return [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

    def test_basic_creation(self):
        msgs = self._make_messages()
        ex = TrainingExample(messages=msgs)
        assert ex.messages == msgs
        assert ex.metadata == {}

    def test_to_dict(self):
        msgs = self._make_messages()
        ex = TrainingExample(messages=msgs)
        d = ex.to_dict()
        assert d == {"messages": msgs}

    def test_to_jsonl(self):
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        ex = TrainingExample(messages=msgs)
        jsonl = ex.to_jsonl()
        parsed = json.loads(jsonl)
        assert parsed["messages"] == msgs

    def test_to_jsonl_valid_json(self):
        ex = TrainingExample(messages=[{"role": "user", "content": "테스트"}])
        jsonl = ex.to_jsonl()
        # Should be valid JSON
        parsed = json.loads(jsonl)
        assert "messages" in parsed

    def test_from_dict(self):
        data = {
            "messages": [{"role": "user", "content": "hello"}],
            "metadata": {"source": "test"},
        }
        ex = TrainingExample.from_dict(data)
        assert ex.messages == data["messages"]
        assert ex.metadata == {"source": "test"}

    def test_from_dict_no_metadata(self):
        data = {"messages": [{"role": "user", "content": "x"}]}
        ex = TrainingExample.from_dict(data)
        assert ex.metadata == {}

    def test_frozen(self):
        ex = TrainingExample(messages=[])
        with pytest.raises((AttributeError, TypeError)):
            ex.messages = []  # type: ignore


# ---------------------------------------------------------------------------
# FineTuningConfig
# ---------------------------------------------------------------------------


class TestFineTuningConfig:
    def test_basic_creation(self):
        config = FineTuningConfig(model="gpt-4o-mini", training_file="file-123")
        assert config.model == "gpt-4o-mini"
        assert config.training_file == "file-123"
        assert config.n_epochs == 3
        assert config.validation_file is None

    def test_custom_values(self):
        config = FineTuningConfig(
            model="gpt-3.5-turbo",
            training_file="file-abc",
            validation_file="file-val",
            n_epochs=5,
            batch_size=4,
            learning_rate_multiplier=0.1,
            suffix="my-model",
        )
        assert config.n_epochs == 5
        assert config.batch_size == 4
        assert config.suffix == "my-model"
        assert config.validation_file == "file-val"

    def test_metadata_default_empty(self):
        config = FineTuningConfig(model="m", training_file="f")
        assert config.metadata == {}


# ---------------------------------------------------------------------------
# FineTuningJob
# ---------------------------------------------------------------------------


class TestFineTuningJob:
    def _make_job(self, status=FineTuningStatus.RUNNING):
        return FineTuningJob(
            job_id="job-123",
            model="gpt-4o-mini",
            status=status,
            created_at=1700000000,
        )

    def test_is_complete_running(self):
        job = self._make_job(FineTuningStatus.RUNNING)
        assert job.is_complete() is False

    def test_is_complete_succeeded(self):
        job = self._make_job(FineTuningStatus.SUCCEEDED)
        assert job.is_complete() is True

    def test_is_complete_failed(self):
        job = self._make_job(FineTuningStatus.FAILED)
        assert job.is_complete() is True

    def test_is_complete_cancelled(self):
        job = self._make_job(FineTuningStatus.CANCELLED)
        assert job.is_complete() is True

    def test_is_success_succeeded(self):
        job = self._make_job(FineTuningStatus.SUCCEEDED)
        assert job.is_success() is True

    def test_is_success_failed(self):
        job = self._make_job(FineTuningStatus.FAILED)
        assert job.is_success() is False

    def test_fields_optional(self):
        job = self._make_job()
        assert job.fine_tuned_model is None
        assert job.error is None
        assert job.result_files == []

    def test_with_fine_tuned_model(self):
        job = FineTuningJob(
            job_id="j",
            model="gpt-4o-mini",
            status=FineTuningStatus.SUCCEEDED,
            created_at=1000,
            fine_tuned_model="ft:gpt-4o-mini:my-org:suffix:id",
        )
        assert "ft:" in job.fine_tuned_model


# ---------------------------------------------------------------------------
# FineTuningMetrics
# ---------------------------------------------------------------------------


class TestFineTuningMetrics:
    def test_basic(self):
        m = FineTuningMetrics(step=10, train_loss=0.5, valid_loss=0.6)
        assert m.step == 10
        assert m.train_loss == pytest.approx(0.5)
        assert m.valid_loss == pytest.approx(0.6)

    def test_defaults(self):
        m = FineTuningMetrics(step=1)
        assert m.train_loss is None
        assert m.valid_accuracy is None
        assert m.learning_rate is None

    def test_accuracy_fields(self):
        m = FineTuningMetrics(step=5, train_accuracy=0.8, valid_accuracy=0.75)
        assert m.train_accuracy == pytest.approx(0.8)
        assert m.valid_accuracy == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# OpenAIFineTuningProvider
# ---------------------------------------------------------------------------


class TestOpenAIFineTuningProviderInit:
    def test_init_with_api_key(self):
        provider = OpenAIFineTuningProvider(api_key="sk-test")
        assert provider.api_key == "sk-test"

    def test_init_from_env(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-env"}):
            provider = OpenAIFineTuningProvider()
        assert provider.api_key == "sk-env"

    def test_init_no_key_raises(self):
        import os

        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(ValueError, match="API key"):
                OpenAIFineTuningProvider()


class TestOpenAIFineTuningProviderPrepareData:
    def setup_method(self):
        self.provider = OpenAIFineTuningProvider(api_key="sk-test")

    def test_prepare_data_writes_jsonl(self, tmp_path):
        examples = [
            TrainingExample(
                messages=[
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ]
            ),
            TrainingExample(
                messages=[
                    {"role": "user", "content": "bye"},
                    {"role": "assistant", "content": "bye"},
                ]
            ),
        ]
        output_path = str(tmp_path / "train.jsonl")
        result_path = self.provider.prepare_data(examples, output_path)
        assert result_path == output_path
        with open(output_path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        parsed = json.loads(lines[0])
        assert "messages" in parsed


class TestOpenAIFineTuningProviderGetClient:
    def test_get_client_without_openai_raises(self):
        provider = OpenAIFineTuningProvider(api_key="sk-test")
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match="openai"):
                provider._get_client()

    def test_get_client_with_mock_openai(self):
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        provider = OpenAIFineTuningProvider(api_key="sk-test")
        with patch.dict("sys.modules", {"openai": mock_openai}):
            client = provider._get_client()

        mock_openai.OpenAI.assert_called_once_with(api_key="sk-test")
        assert client is mock_client


class TestOpenAIFineTuningProviderJobs:
    def setup_method(self):
        self.provider = OpenAIFineTuningProvider(api_key="sk-test")
        self.mock_client = MagicMock()
        self.provider._get_client = MagicMock(return_value=self.mock_client)

    def _make_mock_job_response(self, job_id="job-1", status="running"):
        resp = MagicMock()
        resp.id = job_id
        resp.model = "gpt-4o-mini"
        resp.status = status
        resp.created_at = 1700000000
        resp.finished_at = None
        resp.fine_tuned_model = None
        resp.training_file = "file-abc"
        resp.validation_file = None
        resp.hyperparameters = MagicMock(n_epochs=3, batch_size=None, learning_rate_multiplier=None)
        resp.result_files = []
        resp.error = None
        return resp

    def test_create_job(self):
        mock_resp = self._make_mock_job_response(status="validating_files")
        self.mock_client.fine_tuning.jobs.create.return_value = mock_resp

        config = FineTuningConfig(
            model="gpt-4o-mini",
            training_file="file-abc",
            n_epochs=3,
        )
        job = self.provider.create_job(config)
        assert job.job_id == "job-1"
        assert job.model == "gpt-4o-mini"

    def test_get_job(self):
        mock_resp = self._make_mock_job_response(job_id="job-abc", status="succeeded")
        self.mock_client.fine_tuning.jobs.retrieve.return_value = mock_resp

        job = self.provider.get_job("job-abc")
        assert job.job_id == "job-abc"
        assert job.status == FineTuningStatus.SUCCEEDED

    def test_list_jobs(self):
        mock_resp1 = self._make_mock_job_response("job-1", "succeeded")
        mock_resp2 = self._make_mock_job_response("job-2", "running")
        mock_list = MagicMock()
        mock_list.data = [mock_resp1, mock_resp2]
        self.mock_client.fine_tuning.jobs.list.return_value = mock_list

        jobs = self.provider.list_jobs(limit=10)
        assert len(jobs) == 2

    def test_cancel_job(self):
        mock_resp = self._make_mock_job_response("job-1", "cancelled")
        self.mock_client.fine_tuning.jobs.cancel.return_value = mock_resp

        job = self.provider.cancel_job("job-1")
        assert job.status == FineTuningStatus.CANCELLED
