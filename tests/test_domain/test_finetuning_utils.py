"""Tests for domain/finetuning/utils.py — DatasetBuilder, DataValidator, FineTuningManager."""

import json
from unittest.mock import MagicMock, patch

import pytest

from beanllm.domain.finetuning.types import TrainingExample
from beanllm.domain.finetuning.utils import (
    DatasetBuilder,
    DataValidator,
    FineTuningCostEstimator,
    FineTuningManager,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_example() -> TrainingExample:
    return TrainingExample(
        messages=[
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is artificial intelligence."},
        ]
    )


# ---------------------------------------------------------------------------
# DatasetBuilder
# ---------------------------------------------------------------------------


class TestDatasetBuilderFromConversations:
    def test_single_conversation(self):
        convs = [[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]]
        examples = DatasetBuilder.from_conversations(convs)
        assert len(examples) == 1
        assert examples[0].messages == convs[0]

    def test_multiple_conversations(self):
        convs = [
            [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
            [{"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}],
        ]
        examples = DatasetBuilder.from_conversations(convs)
        assert len(examples) == 2

    def test_empty_conversations(self):
        examples = DatasetBuilder.from_conversations([])
        assert examples == []


class TestDatasetBuilderFromQAPairs:
    def test_basic_pair(self):
        pairs = [{"question": "What is AI?", "answer": "AI is..."}]
        examples = DatasetBuilder.from_qa_pairs(pairs)
        msgs = examples[0].messages
        assert msgs[-2]["role"] == "user"
        assert msgs[-1]["role"] == "assistant"

    def test_with_system_message(self):
        pairs = [{"question": "Q?", "answer": "A."}]
        examples = DatasetBuilder.from_qa_pairs(pairs, system_message="You are helpful.")
        msgs = examples[0].messages
        assert msgs[0]["role"] == "system"
        assert len(msgs) == 3

    def test_without_system_message(self):
        pairs = [{"question": "Q?", "answer": "A."}]
        examples = DatasetBuilder.from_qa_pairs(pairs)
        assert len(examples[0].messages) == 2


class TestDatasetBuilderFromInstructions:
    def test_basic(self):
        insts = [{"instruction": "Do X", "output": "Done X"}]
        examples = DatasetBuilder.from_instructions(insts)
        msgs = examples[0].messages
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_custom_system_template(self):
        insts = [{"instruction": "I", "output": "O"}]
        examples = DatasetBuilder.from_instructions(insts, system_template="Custom system")
        assert examples[0].messages[0]["content"] == "Custom system"


class TestDatasetBuilderSplitDataset:
    def _make_examples(self, n):
        return [
            TrainingExample(
                messages=[
                    {"role": "user", "content": f"q{i}"},
                    {"role": "assistant", "content": f"a{i}"},
                ]
            )
            for i in range(n)
        ]

    def test_split_ratio(self):
        examples = self._make_examples(10)
        train, val = DatasetBuilder.split_dataset(examples, train_ratio=0.8, shuffle=False)
        assert len(train) == 8
        assert len(val) == 2

    def test_no_shuffle_order_preserved(self):
        examples = self._make_examples(4)
        train, val = DatasetBuilder.split_dataset(examples, train_ratio=0.5, shuffle=False)
        assert train[0].messages[0]["content"] == "q0"

    def test_empty_input(self):
        train, val = DatasetBuilder.split_dataset([], train_ratio=0.8)
        assert train == []
        assert val == []


class TestDatasetBuilderFromFiles:
    def test_from_json_file(self, tmp_path):
        data = [
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            }
        ]
        f = tmp_path / "data.json"
        f.write_text(json.dumps(data))
        examples = DatasetBuilder.from_json_file(str(f))
        assert len(examples) == 1

    def test_from_json_file_not_list_raises(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text(json.dumps({"key": "val"}))
        with pytest.raises(ValueError, match="list"):
            DatasetBuilder.from_json_file(str(f))

    def test_from_jsonl_file(self, tmp_path):
        data = [
            {
                "messages": [
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "a1"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "q2"},
                    {"role": "assistant", "content": "a2"},
                ]
            },
        ]
        f = tmp_path / "data.jsonl"
        f.write_text("\n".join(json.dumps(d) for d in data))
        examples = DatasetBuilder.from_jsonl_file(str(f))
        assert len(examples) == 2


# ---------------------------------------------------------------------------
# DataValidator
# ---------------------------------------------------------------------------


class TestDataValidatorValidateExample:
    def test_valid_example_no_errors(self):
        errors = DataValidator.validate_example(_valid_example())
        assert errors == []

    def test_empty_messages_error(self):
        ex = TrainingExample(messages=[])
        errors = DataValidator.validate_example(ex)
        assert any("at least one" in e for e in errors)

    def test_missing_role_key_error(self):
        ex = TrainingExample(messages=[{"content": "hi"}, {"role": "assistant", "content": "ok"}])
        errors = DataValidator.validate_example(ex)
        assert any("role" in e for e in errors)

    def test_invalid_role_value_error(self):
        ex = TrainingExample(
            messages=[
                {"role": "invalid_role", "content": "hi"},
                {"role": "assistant", "content": "ok"},
            ]
        )
        errors = DataValidator.validate_example(ex)
        assert any("invalid role" in e for e in errors)

    def test_missing_content_key_error(self):
        ex = TrainingExample(messages=[{"role": "user"}, {"role": "assistant", "content": "ok"}])
        errors = DataValidator.validate_example(ex)
        assert any("content" in e for e in errors)

    def test_content_not_string_error(self):
        ex = TrainingExample(
            messages=[
                {"role": "user", "content": 123},
                {"role": "assistant", "content": "ok"},
            ]
        )
        errors = DataValidator.validate_example(ex)
        assert any("string" in e for e in errors)

    def test_no_assistant_message_error(self):
        ex = TrainingExample(messages=[{"role": "user", "content": "hi"}])
        errors = DataValidator.validate_example(ex)
        assert any("assistant" in e for e in errors)

    def test_first_message_assistant_triggers_error(self):
        ex = TrainingExample(
            messages=[
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "ok"},
            ]
        )
        errors = DataValidator.validate_example(ex)
        assert any("system" in e or "user" in e for e in errors)

    def test_system_first_message_valid(self):
        ex = TrainingExample(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        )
        errors = DataValidator.validate_example(ex)
        assert errors == []

    def test_missing_role_key_in_first_message_no_crash(self):
        ex = TrainingExample(
            messages=[
                {"content": "no role key here"},
                {"role": "assistant", "content": "ok"},
            ]
        )
        errors = DataValidator.validate_example(ex)
        assert len(errors) > 0  # reports error, doesn't crash


class TestDataValidatorValidateDataset:
    def test_valid_dataset(self):
        report = DataValidator.validate_dataset([_valid_example()])
        assert report["is_valid"] is True
        assert report["invalid_count"] == 0

    def test_invalid_dataset(self):
        bad = TrainingExample(messages=[])
        report = DataValidator.validate_dataset([bad])
        assert report["is_valid"] is False
        assert report["invalid_count"] == 1

    def test_mixed_dataset_counts(self):
        bad = TrainingExample(messages=[])
        report = DataValidator.validate_dataset([_valid_example(), bad])
        assert report["total_examples"] == 2
        assert report["invalid_count"] == 1


class TestDataValidatorEstimateTokens:
    def test_basic_estimate(self):
        result = DataValidator.estimate_tokens([_valid_example()])
        assert result["total_tokens"] > 0
        assert result["average_per_example"] > 0

    def test_empty_examples(self):
        result = DataValidator.estimate_tokens([])
        assert result["total_tokens"] == 0
        assert result["average_per_example"] == 0


# ---------------------------------------------------------------------------
# FineTuningManager
# ---------------------------------------------------------------------------


class TestFineTuningManagerCreate:
    def test_create_openai_returns_manager(self):
        # Patch at definition site since create() uses local import
        mock_prov = MagicMock()
        with patch(
            "beanllm.domain.finetuning.providers.OpenAIFineTuningProvider", return_value=mock_prov
        ):
            manager = FineTuningManager.create(provider="openai", api_key="test-key")
        assert isinstance(manager, FineTuningManager)

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            FineTuningManager.create(provider="nonexistent")

    def test_create_axolotl_raises_on_missing_args(self):
        # local_providers.py exists so ImportError won't fire;
        # AxolotlProvider instantiation will fail with TypeError (missing args)
        with pytest.raises((ImportError, TypeError, Exception)):
            FineTuningManager.create(provider="axolotl")

    def test_create_unsloth_raises_on_missing_args(self):
        with pytest.raises((ImportError, TypeError, Exception)):
            FineTuningManager.create(provider="unsloth")


class TestFineTuningManagerWorkflow:
    def setup_method(self):
        self.mock_provider = MagicMock()
        self.manager = FineTuningManager(provider=self.mock_provider)

    def test_start_training_calls_create_job(self):
        self.mock_provider.create_job.return_value = MagicMock()
        self.manager.start_training("gpt-4", "file-123")
        self.mock_provider.create_job.assert_called_once()

    def test_get_training_progress_with_metrics(self):
        self.mock_provider.get_job.return_value = MagicMock()
        self.mock_provider.get_metrics.return_value = [{"loss": 0.5}]
        result = self.manager.get_training_progress("job-123")
        assert result["latest_metric"] == {"loss": 0.5}

    def test_get_training_progress_no_metrics(self):
        self.mock_provider.get_job.return_value = MagicMock()
        self.mock_provider.get_metrics.return_value = []
        result = self.manager.get_training_progress("job-123")
        assert result["latest_metric"] is None

    def test_prepare_and_upload_validates_invalid_data(self, tmp_path):
        bad = TrainingExample(messages=[])
        with pytest.raises(ValueError, match="validation failed"):
            self.manager.prepare_and_upload([bad], str(tmp_path / "out.jsonl"), validate=True)

    def test_prepare_and_upload_skip_validation(self, tmp_path):
        self.mock_provider.prepare_data.return_value = None
        self.manager.prepare_and_upload(
            [_valid_example()], str(tmp_path / "out.jsonl"), validate=False
        )
        self.mock_provider.prepare_data.assert_called_once()

    def test_wait_for_completion_completes_immediately(self):
        job = MagicMock()
        job.is_complete.return_value = True
        self.mock_provider.get_job.return_value = job
        result = self.manager.wait_for_completion("job-123", poll_interval=0)
        assert result is job

    def test_wait_for_completion_calls_callback(self):
        job = MagicMock()
        job.is_complete.return_value = True
        self.mock_provider.get_job.return_value = job
        callback = MagicMock()
        self.manager.wait_for_completion("job-123", poll_interval=0, callback=callback)
        callback.assert_called_once_with(job)

    def test_wait_for_completion_timeout(self):
        import time as time_module

        job = MagicMock()
        job.is_complete.return_value = False
        self.mock_provider.get_job.return_value = job
        # Use a very short timeout so it triggers immediately
        with patch("beanllm.domain.finetuning.utils.time") as mock_time:
            mock_time.time.side_effect = [0, 0, 999]  # start=0, check=999
            mock_time.sleep = MagicMock()
            with pytest.raises(TimeoutError):
                self.manager.wait_for_completion("job", poll_interval=0, timeout=100)


# ---------------------------------------------------------------------------
# FineTuningCostEstimator
# ---------------------------------------------------------------------------


class TestFineTuningCostEstimator:
    def test_known_model_gpt4o_mini(self):
        result = FineTuningCostEstimator.estimate_training_cost(
            model="gpt-4o-mini", n_tokens=1_000_000, n_epochs=1, provider="openai"
        )
        assert "estimated_cost_usd" in result
        assert result["estimated_cost_usd"] == pytest.approx(3.0, rel=1e-3)

    def test_unknown_model_zero_cost(self):
        result = FineTuningCostEstimator.estimate_training_cost(
            model="nonexistent-model", n_tokens=1_000_000, n_epochs=1, provider="openai"
        )
        assert result["estimated_cost_usd"] == 0.0

    def test_epochs_multiplied(self):
        r1 = FineTuningCostEstimator.estimate_training_cost("gpt-4o-mini", 1_000_000, 1, "openai")
        r3 = FineTuningCostEstimator.estimate_training_cost("gpt-4o-mini", 1_000_000, 3, "openai")
        assert r3["estimated_cost_usd"] == pytest.approx(r1["estimated_cost_usd"] * 3, rel=1e-5)

    def test_unsupported_provider_returns_error(self):
        result = FineTuningCostEstimator.estimate_training_cost(
            model="some-model", n_tokens=1000, n_epochs=1, provider="anthropic"
        )
        assert "error" in result

    def test_total_tokens_calculated(self):
        result = FineTuningCostEstimator.estimate_training_cost(
            "gpt-4o-mini", n_tokens=500_000, n_epochs=2, provider="openai"
        )
        assert result["total_tokens"] == 1_000_000
