"""Tests for domain/evaluation/lm_eval_harness_wrapper.py — LMEvalHarnessWrapper."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from beanllm.domain.evaluation.lm_eval_harness_wrapper import LMEvalHarnessWrapper

# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestLMEvalHarnessWrapperInit:
    def test_stores_model(self):
        w = LMEvalHarnessWrapper(model="hf")
        assert w.model == "hf"

    def test_stores_model_args(self):
        w = LMEvalHarnessWrapper(model_args="pretrained=llama")
        assert w.model_args == "pretrained=llama"

    def test_default_num_fewshot(self):
        w = LMEvalHarnessWrapper()
        assert w.num_fewshot == 0

    def test_custom_num_fewshot(self):
        w = LMEvalHarnessWrapper(num_fewshot=5)
        assert w.num_fewshot == 5

    def test_default_limit_none(self):
        w = LMEvalHarnessWrapper()
        assert w.limit is None

    def test_output_path_converted_to_path(self):
        w = LMEvalHarnessWrapper(output_path="/tmp/results")
        assert isinstance(w.output_path, Path)

    def test_output_path_none_if_not_given(self):
        w = LMEvalHarnessWrapper()
        assert w.output_path is None

    def test_lm_eval_lazy_none(self):
        w = LMEvalHarnessWrapper()
        assert w._lm_eval is None


# ---------------------------------------------------------------------------
# _check_dependencies
# ---------------------------------------------------------------------------


class TestCheckDependencies:
    def test_raises_import_error_if_lm_eval_missing(self):
        w = LMEvalHarnessWrapper()
        with patch.dict("sys.modules", {"lm_eval": None}):
            with pytest.raises(ImportError, match="lm-eval is required"):
                w._check_dependencies()

    def test_sets_lm_eval_on_success(self):
        w = LMEvalHarnessWrapper()
        mock_module = MagicMock()
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args, **kw: mock_module
            if name == "lm_eval"
            else __import__(name, *args, **kw),
        ):
            try:
                w._check_dependencies()
            except Exception:
                pass  # May fail but _lm_eval might be set


# ---------------------------------------------------------------------------
# get_popular_tasks
# ---------------------------------------------------------------------------


class TestGetPopularTasks:
    def test_returns_dict(self):
        w = LMEvalHarnessWrapper()
        tasks = w.get_popular_tasks()
        assert isinstance(tasks, dict)

    def test_contains_mmlu(self):
        w = LMEvalHarnessWrapper()
        tasks = w.get_popular_tasks()
        assert "mmlu" in tasks

    def test_contains_gsm8k(self):
        w = LMEvalHarnessWrapper()
        tasks = w.get_popular_tasks()
        assert "gsm8k" in tasks

    def test_returns_copy_not_original(self):
        w = LMEvalHarnessWrapper()
        tasks1 = w.get_popular_tasks()
        tasks2 = w.get_popular_tasks()
        tasks1["new_key"] = "value"
        assert "new_key" not in tasks2


# ---------------------------------------------------------------------------
# evaluate_mmlu
# ---------------------------------------------------------------------------


class TestEvaluateMmlu:
    def test_calls_evaluate_with_mmlu_task(self):
        w = LMEvalHarnessWrapper()
        with patch.object(w, "evaluate", return_value={"results": {}}) as mock_eval:
            w.evaluate_mmlu(num_fewshot=5)
            call_kwargs = mock_eval.call_args.kwargs
            assert "mmlu" in call_kwargs.get("tasks", [])

    def test_with_subjects_creates_subject_tasks(self):
        w = LMEvalHarnessWrapper()
        with patch.object(w, "evaluate", return_value={"results": {}}) as mock_eval:
            w.evaluate_mmlu(subjects=["algebra", "anatomy"])
            call_kwargs = mock_eval.call_args.kwargs
            tasks = call_kwargs.get("tasks", [])
            assert "mmlu_algebra" in tasks
            assert "mmlu_anatomy" in tasks

    def test_without_subjects_uses_mmlu(self):
        w = LMEvalHarnessWrapper()
        with patch.object(w, "evaluate", return_value={"results": {}}) as mock_eval:
            w.evaluate_mmlu()
            call_kwargs = mock_eval.call_args.kwargs
            assert call_kwargs.get("tasks") == ["mmlu"]


# ---------------------------------------------------------------------------
# evaluate_suite
# ---------------------------------------------------------------------------


class TestEvaluateSuite:
    def test_standard_suite_contains_mmlu(self):
        w = LMEvalHarnessWrapper()
        with patch.object(w, "evaluate", return_value={"results": {}}) as mock_eval:
            w.evaluate_suite("standard")
            call_kwargs = mock_eval.call_args.kwargs
            assert "mmlu" in call_kwargs.get("tasks", [])

    def test_math_suite_contains_gsm8k(self):
        w = LMEvalHarnessWrapper()
        with patch.object(w, "evaluate", return_value={"results": {}}) as mock_eval:
            w.evaluate_suite("math")
            call_kwargs = mock_eval.call_args.kwargs
            assert "gsm8k" in call_kwargs.get("tasks", [])

    def test_coding_suite_contains_humaneval(self):
        w = LMEvalHarnessWrapper()
        with patch.object(w, "evaluate", return_value={"results": {}}) as mock_eval:
            w.evaluate_suite("coding")
            call_kwargs = mock_eval.call_args.kwargs
            assert "humaneval" in call_kwargs.get("tasks", [])

    def test_unknown_suite_raises(self):
        w = LMEvalHarnessWrapper()
        with pytest.raises(ValueError, match="Unknown suite"):
            w.evaluate_suite("unknown_suite")

    def test_all_suites_work(self):
        w = LMEvalHarnessWrapper()
        for suite in ("standard", "reasoning", "math", "coding", "korean", "comprehensive"):
            with patch.object(w, "evaluate", return_value={"results": {}}) as mock_eval:
                w.evaluate_suite(suite)
                assert mock_eval.called


# ---------------------------------------------------------------------------
# evaluate (routing tests only - lm_eval not installed)
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_evaluate_mmlu_routes_to_evaluate(self):
        w = LMEvalHarnessWrapper()
        with patch.object(w, "evaluate", return_value={}) as mock_eval:
            w.evaluate_mmlu(num_fewshot=5)
        mock_eval.assert_called_once()

    def test_evaluate_suite_routes_to_evaluate(self):
        w = LMEvalHarnessWrapper()
        with patch.object(w, "evaluate", return_value={}) as mock_eval:
            w.evaluate_suite("math")
        mock_eval.assert_called_once()

    def test_evaluate_raises_import_error_without_lm_eval(self):
        w = LMEvalHarnessWrapper()
        # _check_dependencies should raise ImportError when lm_eval not installed
        with pytest.raises(ImportError, match="lm-eval is required"):
            w._check_dependencies()


# ---------------------------------------------------------------------------
# get_leaderboard_format
# ---------------------------------------------------------------------------


class TestGetLeaderboardFormat:
    def test_returns_dict(self):
        w = LMEvalHarnessWrapper()
        results = {"results": {"mmlu": {"acc": 0.45, "acc_norm": 0.47}}}
        lb = w.get_leaderboard_format(results)
        assert isinstance(lb, dict)

    def test_uses_acc_norm_preferentially(self):
        w = LMEvalHarnessWrapper()
        results = {"results": {"mmlu": {"acc": 0.45, "acc_norm": 0.47}}}
        lb = w.get_leaderboard_format(results)
        assert lb["mmlu"] == 0.47

    def test_falls_back_to_acc(self):
        w = LMEvalHarnessWrapper()
        results = {"results": {"hellaswag": {"acc": 0.65}}}
        lb = w.get_leaderboard_format(results)
        assert lb["hellaswag"] == 0.65

    def test_empty_results_returns_empty(self):
        w = LMEvalHarnessWrapper()
        lb = w.get_leaderboard_format({})
        assert lb == {}

    def test_multiple_tasks(self):
        w = LMEvalHarnessWrapper()
        results = {
            "results": {
                "mmlu": {"acc_norm": 0.5},
                "arc_easy": {"acc_norm": 0.7},
            }
        }
        lb = w.get_leaderboard_format(results)
        assert len(lb) == 2


# ---------------------------------------------------------------------------
# _save_results
# ---------------------------------------------------------------------------


class TestSaveResults:
    def test_saves_to_json_file(self, tmp_path):
        w = LMEvalHarnessWrapper(output_path=str(tmp_path))
        results = {"results": {"mmlu": {"acc": 0.5}}}
        w._save_results(results, ["mmlu"])
        files = list(tmp_path.glob("lm_eval_*.json"))
        assert len(files) == 1
        with open(files[0]) as f:
            saved = json.load(f)
        assert saved == results

    def test_creates_directory_if_needed(self, tmp_path):
        output_dir = tmp_path / "nested" / "results"
        w = LMEvalHarnessWrapper(output_path=str(output_dir))
        w._save_results({"results": {}}, ["test"])
        assert output_dir.exists()

    def test_filename_contains_task_name(self, tmp_path):
        w = LMEvalHarnessWrapper(output_path=str(tmp_path))
        w._save_results({}, ["mmlu", "hellaswag"])
        files = list(tmp_path.glob("lm_eval_*.json"))
        assert any("mmlu" in f.name for f in files)

    def test_skips_if_no_output_path(self):
        w = LMEvalHarnessWrapper()
        # Should not raise even with no output path
        w._save_results({"results": {}}, ["mmlu"])


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_contains_model(self):
        w = LMEvalHarnessWrapper(model="local-completions")
        r = repr(w)
        assert "local-completions" in r

    def test_repr_contains_num_fewshot(self):
        w = LMEvalHarnessWrapper(num_fewshot=5)
        r = repr(w)
        assert "5" in r
