"""Tests for domain/prompts/ab_testing.py — ABTestConfig, ABTestResult, ABTestRunner."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from beanllm.domain.prompts.ab_testing import (
    ABTestConfig,
    ABTestResult,
    ABTestRunner,
)

# ---------------------------------------------------------------------------
# ABTestConfig
# ---------------------------------------------------------------------------


class TestABTestConfig:
    def test_basic_creation(self):
        config = ABTestConfig(prompt_a="Hello {name}", prompt_b="Hi {name}")
        assert config.prompt_a == "Hello {name}"
        assert config.prompt_b == "Hi {name}"

    def test_defaults(self):
        config = ABTestConfig(prompt_a="A", prompt_b="B")
        assert config.traffic_split == 0.5
        assert config.min_samples == 100
        assert "accuracy" in config.metrics
        assert config.prompt_a_version == "v1"
        assert config.prompt_b_version == "v2"

    def test_custom_metrics(self):
        config = ABTestConfig(prompt_a="A", prompt_b="B", metrics=["latency"])
        assert config.metrics == ["latency"]


# ---------------------------------------------------------------------------
# ABTestResult.get_summary / _std
# ---------------------------------------------------------------------------


class TestABTestResult:
    def _make_result(self):
        config = ABTestConfig(prompt_a="A", prompt_b="B", metrics=["accuracy", "latency"])
        result = ABTestResult(config=config)
        result.results_a = [
            {"accuracy": 0.8, "latency": 1.0},
            {"accuracy": 0.9, "latency": 1.2},
        ]
        result.results_b = [
            {"accuracy": 0.85, "latency": 0.8},
            {"accuracy": 0.95, "latency": 0.9},
        ]
        return result

    def test_get_summary_returns_all_metrics(self):
        result = self._make_result()
        summary = result.get_summary()
        assert "accuracy" in summary
        assert "latency" in summary

    def test_get_summary_mean_calculation(self):
        result = self._make_result()
        summary = result.get_summary()
        assert summary["accuracy"]["a_mean"] == pytest.approx(0.85)
        assert summary["accuracy"]["b_mean"] == pytest.approx(0.9)

    def test_get_summary_improvement(self):
        result = self._make_result()
        summary = result.get_summary()
        # B is better: 0.9 - 0.85 = 0.05
        assert summary["accuracy"]["improvement"] == pytest.approx(0.05, rel=1e-3)

    def test_get_summary_empty_results(self):
        config = ABTestConfig(prompt_a="A", prompt_b="B", metrics=["accuracy"])
        result = ABTestResult(config=config)
        summary = result.get_summary()
        assert summary["accuracy"]["a_mean"] == 0
        assert summary["accuracy"]["b_mean"] == 0

    def test_std_empty_returns_zero(self):
        config = ABTestConfig(prompt_a="A", prompt_b="B")
        result = ABTestResult(config=config)
        assert result._std([]) == 0.0

    def test_std_single_value_is_zero(self):
        config = ABTestConfig(prompt_a="A", prompt_b="B")
        result = ABTestResult(config=config)
        assert result._std([5.0]) == 0.0

    def test_std_multiple_values(self):
        config = ABTestConfig(prompt_a="A", prompt_b="B")
        result = ABTestResult(config=config)
        std = result._std([2.0, 4.0])
        assert std > 0.0

    def test_end_time_is_none_initially(self):
        config = ABTestConfig(prompt_a="A", prompt_b="B")
        result = ABTestResult(config=config)
        assert result.end_time is None


# ---------------------------------------------------------------------------
# ABTestRunner
# ---------------------------------------------------------------------------


class TestABTestRunnerAnalyzeResults:
    def _make_runner(self):
        mock_version_manager = MagicMock()
        return ABTestRunner(version_manager=mock_version_manager)

    def _make_result_with_data(self):
        config = ABTestConfig(prompt_a="A", prompt_b="B", metrics=["accuracy"])
        result = ABTestResult(config=config)
        result.results_a = [{"accuracy": 0.8}, {"accuracy": 0.9}]
        result.results_b = [{"accuracy": 0.85}, {"accuracy": 0.95}]
        result.end_time = datetime.now()
        return result

    def test_analyze_returns_dict_with_keys(self):
        runner = self._make_runner()
        result = self._make_result_with_data()
        analysis = runner.analyze_results(result)
        assert "summary" in analysis
        assert "statistical_significance" in analysis
        assert "recommendation" in analysis
        assert "sample_sizes" in analysis

    def test_sample_sizes_match_results(self):
        runner = self._make_runner()
        result = self._make_result_with_data()
        analysis = runner.analyze_results(result)
        assert analysis["sample_sizes"]["a"] == 2
        assert analysis["sample_sizes"]["b"] == 2

    def test_recommendation_is_string(self):
        runner = self._make_runner()
        result = self._make_result_with_data()
        analysis = runner.analyze_results(result)
        assert analysis["recommendation"] in ["A", "B"]

    def test_insufficient_samples_no_significance(self):
        config = ABTestConfig(prompt_a="A", prompt_b="B", metrics=["accuracy"])
        result = ABTestResult(config=config)
        result.results_a = [{"accuracy": 0.8}]  # only 1 sample
        result.results_b = [{"accuracy": 0.9}]  # only 1 sample
        result.end_time = datetime.now()
        runner = self._make_runner()
        analysis = runner.analyze_results(result)
        sig = analysis["statistical_significance"]["accuracy"]
        assert sig["significant"] is False


class TestABTestRunnerRunTest:
    async def test_run_test_assigns_to_a_or_b(self):
        mock_vm = MagicMock()
        runner = ABTestRunner(version_manager=mock_vm)

        config = ABTestConfig(prompt_a="Q: {q}", prompt_b="Question: {q}", metrics=["accuracy"])
        test_cases = [
            {"q": "What is AI?", "expected": "AI is..."},
            {"q": "What is ML?", "expected": "ML is..."},
        ]

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "AI is artificial intelligence."
        mock_client.chat = AsyncMock(return_value=mock_response)

        result = await runner.run_test(config, test_cases, mock_client)
        total = len(result.results_a) + len(result.results_b)
        assert total == 2

    async def test_run_test_uses_evaluation_function(self):
        mock_vm = MagicMock()
        runner = ABTestRunner(version_manager=mock_vm)

        config = ABTestConfig(prompt_a="A", prompt_b="B", metrics=["score"])
        test_cases = [{"expected": "correct"}]

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "correct"
        mock_client.chat = AsyncMock(return_value=mock_response)

        def eval_fn(inp, expected, output):
            return {"score": 1.0 if output == expected else 0.0}

        result = await runner.run_test(config, test_cases, mock_client, evaluation_function=eval_fn)
        all_results = result.results_a + result.results_b
        assert all_results[0]["score"] == 1.0

    async def test_run_test_sets_end_time(self):
        mock_vm = MagicMock()
        runner = ABTestRunner(version_manager=mock_vm)

        config = ABTestConfig(prompt_a="A", prompt_b="B", metrics=["accuracy"])

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "response"
        mock_client.chat = AsyncMock(return_value=mock_response)

        result = await runner.run_test(config, [], mock_client)
        assert result.end_time is not None

    async def test_run_test_handles_llm_error(self):
        mock_vm = MagicMock()
        runner = ABTestRunner(version_manager=mock_vm)

        config = ABTestConfig(prompt_a="A", prompt_b="B", metrics=["accuracy"])
        test_cases = [{"expected": "answer"}]

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(side_effect=Exception("LLM error"))

        result = await runner.run_test(config, test_cases, mock_client)
        all_results = result.results_a + result.results_b
        assert len(all_results) == 1
        assert "error" in all_results[0]

    async def test_run_test_prompt_without_format_vars(self):
        mock_vm = MagicMock()
        runner = ABTestRunner(version_manager=mock_vm)

        config = ABTestConfig(prompt_a="Simple A", prompt_b="Simple B", metrics=["accuracy"])
        test_cases = [{"expected": "yes"}]  # no format vars in prompt

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "yes"
        mock_client.chat = AsyncMock(return_value=mock_response)

        result = await runner.run_test(config, test_cases, mock_client)
        total = len(result.results_a) + len(result.results_b)
        assert total == 1
