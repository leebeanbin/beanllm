"""
Comprehensive tests for DeepEvalWrapper.
Target: src/beanllm/domain/evaluation/deepeval_wrapper.py
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers — mock deepeval modules before importing DeepEvalWrapper
# ---------------------------------------------------------------------------


def _build_mock_metric(score: float = 0.85, reason: str = "Good", successful: bool = True):
    """Return a mock DeepEval metric with .score, .reason, .is_successful()."""
    m = MagicMock()
    m.score = score
    m.reason = reason
    m.is_successful.return_value = successful
    m.measure = MagicMock()
    return m


def _mock_deepeval_modules(score: float = 0.85):
    """Patch deepeval imports so the wrapper can be used without the library."""
    mock_test_case = MagicMock()
    mock_test_case_class = MagicMock(return_value=mock_test_case)

    deepeval_mock = MagicMock()
    deepeval_test_case_mock = MagicMock()
    deepeval_test_case_mock.LLMTestCase = mock_test_case_class
    deepeval_metrics_mock = MagicMock()

    metric_instance = _build_mock_metric(score)
    deepeval_metrics_mock.AnswerRelevancyMetric = MagicMock(return_value=metric_instance)
    deepeval_metrics_mock.FaithfulnessMetric = MagicMock(return_value=metric_instance)
    deepeval_metrics_mock.ContextualPrecisionMetric = MagicMock(return_value=metric_instance)
    deepeval_metrics_mock.ContextualRecallMetric = MagicMock(return_value=metric_instance)
    deepeval_metrics_mock.HallucinationMetric = MagicMock(return_value=metric_instance)
    deepeval_metrics_mock.ToxicityMetric = MagicMock(return_value=metric_instance)

    return {
        "deepeval": deepeval_mock,
        "deepeval.test_case": deepeval_test_case_mock,
        "deepeval.metrics": deepeval_metrics_mock,
    }, metric_instance


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestDeepEvalWrapperInit:
    def test_default_init(self):
        from beanllm.domain.evaluation.deepeval_wrapper import DeepEvalWrapper

        ew = DeepEvalWrapper.__new__(DeepEvalWrapper)
        ew.model = "gpt-4o-mini"
        ew.threshold = 0.5
        ew.include_reason = True
        ew.async_mode = True
        ew.kwargs = {}
        # Just checks attributes survive
        assert ew.model == "gpt-4o-mini"
        assert ew.threshold == 0.5

    def test_init_stores_params(self):
        from unittest.mock import MagicMock, patch

        # Patch the composed sub-evaluators so we don't need deepeval installed
        with (
            patch("beanllm.domain.evaluation.deepeval_wrapper.RAGEvaluators", MagicMock()),
            patch("beanllm.domain.evaluation.deepeval_wrapper.SafetyEvaluators", MagicMock()),
            patch("beanllm.domain.evaluation.deepeval_wrapper.BatchEvaluator", MagicMock()),
        ):
            from beanllm.domain.evaluation.deepeval_wrapper import DeepEvalWrapper

            ew = DeepEvalWrapper(
                model="gpt-4o",
                api_key="sk-test",
                threshold=0.7,
                include_reason=False,
                async_mode=False,
            )
            assert ew.model == "gpt-4o"
            assert ew.api_key == "sk-test"
            assert ew.threshold == 0.7
            assert not ew.include_reason
            assert not ew.async_mode

    def test_repr(self):
        with (
            patch("beanllm.domain.evaluation.deepeval_wrapper.RAGEvaluators", MagicMock()),
            patch("beanllm.domain.evaluation.deepeval_wrapper.SafetyEvaluators", MagicMock()),
            patch("beanllm.domain.evaluation.deepeval_wrapper.BatchEvaluator", MagicMock()),
        ):
            from beanllm.domain.evaluation.deepeval_wrapper import DeepEvalWrapper

            ew = DeepEvalWrapper()
            r = repr(ew)
            assert "DeepEvalWrapper" in r


# ---------------------------------------------------------------------------
# list_tasks
# ---------------------------------------------------------------------------


class TestListTasks:
    def _make_wrapper(self):
        with (
            patch("beanllm.domain.evaluation.deepeval_wrapper.RAGEvaluators", MagicMock()),
            patch("beanllm.domain.evaluation.deepeval_wrapper.SafetyEvaluators", MagicMock()),
            patch("beanllm.domain.evaluation.deepeval_wrapper.BatchEvaluator", MagicMock()),
        ):
            from beanllm.domain.evaluation.deepeval_wrapper import DeepEvalWrapper

            return DeepEvalWrapper()

    def test_list_tasks_returns_dict(self):
        ew = self._make_wrapper()
        tasks = ew.list_tasks()
        assert isinstance(tasks, dict)

    def test_list_tasks_has_expected_keys(self):
        ew = self._make_wrapper()
        tasks = ew.list_tasks()
        assert "answer_relevancy" in tasks
        assert "faithfulness" in tasks
        assert "hallucination" in tasks
        assert "toxicity" in tasks


# ---------------------------------------------------------------------------
# evaluate() dispatch
# ---------------------------------------------------------------------------


class TestEvaluateDispatch:
    def _make_wrapper_with_mocks(self):
        mock_rag = MagicMock()
        mock_rag.evaluate_answer_relevancy.return_value = {
            "score": 0.9,
            "reason": "ok",
            "is_successful": True,
            "threshold": 0.5,
        }
        mock_rag.evaluate_faithfulness.return_value = {
            "score": 0.8,
            "reason": "ok",
            "is_successful": True,
            "threshold": 0.5,
        }
        mock_rag.evaluate_contextual_precision.return_value = {
            "score": 0.7,
            "reason": "ok",
            "is_successful": True,
            "threshold": 0.5,
        }
        mock_rag.evaluate_contextual_recall.return_value = {
            "score": 0.6,
            "reason": "ok",
            "is_successful": True,
            "threshold": 0.5,
        }
        mock_safety = MagicMock()
        mock_safety.evaluate_hallucination.return_value = {
            "score": 0.1,
            "reason": "ok",
            "is_successful": False,
            "threshold": 0.5,
        }
        mock_safety.evaluate_toxicity.return_value = {
            "score": 0.0,
            "reason": "ok",
            "is_successful": True,
            "threshold": 0.5,
        }
        mock_batch = MagicMock()
        mock_batch.batch_evaluate.return_value = [{"score": 0.9}]

        with (
            patch(
                "beanllm.domain.evaluation.deepeval_wrapper.RAGEvaluators",
                MagicMock(return_value=mock_rag),
            ),
            patch(
                "beanllm.domain.evaluation.deepeval_wrapper.SafetyEvaluators",
                MagicMock(return_value=mock_safety),
            ),
            patch(
                "beanllm.domain.evaluation.deepeval_wrapper.BatchEvaluator",
                MagicMock(return_value=mock_batch),
            ),
        ):
            from importlib import reload

            import beanllm.domain.evaluation.deepeval_wrapper as dew_mod

            # Need a fresh instance with our mocks
            from beanllm.domain.evaluation.deepeval_wrapper import DeepEvalWrapper

            ew = DeepEvalWrapper.__new__(DeepEvalWrapper)
            ew.model = "gpt-4o-mini"
            ew.threshold = 0.5
            ew.include_reason = True
            ew.async_mode = True
            ew.kwargs = {}
            ew._rag_evaluators = mock_rag
            ew._safety_evaluators = mock_safety
            ew._batch_evaluator = mock_batch
            return ew

    def test_evaluate_answer_relevancy(self):
        ew = self._make_wrapper_with_mocks()
        result = ew.evaluate(
            metric="answer_relevancy",
            data={"question": "What is AI?", "answer": "AI is artificial intelligence."},
        )
        assert "score" in result

    def test_evaluate_faithfulness(self):
        ew = self._make_wrapper_with_mocks()
        result = ew.evaluate(
            metric="faithfulness",
            data={
                "answer": "Paris is the capital.",
                "context": ["Paris is the capital of France."],
            },
        )
        assert "score" in result

    def test_evaluate_contextual_precision(self):
        ew = self._make_wrapper_with_mocks()
        result = ew.evaluate(
            metric="contextual_precision",
            data={"question": "Q", "context": ["C"], "expected_output": "E"},
        )
        assert "score" in result

    def test_evaluate_contextual_recall(self):
        ew = self._make_wrapper_with_mocks()
        result = ew.evaluate(
            metric="contextual_recall",
            data={"question": "Q", "context": ["C"], "expected_output": "E"},
        )
        assert "score" in result

    def test_evaluate_hallucination(self):
        ew = self._make_wrapper_with_mocks()
        result = ew.evaluate(
            metric="hallucination",
            data={"answer": "Some text.", "context": ["Context"]},
        )
        assert "score" in result

    def test_evaluate_toxicity(self):
        ew = self._make_wrapper_with_mocks()
        result = ew.evaluate(
            metric="toxicity",
            data={"text": "Nice weather today."},
        )
        assert "score" in result

    def test_evaluate_batch(self):
        ew = self._make_wrapper_with_mocks()
        result = ew.evaluate(
            metric="answer_relevancy",
            data=[
                {"question": "Q1", "answer": "A1"},
                {"question": "Q2", "answer": "A2"},
            ],
        )
        assert "results" in result

    def test_evaluate_missing_metric_raises(self):
        ew = self._make_wrapper_with_mocks()
        with pytest.raises(ValueError, match="Both 'metric' and 'data' are required"):
            ew.evaluate(data={"question": "Q"})

    def test_evaluate_missing_data_raises(self):
        ew = self._make_wrapper_with_mocks()
        with pytest.raises(ValueError, match="Both 'metric' and 'data' are required"):
            ew.evaluate(metric="answer_relevancy")

    def test_evaluate_unknown_metric_raises(self):
        ew = self._make_wrapper_with_mocks()
        with pytest.raises(ValueError, match="Unknown metric"):
            ew.evaluate(metric="totally_unknown", data={"foo": "bar"})


# ---------------------------------------------------------------------------
# Delegation methods
# ---------------------------------------------------------------------------


class TestDelegationMethods:
    def _make_wrapper_with_mocks(self):
        mock_rag = MagicMock()
        mock_rag.evaluate_answer_relevancy.return_value = {
            "score": 0.9,
            "reason": "r",
            "is_successful": True,
            "threshold": 0.5,
        }
        mock_rag.evaluate_faithfulness.return_value = {
            "score": 0.8,
            "reason": "r",
            "is_successful": True,
            "threshold": 0.5,
        }
        mock_rag.evaluate_contextual_precision.return_value = {
            "score": 0.7,
            "reason": "r",
            "is_successful": True,
            "threshold": 0.5,
        }
        mock_rag.evaluate_contextual_recall.return_value = {
            "score": 0.6,
            "reason": "r",
            "is_successful": True,
            "threshold": 0.5,
        }
        mock_safety = MagicMock()
        mock_safety.evaluate_hallucination.return_value = {
            "score": 0.1,
            "reason": "r",
            "is_successful": False,
            "threshold": 0.5,
        }
        mock_safety.evaluate_toxicity.return_value = {
            "score": 0.0,
            "reason": "r",
            "is_successful": True,
            "threshold": 0.5,
        }
        mock_batch = MagicMock()
        mock_batch.batch_evaluate.return_value = [{"score": 0.9}]

        from beanllm.domain.evaluation.deepeval_wrapper import DeepEvalWrapper

        ew = DeepEvalWrapper.__new__(DeepEvalWrapper)
        ew.model = "gpt-4o-mini"
        ew.threshold = 0.5
        ew.include_reason = True
        ew.async_mode = True
        ew.kwargs = {}
        ew._rag_evaluators = mock_rag
        ew._safety_evaluators = mock_safety
        ew._batch_evaluator = mock_batch
        return ew

    def test_evaluate_answer_relevancy_direct(self):
        ew = self._make_wrapper_with_mocks()
        result = ew.evaluate_answer_relevancy(question="Q", answer="A")
        assert result["score"] == 0.9

    def test_evaluate_faithfulness_direct(self):
        ew = self._make_wrapper_with_mocks()
        result = ew.evaluate_faithfulness(answer="A", context=["C"])
        assert result["score"] == 0.8

    def test_evaluate_faithfulness_string_context(self):
        ew = self._make_wrapper_with_mocks()
        result = ew.evaluate_faithfulness(answer="A", context="single context string")
        assert "score" in result

    def test_evaluate_contextual_precision_direct(self):
        ew = self._make_wrapper_with_mocks()
        result = ew.evaluate_contextual_precision(question="Q", context=["C"], expected_output="E")
        assert result["score"] == 0.7

    def test_evaluate_contextual_recall_direct(self):
        ew = self._make_wrapper_with_mocks()
        result = ew.evaluate_contextual_recall(question="Q", context=["C"], expected_output="E")
        assert result["score"] == 0.6

    def test_evaluate_hallucination_direct(self):
        ew = self._make_wrapper_with_mocks()
        result = ew.evaluate_hallucination(answer="A", context=["C"])
        assert result["score"] == 0.1

    def test_evaluate_toxicity_direct(self):
        ew = self._make_wrapper_with_mocks()
        result = ew.evaluate_toxicity(text="Hello world")
        assert result["score"] == 0.0

    def test_batch_evaluate_direct(self):
        ew = self._make_wrapper_with_mocks()
        results = ew.batch_evaluate(
            metric="answer_relevancy",
            data=[{"question": "Q1", "answer": "A1"}],
        )
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# BatchEvaluator unit tests
# ---------------------------------------------------------------------------


class TestBatchEvaluator:
    def _make_batch_evaluator(self):
        mock_rag = MagicMock()
        mock_rag.evaluate_answer_relevancy.return_value = {"score": 0.9}
        mock_rag.evaluate_faithfulness.return_value = {"score": 0.8}
        mock_rag.evaluate_contextual_precision.return_value = {"score": 0.7}
        mock_rag.evaluate_contextual_recall.return_value = {"score": 0.6}
        mock_safety = MagicMock()
        mock_safety.evaluate_hallucination.return_value = {"score": 0.1}
        mock_safety.evaluate_toxicity.return_value = {"score": 0.0}

        from beanllm.domain.evaluation.deepeval.batch_evaluator import BatchEvaluator

        return BatchEvaluator(rag_evaluators=mock_rag, safety_evaluators=mock_safety)

    def test_batch_answer_relevancy(self):
        be = self._make_batch_evaluator()
        results = be.batch_evaluate(
            "answer_relevancy",
            [{"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2"}],
        )
        assert len(results) == 2

    def test_batch_faithfulness(self):
        be = self._make_batch_evaluator()
        results = be.batch_evaluate(
            "faithfulness",
            [{"answer": "A", "context": ["C"]}],
        )
        assert len(results) == 1

    def test_batch_contextual_precision(self):
        be = self._make_batch_evaluator()
        results = be.batch_evaluate(
            "contextual_precision",
            [{"question": "Q", "context": ["C"], "expected_output": "E"}],
        )
        assert len(results) == 1

    def test_batch_contextual_recall(self):
        be = self._make_batch_evaluator()
        results = be.batch_evaluate(
            "contextual_recall",
            [{"question": "Q", "context": ["C"], "expected_output": "E"}],
        )
        assert len(results) == 1

    def test_batch_hallucination(self):
        be = self._make_batch_evaluator()
        results = be.batch_evaluate(
            "hallucination",
            [{"answer": "A", "context": ["C"]}],
        )
        assert len(results) == 1

    def test_batch_toxicity(self):
        be = self._make_batch_evaluator()
        results = be.batch_evaluate(
            "toxicity",
            [{"text": "Hello"}],
        )
        assert len(results) == 1

    def test_batch_unknown_metric_error_in_result(self):
        be = self._make_batch_evaluator()
        results = be.batch_evaluate(
            "unknown_metric",
            [{"question": "Q"}],
        )
        assert len(results) == 1
        assert results[0]["score"] == 0.0
        assert results[0]["is_successful"] is False

    def test_batch_handles_exception_per_item(self):
        mock_rag = MagicMock()
        mock_rag.evaluate_answer_relevancy.side_effect = [
            {"score": 0.9},
            RuntimeError("API error"),
        ]
        mock_safety = MagicMock()
        from beanllm.domain.evaluation.deepeval.batch_evaluator import BatchEvaluator

        be = BatchEvaluator(rag_evaluators=mock_rag, safety_evaluators=mock_safety)
        results = be.batch_evaluate(
            "answer_relevancy",
            [{"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2"}],
        )
        assert len(results) == 2
        assert results[1]["score"] == 0.0
        assert "error" in results[1]
