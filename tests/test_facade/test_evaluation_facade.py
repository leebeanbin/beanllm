"""Tests for facade/ml/evaluation_facade.py — EvaluatorFacade and convenience functions."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from beanllm.domain.evaluation.results import BatchEvaluationResult
from beanllm.facade.ml.evaluation_facade import (
    Evaluator,
    EvaluatorFacade,
    create_evaluator,
    evaluate_rag,
    evaluate_text,
)


def _make_eval_result(score=0.75):
    from beanllm.domain.evaluation.results import EvaluationResult

    return BatchEvaluationResult(
        results=[EvaluationResult(metric_name="bleu", score=score)],
        average_score=score,
    )


def _make_evaluator():
    """Create EvaluatorFacade with fully mocked dependencies."""
    mock_handler = MagicMock()
    mock_result = _make_eval_result()
    mock_response = MagicMock()
    mock_response.result = mock_result
    mock_batch_response = MagicMock()
    mock_batch_response.results = [mock_result, mock_result]

    mock_handler.handle_evaluate = AsyncMock(return_value=mock_response)
    mock_handler.handle_batch_evaluate = AsyncMock(return_value=mock_batch_response)
    mock_handler.handle_evaluate_text = AsyncMock(return_value=mock_response)
    mock_handler.handle_evaluate_rag = AsyncMock(return_value=mock_response)
    mock_handler.handle_create_evaluator = AsyncMock(return_value=MagicMock())

    container_patcher = patch("beanllm.utils.core.di_container.get_container")
    handler_patcher = patch("beanllm.handler.ml.evaluation_handler.EvaluationHandler")

    mock_get_container = container_patcher.start()
    MockHandler = handler_patcher.start()

    mock_container = MagicMock()
    mock_get_container.return_value = mock_container
    MockHandler.return_value = mock_handler

    evaluator = EvaluatorFacade()
    return evaluator, mock_handler, [container_patcher, handler_patcher]


def _stop(patchers):
    for p in patchers:
        p.stop()


# ---------------------------------------------------------------------------
# EvaluatorFacade.__init__
# ---------------------------------------------------------------------------


class TestEvaluatorFacadeInit:
    def test_creates_with_empty_metrics(self):
        evaluator, _, p = _make_evaluator()
        try:
            assert evaluator.metrics == []
        finally:
            _stop(p)

    def test_creates_with_initial_metrics(self):
        mock_metric = MagicMock()
        with (
            patch("beanllm.utils.core.di_container.get_container") as mc,
            patch("beanllm.handler.ml.evaluation_handler.EvaluationHandler"),
        ):
            mc.return_value = MagicMock()
            evaluator = EvaluatorFacade(metrics=[mock_metric])
        assert len(evaluator.metrics) == 1

    def test_evaluator_alias_equals_evaluatorfacade(self):
        assert Evaluator is EvaluatorFacade


# ---------------------------------------------------------------------------
# add_metric
# ---------------------------------------------------------------------------


class TestEvaluatorAddMetric:
    def test_add_metric_appends(self):
        evaluator, _, p = _make_evaluator()
        try:
            mock_metric = MagicMock()
            evaluator.add_metric(mock_metric)
            assert mock_metric in evaluator.metrics
        finally:
            _stop(p)

    def test_add_metric_returns_self(self):
        evaluator, _, p = _make_evaluator()
        try:
            result = evaluator.add_metric(MagicMock())
            assert result is evaluator
        finally:
            _stop(p)

    def test_chaining(self):
        evaluator, _, p = _make_evaluator()
        try:
            m1, m2 = MagicMock(), MagicMock()
            evaluator.add_metric(m1).add_metric(m2)
            assert len(evaluator.metrics) == 2
        finally:
            _stop(p)


# ---------------------------------------------------------------------------
# evaluate (sync)
# ---------------------------------------------------------------------------


class TestEvaluatorEvaluate:
    def test_evaluate_returns_batch_result(self):
        evaluator, _, p = _make_evaluator()
        try:
            result = evaluator.evaluate("prediction", "reference")
            assert isinstance(result, BatchEvaluationResult)
        finally:
            _stop(p)

    def test_evaluate_passes_prediction_and_reference(self):
        evaluator, handler, p = _make_evaluator()
        try:
            evaluator.evaluate("my pred", "my ref")
            call_kwargs = handler.handle_evaluate.call_args.kwargs
            assert call_kwargs.get("prediction") == "my pred"
            assert call_kwargs.get("reference") == "my ref"
        finally:
            _stop(p)

    def test_evaluate_result_has_correct_score(self):
        evaluator, _, p = _make_evaluator()
        try:
            result = evaluator.evaluate("p", "r")
            assert result.average_score == 0.75
        finally:
            _stop(p)


# ---------------------------------------------------------------------------
# evaluate_async
# ---------------------------------------------------------------------------


class TestEvaluatorEvaluateAsync:
    async def test_evaluate_async_returns_result(self):
        evaluator, _, p = _make_evaluator()
        try:
            result = await evaluator.evaluate_async("pred", "ref")
            assert isinstance(result, BatchEvaluationResult)
        finally:
            _stop(p)

    async def test_evaluate_async_calls_handler(self):
        evaluator, handler, p = _make_evaluator()
        try:
            await evaluator.evaluate_async("p", "r")
            handler.handle_evaluate.assert_awaited()
        finally:
            _stop(p)


# ---------------------------------------------------------------------------
# batch_evaluate (sync)
# ---------------------------------------------------------------------------


class TestEvaluatorBatchEvaluate:
    def test_batch_evaluate_returns_list(self):
        evaluator, _, p = _make_evaluator()
        try:
            results = evaluator.batch_evaluate(["p1", "p2"], ["r1", "r2"])
            assert isinstance(results, list)
            assert len(results) == 2
        finally:
            _stop(p)

    def test_batch_evaluate_passes_predictions(self):
        evaluator, handler, p = _make_evaluator()
        try:
            evaluator.batch_evaluate(["a", "b"], ["c", "d"])
            call_kwargs = handler.handle_batch_evaluate.call_args.kwargs
            assert call_kwargs.get("predictions") == ["a", "b"]
        finally:
            _stop(p)


# ---------------------------------------------------------------------------
# batch_evaluate_async
# ---------------------------------------------------------------------------


class TestEvaluatorBatchEvaluateAsync:
    async def test_batch_evaluate_async_returns_list(self):
        evaluator, _, p = _make_evaluator()
        try:
            results = await evaluator.batch_evaluate_async(["p1"], ["r1"])
            assert isinstance(results, list)
        finally:
            _stop(p)


# ---------------------------------------------------------------------------
# evaluate_text convenience function
# ---------------------------------------------------------------------------


class TestEvaluateText:
    def test_evaluate_text_returns_result(self):
        mock_result = _make_eval_result()
        mock_response = MagicMock()
        mock_response.result = mock_result
        with (
            patch("beanllm.utils.core.di_container.get_container") as mc,
            patch("beanllm.handler.ml.evaluation_handler.EvaluationHandler") as MockH,
        ):
            mc.return_value = MagicMock()
            mock_handler = MagicMock()
            mock_handler.handle_evaluate_text = AsyncMock(return_value=mock_response)
            MockH.return_value = mock_handler
            result = evaluate_text("prediction", "reference")
        assert isinstance(result, BatchEvaluationResult)

    def test_evaluate_text_passes_params(self):
        mock_result = _make_eval_result()
        mock_response = MagicMock()
        mock_response.result = mock_result
        with (
            patch("beanllm.utils.core.di_container.get_container") as mc,
            patch("beanllm.handler.ml.evaluation_handler.EvaluationHandler") as MockH,
        ):
            mc.return_value = MagicMock()
            mock_handler = MagicMock()
            mock_handler.handle_evaluate_text = AsyncMock(return_value=mock_response)
            MockH.return_value = mock_handler
            evaluate_text("pred", "ref", metrics=["bleu"])
            call_kwargs = mock_handler.handle_evaluate_text.call_args.kwargs
            assert call_kwargs.get("prediction") == "pred"
            assert call_kwargs.get("reference") == "ref"


# ---------------------------------------------------------------------------
# evaluate_rag convenience function
# ---------------------------------------------------------------------------


class TestEvaluateRAG:
    def test_evaluate_rag_returns_result(self):
        mock_result = _make_eval_result()
        mock_response = MagicMock()
        mock_response.result = mock_result
        with (
            patch("beanllm.utils.core.di_container.get_container") as mc,
            patch("beanllm.handler.ml.evaluation_handler.EvaluationHandler") as MockH,
        ):
            mc.return_value = MagicMock()
            mock_handler = MagicMock()
            mock_handler.handle_evaluate_rag = AsyncMock(return_value=mock_response)
            MockH.return_value = mock_handler
            result = evaluate_rag("question", "answer", ["context1"])
        assert isinstance(result, BatchEvaluationResult)

    def test_evaluate_rag_passes_params(self):
        mock_result = _make_eval_result()
        mock_response = MagicMock()
        mock_response.result = mock_result
        with (
            patch("beanllm.utils.core.di_container.get_container") as mc,
            patch("beanllm.handler.ml.evaluation_handler.EvaluationHandler") as MockH,
        ):
            mc.return_value = MagicMock()
            mock_handler = MagicMock()
            mock_handler.handle_evaluate_rag = AsyncMock(return_value=mock_response)
            MockH.return_value = mock_handler
            evaluate_rag("Q", "A", ["ctx"], ground_truth="GT")
            call_kwargs = mock_handler.handle_evaluate_rag.call_args.kwargs
            assert call_kwargs.get("question") == "Q"
            assert call_kwargs.get("answer") == "A"
            assert call_kwargs.get("ground_truth") == "GT"


# ---------------------------------------------------------------------------
# create_evaluator convenience function
# ---------------------------------------------------------------------------


class TestCreateEvaluator:
    def test_create_evaluator_returns_facade_instance(self):
        with (
            patch("beanllm.utils.core.di_container.get_container") as mc,
            patch("beanllm.handler.ml.evaluation_handler.EvaluationHandler") as MockH,
        ):
            mc.return_value = MagicMock()
            mock_handler = MagicMock()
            mock_facade = MagicMock(spec=EvaluatorFacade)
            mock_handler.handle_create_evaluator = AsyncMock(return_value=mock_facade)
            MockH.return_value = mock_handler
            result = create_evaluator(["bleu", "rouge"])
        assert result is mock_facade
