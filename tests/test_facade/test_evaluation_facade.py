"""
Evaluation Facade 테스트
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

try:
    from llmkit.facade.evaluation_facade import EvaluatorFacade
    from llmkit.domain.evaluation.results import BatchEvaluationResult

    FACADE_AVAILABLE = True
except ImportError:
    FACADE_AVAILABLE = False


@pytest.mark.skipif(not FACADE_AVAILABLE, reason="EvaluatorFacade not available")
class TestEvaluatorFacade:
    @pytest.fixture
    def evaluator(self):
        with patch("llmkit.facade.evaluation_facade.HandlerFactory") as mock_factory:
            mock_handler = MagicMock()
            from llmkit.domain.evaluation.results import EvaluationResult

            mock_response = Mock()
            mock_response.result = BatchEvaluationResult(
                results=[EvaluationResult(metric_name="test", score=0.5)], average_score=0.5
            )

            async def mock_handle_evaluate(*args, **kwargs):
                return mock_response

            mock_handler.handle_evaluate = MagicMock(side_effect=mock_handle_evaluate)

            mock_response_batch = Mock()
            mock_response_batch.results = [
                BatchEvaluationResult(
                    results=[EvaluationResult(metric_name="test", score=0.5)], average_score=0.5
                )
            ]

            async def mock_handle_batch_evaluate(*args, **kwargs):
                return mock_response_batch

            mock_handler.handle_batch_evaluate = MagicMock(side_effect=mock_handle_batch_evaluate)

            mock_handler_factory = Mock()
            mock_handler_factory.create_evaluation_handler.return_value = mock_handler
            mock_factory.return_value = mock_handler_factory

            evaluator = EvaluatorFacade()
            evaluator._evaluation_handler = mock_handler
            return evaluator

    def test_evaluate(self, evaluator):
        result = evaluator.evaluate("prediction", "reference")
        assert isinstance(result, BatchEvaluationResult)
        assert result.average_score == 0.5
        assert evaluator._evaluation_handler.handle_evaluate.called

    def test_batch_evaluate(self, evaluator):
        results = evaluator.batch_evaluate(["pred1", "pred2"], ["ref1", "ref2"])
        assert isinstance(results, list)
        assert len(results) == 1
        assert evaluator._evaluation_handler.handle_batch_evaluate.called

    def test_add_metric(self, evaluator):
        mock_metric = Mock()
        result = evaluator.add_metric(mock_metric)
        assert result is evaluator
        assert len(evaluator.metrics) == 1

