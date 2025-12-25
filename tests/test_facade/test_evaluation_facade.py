"""
Evaluation Facade 테스트
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

try:
    from beanllm.facade.evaluation_facade import EvaluatorFacade
    from beanllm.domain.evaluation.results import BatchEvaluationResult

    FACADE_AVAILABLE = True
except ImportError:
    FACADE_AVAILABLE = False


@pytest.mark.skipif(not FACADE_AVAILABLE, reason="EvaluatorFacade not available")
class TestEvaluatorFacade:
    @pytest.fixture
    def evaluator(self):
        from beanllm.domain.evaluation.results import EvaluationResult
        from beanllm.dto.response.evaluation_response import EvaluationResponse, BatchEvaluationResponse

        # Facade가 직접 Handler를 생성하므로 Handler를 Mock으로 교체
        with patch("beanllm.handler.evaluation_handler.EvaluationHandler") as mock_handler_class:
            mock_handler = MagicMock()
            
            # handle_evaluate는 EvaluationResponse를 반환
            mock_response = EvaluationResponse(
                result=BatchEvaluationResult(
                    results=[EvaluationResult(metric_name="test", score=0.5)], average_score=0.5
                )
            )

            async def mock_handle_evaluate(*args, **kwargs):
                return mock_response

            mock_handler.handle_evaluate = MagicMock(side_effect=mock_handle_evaluate)

            # handle_batch_evaluate는 BatchEvaluationResponse를 반환
            mock_response_batch = BatchEvaluationResponse(
                results=[
                    BatchEvaluationResult(
                        results=[EvaluationResult(metric_name="test", score=0.5)], average_score=0.5
                    )
                ]
            )

            async def mock_handle_batch_evaluate(*args, **kwargs):
                return mock_response_batch

            mock_handler.handle_batch_evaluate = MagicMock(side_effect=mock_handle_batch_evaluate)
            
            # Handler 클래스가 인스턴스화될 때 mock_handler 반환
            mock_handler_class.return_value = mock_handler

            evaluator = EvaluatorFacade()
            # 실제 생성된 Handler를 Mock으로 교체
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

