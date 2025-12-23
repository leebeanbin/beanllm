"""
EvaluationService 테스트 - Evaluation 서비스 구현체 테스트
"""

import pytest
from unittest.mock import Mock

from llmkit.dto.request.evaluation_request import (
    EvaluationRequest,
    BatchEvaluationRequest,
    TextEvaluationRequest,
    RAGEvaluationRequest,
    CreateEvaluatorRequest,
)
from llmkit.dto.response.evaluation_response import EvaluationResponse, BatchEvaluationResponse
from llmkit.domain.evaluation.metrics import BLEUMetric, ROUGEMetric, F1ScoreMetric
from llmkit.service.impl.evaluation_service_impl import EvaluationServiceImpl


class TestEvaluationService:
    """EvaluationService 테스트"""

    @pytest.fixture
    def evaluation_service(self):
        """EvaluationService 인스턴스"""
        return EvaluationServiceImpl()

    @pytest.mark.asyncio
    async def test_evaluate_basic(self, evaluation_service):
        """기본 평가 테스트"""
        metrics = [BLEUMetric(), ROUGEMetric("rouge-1")]
        request = EvaluationRequest(
            prediction="The cat sat on the mat",
            reference="The cat is on the mat",
            metrics=metrics,
        )

        response = await evaluation_service.evaluate(request)

        assert response is not None
        assert isinstance(response, EvaluationResponse)
        assert response.result is not None

    @pytest.mark.asyncio
    async def test_evaluate_with_f1(self, evaluation_service):
        """F1 Score 포함 평가 테스트"""
        metrics = [F1ScoreMetric()]
        request = EvaluationRequest(
            prediction="The cat sat on the mat",
            reference="The cat is on the mat",
            metrics=metrics,
        )

        response = await evaluation_service.evaluate(request)

        assert response is not None
        assert response.result is not None

    @pytest.mark.asyncio
    async def test_batch_evaluate(self, evaluation_service):
        """배치 평가 테스트"""
        metrics = [BLEUMetric()]
        request = BatchEvaluationRequest(
            predictions=["The cat sat", "The dog ran"],
            references=["The cat is", "The dog runs"],
            metrics=metrics,
        )

        response = await evaluation_service.batch_evaluate(request)

        assert response is not None
        assert isinstance(response, BatchEvaluationResponse)
        assert len(response.results) == 2

    @pytest.mark.asyncio
    async def test_evaluate_text_bleu(self, evaluation_service):
        """텍스트 평가 - BLEU 테스트"""
        request = TextEvaluationRequest(
            prediction="The cat sat on the mat",
            reference="The cat is on the mat",
            metrics=["bleu"],
        )

        response = await evaluation_service.evaluate_text(request)

        assert response is not None
        assert response.result is not None

    @pytest.mark.asyncio
    async def test_evaluate_text_rouge(self, evaluation_service):
        """텍스트 평가 - ROUGE 테스트"""
        request = TextEvaluationRequest(
            prediction="The cat sat on the mat",
            reference="The cat is on the mat",
            metrics=["rouge-1"],
        )

        response = await evaluation_service.evaluate_text(request)

        assert response is not None
        assert response.result is not None

    @pytest.mark.asyncio
    async def test_evaluate_text_f1(self, evaluation_service):
        """텍스트 평가 - F1 테스트"""
        request = TextEvaluationRequest(
            prediction="The cat sat on the mat",
            reference="The cat is on the mat",
            metrics=["f1"],
        )

        response = await evaluation_service.evaluate_text(request)

        assert response is not None
        assert response.result is not None

    @pytest.mark.asyncio
    async def test_evaluate_text_unknown_metric(self, evaluation_service):
        """알 수 없는 메트릭 테스트"""
        request = TextEvaluationRequest(
            prediction="The cat sat on the mat",
            reference="The cat is on the mat",
            metrics=["unknown_metric"],
        )

        with pytest.raises(ValueError, match="Unknown metric"):
            await evaluation_service.evaluate_text(request)

    @pytest.mark.asyncio
    async def test_evaluate_rag_basic(self, evaluation_service):
        """RAG 평가 기본 테스트"""
        # Mock client
        mock_client = Mock()
        evaluation_service.client = mock_client

        request = RAGEvaluationRequest(
            question="What is this about?",
            answer="This is about cats",
            contexts=["Context about cats", "More context"],
        )

        response = await evaluation_service.evaluate_rag(request)

        assert response is not None
        assert response.result is not None

    @pytest.mark.asyncio
    async def test_evaluate_rag_with_ground_truth(self, evaluation_service):
        """Ground truth 포함 RAG 평가 테스트"""
        # Mock client
        mock_client = Mock()
        evaluation_service.client = mock_client

        request = RAGEvaluationRequest(
            question="What is this about?",
            answer="This is about cats",
            contexts=["Context about cats"],
            ground_truth="This is about cats and dogs",
        )

        response = await evaluation_service.evaluate_rag(request)

        assert response is not None
        assert response.result is not None

    @pytest.mark.asyncio
    async def test_create_evaluator(self, evaluation_service):
        """Evaluator 생성 테스트"""
        request = CreateEvaluatorRequest(
            metric_names=["bleu", "rouge-1", "f1"],
        )

        evaluator = await evaluation_service.create_evaluator(request)

        assert evaluator is not None
        assert len(evaluator.metrics) == 3

    @pytest.mark.asyncio
    async def test_create_evaluator_unknown_metric(self, evaluation_service):
        """알 수 없는 메트릭으로 Evaluator 생성 테스트"""
        request = CreateEvaluatorRequest(
            metric_names=["unknown_metric"],
        )

        with pytest.raises(ValueError, match="Unknown metric"):
            await evaluation_service.create_evaluator(request)

    @pytest.mark.asyncio
    async def test_evaluate_text_multiple_metrics(self, evaluation_service):
        """여러 메트릭 포함 텍스트 평가 테스트"""
        request = TextEvaluationRequest(
            prediction="The cat sat on the mat",
            reference="The cat is on the mat",
            metrics=["bleu", "rouge-1", "rouge-l", "f1"],
        )

        response = await evaluation_service.evaluate_text(request)

        assert response is not None
        assert response.result is not None


