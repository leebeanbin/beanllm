"""
Evaluation Service Implementation
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ...domain.evaluation.evaluator import Evaluator
from ...domain.evaluation.metrics import (
    AnswerRelevanceMetric,
    BLEUMetric,
    ContextPrecisionMetric,
    ExactMatchMetric,
    F1ScoreMetric,
    FaithfulnessMetric,
    ROUGEMetric,
    SemanticSimilarityMetric,
)
from ...dto.request.evaluation_request import (
    BatchEvaluationRequest,
    CreateEvaluatorRequest,
    EvaluationRequest,
    RAGEvaluationRequest,
    TextEvaluationRequest,
)
from ...dto.response.evaluation_response import (
    BatchEvaluationResponse,
    EvaluationResponse,
)
from ..evaluation_service import IEvaluationService

if TYPE_CHECKING:
    from ...domain.embeddings.base import Embedding
    from ...facade.client_facade import Client


class EvaluationServiceImpl(IEvaluationService):
    """평가 서비스 구현체"""

    def __init__(
        self,
        client: Optional["Client"] = None,
        embedding_model: Optional["Embedding"] = None,
    ):
        """
        Args:
            client: LLM 클라이언트 (LLMJudgeMetric 등에서 사용)
            embedding_model: 임베딩 모델 (SemanticSimilarityMetric에서 사용)
        """
        self.client = client
        self.embedding_model = embedding_model

    async def evaluate(self, request: "EvaluationRequest") -> "EvaluationResponse":
        """단일 평가 실행"""
        evaluator = Evaluator(metrics=request.metrics)
        result = evaluator.evaluate(
            prediction=request.prediction,
            reference=request.reference,
            **request.kwargs,
        )
        return EvaluationResponse(result=result)

    async def batch_evaluate(self, request: "BatchEvaluationRequest") -> "BatchEvaluationResponse":
        """배치 평가 실행 (내부적으로 자동 병렬 처리)"""
        evaluator = Evaluator(metrics=request.metrics)

        # 내부적으로 자동 병렬 처리 (사용자는 신경 쓸 필요 없음)
        # 기본 설정: max_concurrent=10, rate_limiter 자동 생성
        from ...utils.error_handling import AsyncTokenBucket

        rate_limiter = AsyncTokenBucket(rate=1.0, capacity=20.0)
        max_concurrent = 10

        results = await evaluator.batch_evaluate_async(
            predictions=request.predictions,
            references=request.references,
            max_concurrent=max_concurrent,
            rate_limiter=rate_limiter,
            **request.kwargs,
        )
        return BatchEvaluationResponse(results=results)

    async def evaluate_text(self, request: "TextEvaluationRequest") -> "EvaluationResponse":
        """텍스트 평가 (편의 함수)"""
        evaluator = Evaluator()

        for metric_name in request.metrics:
            if metric_name == "bleu":
                evaluator.add_metric(BLEUMetric())
            elif metric_name.startswith("rouge"):
                evaluator.add_metric(ROUGEMetric(rouge_type=metric_name))
            elif metric_name == "f1":
                evaluator.add_metric(F1ScoreMetric())
            elif metric_name == "exact_match":
                evaluator.add_metric(ExactMatchMetric())
            elif metric_name == "semantic":
                evaluator.add_metric(SemanticSimilarityMetric(embedding_model=self.embedding_model))
            else:
                raise ValueError(f"Unknown metric: {metric_name}")

        result = evaluator.evaluate(
            prediction=request.prediction,
            reference=request.reference,
            **request.kwargs,
        )
        return EvaluationResponse(result=result)

    async def evaluate_rag(self, request: "RAGEvaluationRequest") -> "EvaluationResponse":
        """RAG 평가"""
        evaluator = Evaluator()

        # Answer Relevance
        evaluator.add_metric(AnswerRelevanceMetric(client=self.client))

        # Context Precision
        evaluator.add_metric(ContextPrecisionMetric())

        # Faithfulness
        evaluator.add_metric(FaithfulnessMetric(client=self.client))

        # Ground truth가 있으면 일반 메트릭도 추가
        if request.ground_truth:
            evaluator.add_metric(F1ScoreMetric())
            evaluator.add_metric(ROUGEMetric("rouge-l"))

        result = evaluator.evaluate(
            prediction=request.answer,
            reference=request.ground_truth or request.question,
            contexts=request.contexts,
            **request.kwargs,
        )
        return EvaluationResponse(result=result)

    async def create_evaluator(self, request: "CreateEvaluatorRequest") -> "Evaluator":
        """Evaluator 생성"""
        evaluator = Evaluator()

        for name in request.metric_names:
            if name == "bleu":
                evaluator.add_metric(BLEUMetric())
            elif name.startswith("rouge"):
                evaluator.add_metric(ROUGEMetric(rouge_type=name))
            elif name == "f1":
                evaluator.add_metric(F1ScoreMetric())
            elif name == "exact_match":
                evaluator.add_metric(ExactMatchMetric())
            elif name == "semantic":
                evaluator.add_metric(SemanticSimilarityMetric(embedding_model=self.embedding_model))
            else:
                raise ValueError(f"Unknown metric: {name}")

        return evaluator
