"""
Evaluation Handler - 평가 요청 처리
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from beanllm.decorators.error_handler import handle_errors
from beanllm.decorators.validation import validate_input
from beanllm.dto.request.ml.evaluation_request import (
    BatchEvaluationRequest,
    CreateEvaluatorRequest,
    EvaluationRequest,
    RAGEvaluationRequest,
    TextEvaluationRequest,
)
from beanllm.dto.response.ml.evaluation_response import (
    BatchEvaluationResponse,
    EvaluationResponse,
)
from beanllm.handler.base_handler import BaseHandler
from beanllm.service.evaluation_service import IEvaluationService

if TYPE_CHECKING:
    from beanllm.domain.evaluation.base_metric import BaseMetric
    from beanllm.domain.evaluation.evaluator import Evaluator


class EvaluationHandler(BaseHandler[IEvaluationService]):
    """평가 요청 핸들러"""

    def __init__(self, evaluation_service: IEvaluationService):
        """
        Args:
            evaluation_service: 평가 서비스
        """
        super().__init__(evaluation_service)

    @handle_errors(error_message="Evaluation failed")
    @validate_input(
        required_params=["prediction", "reference"],
        param_types={"prediction": str, "reference": str, "metrics": list},
    )
    async def handle_evaluate(
        self,
        prediction: str,
        reference: str,
        metrics: Optional[List["BaseMetric"]] = None,
        **kwargs,
    ) -> "EvaluationResponse":
        """단일 평가 처리"""
        request = EvaluationRequest(
            prediction=prediction,
            reference=reference,
            metrics=metrics or [],
            **kwargs,
        )
        return await self._service.evaluate(request)

    @handle_errors(error_message="Batch evaluation failed")
    @validate_input(
        required_params=["predictions", "references"],
        param_types={"predictions": list, "references": list, "metrics": list},
    )
    async def handle_batch_evaluate(
        self,
        predictions: List[str],
        references: List[str],
        metrics: Optional[List["BaseMetric"]] = None,
        **kwargs,
    ) -> "BatchEvaluationResponse":
        """배치 평가 처리"""
        request = BatchEvaluationRequest(
            predictions=predictions,
            references=references,
            metrics=metrics or [],
            **kwargs,
        )
        return await self._service.batch_evaluate(request)

    @handle_errors(error_message="Text evaluation failed")
    @validate_input(
        required_params=["prediction", "reference"],
        param_types={"prediction": str, "reference": str, "metrics": list},
    )
    async def handle_evaluate_text(
        self,
        prediction: str,
        reference: str,
        metrics: Optional[List[str]] = None,
        **kwargs,
    ) -> "EvaluationResponse":
        """텍스트 평가 처리 (편의 함수)"""
        request = TextEvaluationRequest(
            prediction=prediction,
            reference=reference,
            metrics=metrics or [],
            **kwargs,
        )
        return await self._service.evaluate_text(request)

    @handle_errors(error_message="RAG evaluation failed")
    @validate_input(
        required_params=["question", "answer", "contexts"],
        param_types={"question": str, "answer": str, "contexts": list, "ground_truth": str},
    )
    async def handle_evaluate_rag(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        **kwargs,
    ) -> "EvaluationResponse":
        """RAG 평가 처리"""
        request = RAGEvaluationRequest(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
            **kwargs,
        )
        return await self._service.evaluate_rag(request)

    @handle_errors(error_message="Create evaluator failed")
    @validate_input(
        required_params=["metric_names"],
        param_types={"metric_names": list},
    )
    async def handle_create_evaluator(self, metric_names: List[str]) -> "Evaluator":
        """Evaluator 생성 처리"""
        request = CreateEvaluatorRequest(metric_names=metric_names)
        return await self._service.create_evaluator(request)
