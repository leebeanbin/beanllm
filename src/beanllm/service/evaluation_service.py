"""
Evaluation Service Interface

LLM 출력 품질 평가의 계약을 정의합니다.
텍스트 유사도, RAG 정확도, 배치 평가 등을 지원합니다.
Handler 레이어는 이 인터페이스에만 의존합니다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from beanllm.domain.evaluation.evaluator import Evaluator
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


class IEvaluationService(ABC):
    """평가 서비스 인터페이스

    LLM 출력의 품질을 다양한 메트릭으로 측정합니다.
    BLEU, ROUGE, F1, Semantic Similarity, LLM Judge 등을 지원합니다.

    Example:
        >>> service: IEvaluationService = factory.create_evaluation_service()
        >>> result = await service.evaluate_text(
        ...     TextEvaluationRequest(
        ...         prediction="Paris is the capital of France.",
        ...         reference="The capital of France is Paris.",
        ...         metrics=["bleu", "rouge", "semantic_similarity"],
        ...     )
        ... )
        >>> print(result.scores)
    """

    @abstractmethod
    async def evaluate(self, request: "EvaluationRequest") -> "EvaluationResponse":
        """단일 평가를 실행합니다.

        Args:
            request: 예측값, 참조값, 메트릭 목록을 포함하는 요청

        Returns:
            EvaluationResponse: 메트릭별 점수 및 요약

        Raises:
            ValueError: 지원하지 않는 메트릭 지정 시
        """

    @abstractmethod
    async def batch_evaluate(self, request: "BatchEvaluationRequest") -> "BatchEvaluationResponse":
        """여러 샘플에 대한 배치 평가를 실행합니다.

        Args:
            request: 평가 샘플 목록 및 메트릭 설정

        Returns:
            BatchEvaluationResponse: 개별 결과 및 전체 평균 점수

        Raises:
            ValueError: 빈 샘플 목록
        """

    @abstractmethod
    async def evaluate_text(self, request: "TextEvaluationRequest") -> "EvaluationResponse":
        """텍스트 품질을 평가합니다 (편의 메서드).

        예측 텍스트와 참조 텍스트 간의 유사도를 측정합니다.

        Args:
            request: 예측 텍스트, 참조 텍스트, 메트릭 목록

        Returns:
            EvaluationResponse: 메트릭별 점수
        """

    @abstractmethod
    async def evaluate_rag(self, request: "RAGEvaluationRequest") -> "EvaluationResponse":
        """RAG 파이프라인의 품질을 평가합니다.

        Faithfulness, Answer Relevance, Context Precision 등
        RAG 전용 메트릭을 측정합니다.

        Args:
            request: 질문, 답변, 컨텍스트, 참조 답변을 포함하는 요청

        Returns:
            EvaluationResponse: RAG 메트릭별 점수

        Raises:
            ValueError: 필수 필드 누락 시
        """

    @abstractmethod
    async def create_evaluator(self, request: "CreateEvaluatorRequest") -> "Evaluator":
        """커스텀 Evaluator 인스턴스를 생성합니다.

        재사용 가능한 평가기를 설정하여 여러 평가에 활용합니다.

        Args:
            request: 메트릭 설정, LLM 클라이언트, 임베딩 모델을 포함하는 요청

        Returns:
            Evaluator: 설정된 평가기 인스턴스
        """
