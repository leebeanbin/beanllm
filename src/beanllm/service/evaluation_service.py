"""
Evaluation Service Interface

LLM 출력의 품질을 평가하는 서비스 인터페이스.
텍스트 품질, RAG 성능 (Faithfulness, Relevance 등) 평가를 지원합니다.
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
    """
    평가 서비스 인터페이스

    LLM 응답 품질을 다양한 메트릭으로 평가합니다.
    단일/배치 평가, 텍스트 평가, RAG 전용 평가를 지원합니다.

    Example:
        >>> service: IEvaluationService = factory.create_evaluation_service(client)
        >>> result = await service.evaluate_text(
        ...     TextEvaluationRequest(
        ...         text="Paris is the capital of France.",
        ...         reference="Paris is the capital of France.",
        ...         metrics=["exact_match", "bleu"],
        ...     )
        ... )
        >>> print(result.scores)
    """

    @abstractmethod
    async def evaluate(self, request: "EvaluationRequest") -> "EvaluationResponse":
        """
        단일 평가를 실행합니다.

        Args:
            request: 평가 대상 텍스트, 참조 텍스트, 메트릭 설정이 포함된 요청

        Returns:
            EvaluationResponse: 메트릭별 점수가 포함된 평가 결과

        Raises:
            ValueError: 메트릭이 유효하지 않은 경우
            RuntimeError: 평가 실행 실패 시
        """
        pass

    @abstractmethod
    async def batch_evaluate(self, request: "BatchEvaluationRequest") -> "BatchEvaluationResponse":
        """
        여러 샘플에 대해 배치 평가를 실행합니다.

        Args:
            request: 여러 평가 항목이 포함된 요청

        Returns:
            BatchEvaluationResponse: 전체 평가 결과 및 집계 통계

        Raises:
            ValueError: 요청이 비어있거나 형식이 잘못된 경우
        """
        pass

    @abstractmethod
    async def evaluate_text(self, request: "TextEvaluationRequest") -> "EvaluationResponse":
        """
        텍스트 품질을 평가합니다 (편의 함수).

        BLEU, ROUGE, F1, Exact Match 등의 메트릭을 사용합니다.

        Args:
            request: 평가 대상 텍스트, 참조 텍스트, 메트릭 목록이 포함된 요청

        Returns:
            EvaluationResponse: 평가 결과

        Raises:
            ValueError: 텍스트 또는 참조가 비어있는 경우
        """
        pass

    @abstractmethod
    async def evaluate_rag(self, request: "RAGEvaluationRequest") -> "EvaluationResponse":
        """
        RAG 파이프라인 품질을 평가합니다.

        Faithfulness, Answer Relevance, Context Precision 등
        RAG 전용 메트릭을 사용하여 검색 및 생성 품질을 평가합니다.

        Args:
            request: 질의, 응답, 컨텍스트, 참조 답변이 포함된 요청

        Returns:
            EvaluationResponse: RAG 평가 결과

        Raises:
            ValueError: 필수 필드가 누락된 경우
            RuntimeError: 평가 프레임워크 실행 실패 시
        """
        pass

    @abstractmethod
    async def create_evaluator(self, request: "CreateEvaluatorRequest") -> "Evaluator":
        """
        커스텀 설정의 Evaluator 인스턴스를 생성합니다.

        Args:
            request: 메트릭 설정, LLM 클라이언트, 임베딩 모델이 포함된 요청

        Returns:
            Evaluator: 설정된 평가기 인스턴스

        Raises:
            ValueError: 설정이 유효하지 않은 경우
        """
        pass
