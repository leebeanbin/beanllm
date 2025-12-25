"""
Evaluation Service Interface
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..domain.evaluation.evaluator import Evaluator
    from ..dto.request.evaluation_request import (
        BatchEvaluationRequest,
        CreateEvaluatorRequest,
        EvaluationRequest,
        RAGEvaluationRequest,
        TextEvaluationRequest,
    )
    from ..dto.response.evaluation_response import (
        BatchEvaluationResponse,
        EvaluationResponse,
    )


class IEvaluationService(ABC):
    """평가 서비스 인터페이스"""

    @abstractmethod
    async def evaluate(self, request: "EvaluationRequest") -> "EvaluationResponse":
        """단일 평가 실행"""
        pass

    @abstractmethod
    async def batch_evaluate(self, request: "BatchEvaluationRequest") -> "BatchEvaluationResponse":
        """배치 평가 실행"""
        pass

    @abstractmethod
    async def evaluate_text(self, request: "TextEvaluationRequest") -> "EvaluationResponse":
        """텍스트 평가 (편의 함수)"""
        pass

    @abstractmethod
    async def evaluate_rag(self, request: "RAGEvaluationRequest") -> "EvaluationResponse":
        """RAG 평가"""
        pass

    @abstractmethod
    async def create_evaluator(self, request: "CreateEvaluatorRequest") -> "Evaluator":
        """Evaluator 생성"""
        pass
