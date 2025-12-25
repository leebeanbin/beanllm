"""
Evaluation Facade - 기존 Evaluation API를 위한 Facade
책임: 하위 호환성 유지, 내부적으로는 Handler/Service 사용
SOLID 원칙:
- Facade 패턴: 복잡한 내부 구조를 단순한 인터페이스로
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, List, Optional

from ..domain.evaluation.results import BatchEvaluationResult

if TYPE_CHECKING:
    from ..domain.evaluation.base_metric import BaseMetric


class EvaluatorFacade:
    """
    통합 평가기 (Facade 패턴)

    기존 API를 유지하면서 내부적으로는 Handler/Service 사용

    Example:
        >>> evaluator = EvaluatorFacade()
        >>> evaluator.add_metric(BLEUMetric())
        >>> result = evaluator.evaluate("prediction", "reference")
    """

    def __init__(self, metrics: Optional[List["BaseMetric"]] = None):
        """
        Args:
            metrics: 초기 메트릭 리스트
        """
        self.metrics = metrics or []

        # Handler/Service 초기화 (의존성 주입)
        self._init_services()

    def _init_services(self) -> None:
        """Service 및 Handler 초기화 (의존성 주입) - DI Container 사용"""
        from ..handler.evaluation_handler import EvaluationHandler
        from ..service.impl.evaluation_service_impl import EvaluationServiceImpl

        # EvaluationService 생성
        evaluation_service = EvaluationServiceImpl()

        # EvaluationHandler 생성 (직접 생성 - 커스텀 Service 사용)
        self._evaluation_handler = EvaluationHandler(evaluation_service)

    def add_metric(self, metric: "BaseMetric") -> "EvaluatorFacade":
        """메트릭 추가"""
        self.metrics.append(metric)
        return self

    def evaluate(self, prediction: str, reference: str, **kwargs) -> BatchEvaluationResult:
        """모든 메트릭으로 평가"""
        # 동기 메서드이지만 내부적으로는 비동기 사용
        response = asyncio.run(
            self._evaluation_handler.handle_evaluate(
                prediction=prediction,
                reference=reference,
                metrics=self.metrics,
                **kwargs,
            )
        )
        return response.result

    def batch_evaluate(
        self, predictions: List[str], references: List[str], **kwargs
    ) -> List[BatchEvaluationResult]:
        """배치 평가"""
        # 동기 메서드이지만 내부적으로는 비동기 사용
        response = asyncio.run(
            self._evaluation_handler.handle_batch_evaluate(
                predictions=predictions,
                references=references,
                metrics=self.metrics,
                **kwargs,
            )
        )
        return response.results


# 편의 함수들 (기존 API 유지)


def evaluate_text(
    prediction: str, reference: str, metrics: Optional[List[str]] = None, **kwargs
) -> BatchEvaluationResult:
    """
    간편한 텍스트 평가

    Args:
        prediction: 예측 텍스트
        reference: 참조 텍스트
        metrics: 사용할 메트릭 이름 리스트 (기본: ["bleu", "rouge", "f1"])
    """
    # Handler/Service 초기화 - DI Container 사용
    from ..handler.evaluation_handler import EvaluationHandler
    from ..service.impl.evaluation_service_impl import EvaluationServiceImpl

    evaluation_service = EvaluationServiceImpl()
    handler = EvaluationHandler(evaluation_service)

    # 동기 메서드이지만 내부적으로는 비동기 사용
    response = asyncio.run(
        handler.handle_evaluate_text(
            prediction=prediction,
            reference=reference,
            metrics=metrics,
            **kwargs,
        )
    )
    return response.result


def evaluate_rag(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None,
    **kwargs,
) -> BatchEvaluationResult:
    """
    RAG 시스템 평가

    Args:
        question: 원래 질문
        answer: 생성된 답변
        contexts: 검색된 컨텍스트
        ground_truth: 정답 (있는 경우)
    """
    # Handler/Service 초기화 - DI Container 사용
    from ..handler.evaluation_handler import EvaluationHandler
    from ..service.impl.evaluation_service_impl import EvaluationServiceImpl

    evaluation_service = EvaluationServiceImpl()
    handler = EvaluationHandler(evaluation_service)

    # 동기 메서드이지만 내부적으로는 비동기 사용
    response = asyncio.run(
        handler.handle_evaluate_rag(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
            **kwargs,
        )
    )
    return response.result


def create_evaluator(metric_names: List[str]) -> Evaluator:
    """간편한 Evaluator 생성"""
    # Handler/Service 초기화 - DI Container 사용
    from ..handler.evaluation_handler import EvaluationHandler
    from ..service.impl.evaluation_service_impl import EvaluationServiceImpl

    evaluation_service = EvaluationServiceImpl()
    handler = EvaluationHandler(evaluation_service)

    # 동기 메서드이지만 내부적으로는 비동기 사용
    return asyncio.run(handler.handle_create_evaluator(metric_names=metric_names))


# 기존 Evaluator 클래스를 EvaluatorFacade로 alias (하위 호환성)
Evaluator = EvaluatorFacade
