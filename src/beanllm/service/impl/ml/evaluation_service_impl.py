"""
Evaluation Service Implementation

OCP 준수: metric registry 패턴으로 새 메트릭 추가 시 기존 코드 수정 불필요
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, cast

from beanllm.domain.evaluation.evaluator import Evaluator
from beanllm.domain.evaluation.metrics import (
    AnswerRelevanceMetric,
    BLEUMetric,
    ContextPrecisionMetric,
    ExactMatchMetric,
    F1ScoreMetric,
    FaithfulnessMetric,
    ROUGEMetric,
    SemanticSimilarityMetric,
)
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
from beanllm.service.evaluation_service import IEvaluationService

if TYPE_CHECKING:
    from beanllm.domain.embeddings.base import BaseEmbedding
    from beanllm.domain.evaluation.base_metric import BaseMetric
    from beanllm.domain.evaluation.protocols import LLMClientProtocol

# MetricFactory: metric_name -> factory function
MetricFactory = Callable[..., "BaseMetric"]


class EvaluationServiceImpl(IEvaluationService):
    """평가 서비스 구현체

    OCP: 새 메트릭 추가 시 register_metric()만 호출하면 됩니다.

    Example:
        >>> service = EvaluationServiceImpl()
        >>> service.register_metric("my_metric", lambda **kw: MyCustomMetric())
    """

    # 기본 메트릭 레지스트리 (클래스 레벨)
    _default_metric_registry: Dict[str, MetricFactory] = {}

    def __init__(
        self,
        client: Optional["LLMClientProtocol"] = None,
        embedding_model: Optional["BaseEmbedding"] = None,
    ):
        """
        Args:
            client: LLM 클라이언트 프로토콜 (LLMJudgeMetric 등에서 사용)
            embedding_model: 임베딩 모델 (SemanticSimilarityMetric에서 사용)
        """
        self.client = client
        self.embedding_model = embedding_model
        # 인스턴스별 레지스트리 (기본 + 커스텀)
        self._metric_registry: Dict[str, MetricFactory] = {
            **self._get_builtin_metrics(),
        }

    def _get_builtin_metrics(self) -> Dict[str, MetricFactory]:
        """내장 메트릭 팩토리 매핑을 반환합니다."""
        return {
            "bleu": lambda **_kw: BLEUMetric(),
            "f1": lambda **_kw: F1ScoreMetric(),
            "exact_match": lambda **_kw: ExactMatchMetric(),
            "semantic": lambda **kw: SemanticSimilarityMetric(
                embedding_model=kw.get("embedding_model", self.embedding_model)
            ),
        }

    def register_metric(self, name: str, factory: MetricFactory) -> None:
        """
        커스텀 메트릭을 등록합니다 (OCP: 기존 코드 수정 없이 확장).

        Args:
            name: 메트릭 이름
            factory: 메트릭 인스턴스를 생성하는 팩토리 함수

        Example:
            >>> service.register_metric("my_metric", lambda **kw: MyCustomMetric())
        """
        self._metric_registry[name] = factory

    def _resolve_metric(self, name: str, **kwargs: Any) -> "BaseMetric":
        """메트릭 이름으로 인스턴스를 생성합니다.

        Args:
            name: 메트릭 이름 (예: "bleu", "rouge-l", "semantic")
            **kwargs: 메트릭 생성에 필요한 추가 인수

        Returns:
            BaseMetric: 생성된 메트릭 인스턴스

        Raises:
            ValueError: 등록되지 않은 메트릭 이름인 경우
        """
        # 정확히 일치하는 레지스트리 항목 우선
        if name in self._metric_registry:
            return self._metric_registry[name](**kwargs)
        # rouge-1, rouge-l 등 rouge 계열 패턴 매칭
        if name.startswith("rouge"):
            return ROUGEMetric(rouge_type=name)
        raise ValueError(
            f"Unknown metric: {name!r}. "
            f"Available: {sorted(self._metric_registry.keys())} + rouge-* variants. "
            f"Use register_metric() to add custom metrics."
        )

    def _build_evaluator_from_names(self, metric_names: list[str]) -> Evaluator:
        """메트릭 이름 목록으로 Evaluator를 구성합니다."""
        evaluator = Evaluator()
        for name in metric_names:
            evaluator.add_metric(self._resolve_metric(name))
        return evaluator

    async def evaluate(self, request: "EvaluationRequest") -> "EvaluationResponse":
        """단일 평가 실행"""
        evaluator = Evaluator(metrics=request.metrics)
        result = evaluator.evaluate(
            prediction=request.prediction,
            reference=request.reference,
            **request.extra_params,
        )
        return EvaluationResponse(result=result)

    async def batch_evaluate(self, request: "BatchEvaluationRequest") -> "BatchEvaluationResponse":
        """배치 평가 실행 (내부적으로 자동 병렬 처리)"""
        evaluator = Evaluator(metrics=request.metrics)

        from beanllm.domain.protocols import (
            ConcurrencyControllerProtocol,
            RateLimiterProtocol,
        )
        from beanllm.infrastructure.distributed import (
            ConcurrencyController,
            get_rate_limiter,
        )

        rate_limiter = cast(Optional["RateLimiterProtocol"], get_rate_limiter())
        concurrency_controller = cast(
            Optional["ConcurrencyControllerProtocol"], ConcurrencyController()
        )
        max_concurrent = 10

        results = await evaluator.batch_evaluate_async(
            predictions=request.predictions,
            references=request.references,
            max_concurrent=max_concurrent,
            rate_limiter=rate_limiter,
            concurrency_controller=concurrency_controller,
            **request.extra_params,
        )
        return BatchEvaluationResponse(results=results)

    async def evaluate_text(self, request: "TextEvaluationRequest") -> "EvaluationResponse":
        """텍스트 평가 (편의 함수)"""
        evaluator = self._build_evaluator_from_names(list(request.metrics))
        result = evaluator.evaluate(
            prediction=request.prediction,
            reference=request.reference,
            **request.extra_params,
        )
        return EvaluationResponse(result=result)

    async def evaluate_rag(self, request: "RAGEvaluationRequest") -> "EvaluationResponse":
        """RAG 평가"""
        evaluator = Evaluator()

        evaluator.add_metric(AnswerRelevanceMetric(client=self.client))
        evaluator.add_metric(ContextPrecisionMetric())
        evaluator.add_metric(FaithfulnessMetric(client=self.client))

        if request.ground_truth:
            evaluator.add_metric(F1ScoreMetric())
            evaluator.add_metric(ROUGEMetric("rouge-l"))

        result = evaluator.evaluate(
            prediction=request.answer,
            reference=request.ground_truth or request.question,
            contexts=request.contexts,
            **request.extra_params,
        )
        return EvaluationResponse(result=result)

    async def create_evaluator(self, request: "CreateEvaluatorRequest") -> "Evaluator":
        """Evaluator 생성"""
        return self._build_evaluator_from_names(list(request.metric_names))
