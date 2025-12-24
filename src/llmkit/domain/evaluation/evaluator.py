"""
Evaluator - 통합 평가기
"""

import asyncio
from typing import TYPE_CHECKING, List, Optional

from .base_metric import BaseMetric
from .results import BatchEvaluationResult, EvaluationResult

if TYPE_CHECKING:
    from ...utils.error_handling import AsyncTokenBucket


class Evaluator:
    """
    통합 평가기

    여러 메트릭을 한 번에 실행
    """

    def __init__(self, metrics: Optional[List[BaseMetric]] = None):
        self.metrics = metrics or []

    def add_metric(self, metric: BaseMetric) -> "Evaluator":
        """메트릭 추가"""
        self.metrics.append(metric)
        return self

    def evaluate(self, prediction: str, reference: str, **kwargs) -> BatchEvaluationResult:
        """모든 메트릭으로 평가"""
        results = []

        for metric in self.metrics:
            try:
                result = metric.compute(prediction, reference, **kwargs)
                results.append(result)
            except Exception as e:
                # 에러가 나도 다른 메트릭은 계속 실행
                results.append(
                    EvaluationResult(metric_name=metric.name, score=0.0, metadata={"error": str(e)})
                )

        if not results:
            average_score = 0.0
        else:
            average_score = sum(r.score for r in results) / len(results)

        return BatchEvaluationResult(
            results=results, average_score=average_score, metadata={"metrics_count": len(results)}
        )

    def batch_evaluate(
        self, predictions: List[str], references: List[str], **kwargs
    ) -> List[BatchEvaluationResult]:
        """배치 평가 (순차 처리)"""
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        batch_results = []
        for pred, ref in zip(predictions, references):
            result = self.evaluate(pred, ref, **kwargs)
            batch_results.append(result)

        return batch_results

    async def batch_evaluate_async(
        self,
        predictions: List[str],
        references: List[str],
        max_concurrent: int = 10,
        rate_limiter: Optional["AsyncTokenBucket"] = None,
        **kwargs,
    ) -> List[BatchEvaluationResult]:
        """
        배치 평가 (병렬 처리 + Rate Limiting)

        Args:
            predictions: 예측 리스트
            references: 참조 리스트
            max_concurrent: 최대 동시 실행 수
            rate_limiter: Token Bucket Rate Limiter (None이면 기본값 사용)
            **kwargs: 추가 파라미터

        Returns:
            평가 결과 리스트
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        # Token Bucket (기본값)
        if rate_limiter is None:
            from ...utils.error_handling import AsyncTokenBucket

            rate_limiter = AsyncTokenBucket(rate=1.0, capacity=20.0)

        semaphore = asyncio.Semaphore(max_concurrent)

        async def evaluate_one(pred: str, ref: str):
            """단일 평가 (Rate Limiting + Semaphore)"""
            await rate_limiter.wait(cost=1.0)
            async with semaphore:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self.evaluate, pred, ref, **kwargs)

        # 모든 평가를 병렬 실행
        tasks = [evaluate_one(pred, ref) for pred, ref in zip(predictions, references)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 예외 처리
        batch_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # 예외 발생 시 빈 결과 생성
                batch_results.append(
                    BatchEvaluationResult(
                        results=[], average_score=0.0, metadata={"error": str(result), "index": i}
                    )
                )
            else:
                batch_results.append(result)

        return batch_results
