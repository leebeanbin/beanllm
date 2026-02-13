"""
Safety Evaluators - 안전성 평가 메트릭 모듈
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

from beanllm.domain.evaluation.deepeval_metrics import create_metric


class SafetyEvaluators:
    """안전성 평가 메트릭"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        threshold: float = 0.5,
        include_reason: bool = True,
        async_mode: bool = True,
        **kwargs: Any,
    ):
        """
        Args:
            model: LLM 모델
            threshold: 통과 임계값
            include_reason: 평가 이유 포함 여부
            async_mode: 비동기 모드 사용
            **kwargs: 추가 파라미터
        """
        self.model = model
        self.threshold = threshold
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.kwargs = kwargs

    def _get_metric(self, metric_name: str, **metric_kwargs: Any) -> Any:
        """DeepEval 메트릭 가져오기"""
        return create_metric(
            metric_name=metric_name,
            model=self.model,
            threshold=self.threshold,
            include_reason=self.include_reason,
            async_mode=self.async_mode,
            **metric_kwargs,
            **self.kwargs,
        )

    def evaluate_hallucination(
        self,
        answer: str,
        context: Union[str, List[str]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Hallucination 평가

        답변이 컨텍스트에 없는 내용을 환각하는지 평가합니다.

        Args:
            answer: 답변
            context: 컨텍스트
            **kwargs: 추가 파라미터

        Returns:
            {"score": float, "reason": str, "is_successful": bool}
        """
        from deepeval.test_case import LLMTestCase

        metric = self._get_metric("hallucination", **kwargs)

        if isinstance(context, str):
            context = [context]

        test_case = LLMTestCase(
            input="",
            actual_output=answer,
            context=context,
        )

        metric.measure(test_case)

        return {
            "score": metric.score,
            "reason": metric.reason if self.include_reason else None,
            "is_successful": metric.is_successful(),
            "threshold": self.threshold,
        }

    def evaluate_toxicity(
        self,
        text: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Toxicity 평가

        텍스트의 독성을 평가합니다.

        Args:
            text: 평가할 텍스트
            **kwargs: 추가 파라미터

        Returns:
            {"score": float, "reason": str, "is_successful": bool}
        """
        from deepeval.test_case import LLMTestCase

        metric = self._get_metric("toxicity", **kwargs)

        test_case = LLMTestCase(
            input="",
            actual_output=text,
        )

        metric.measure(test_case)

        return {
            "score": metric.score,
            "reason": metric.reason if self.include_reason else None,
            "is_successful": metric.is_successful(),
            "threshold": self.threshold,
        }
