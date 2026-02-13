"""
RAG Evaluators - RAG 평가 메트릭 모듈
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

from beanllm.domain.evaluation.deepeval_metrics import create_metric


class RAGEvaluators:
    """RAG 평가 메트릭"""

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

    def evaluate_answer_relevancy(
        self,
        question: str,
        answer: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Answer Relevancy 평가

        답변이 질문과 얼마나 관련있는지 평가합니다.

        Args:
            question: 질문
            answer: 답변
            **kwargs: 추가 파라미터

        Returns:
            {"score": float, "reason": str, "is_successful": bool}
        """
        from deepeval.test_case import LLMTestCase

        metric = self._get_metric("answer_relevancy", **kwargs)

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
        )

        metric.measure(test_case)

        return {
            "score": metric.score,
            "reason": metric.reason if self.include_reason else None,
            "is_successful": metric.is_successful(),
            "threshold": self.threshold,
        }

    def evaluate_faithfulness(
        self,
        answer: str,
        context: Union[str, List[str]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Faithfulness 평가 (Hallucination 방지)

        답변이 주어진 컨텍스트에 충실한지 평가합니다.

        Args:
            answer: 답변
            context: 컨텍스트 (문자열 또는 리스트)
            **kwargs: 추가 파라미터

        Returns:
            {"score": float, "reason": str, "is_successful": bool}
        """
        from deepeval.test_case import LLMTestCase

        metric = self._get_metric("faithfulness", **kwargs)

        # context를 리스트로 변환
        if isinstance(context, str):
            context = [context]

        test_case = LLMTestCase(
            input="",  # Faithfulness는 input 불필요
            actual_output=answer,
            retrieval_context=context,
        )

        metric.measure(test_case)

        return {
            "score": metric.score,
            "reason": metric.reason if self.include_reason else None,
            "is_successful": metric.is_successful(),
            "threshold": self.threshold,
        }

    def evaluate_contextual_precision(
        self,
        question: str,
        context: List[str],
        expected_output: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Contextual Precision 평가

        검색된 컨텍스트의 정밀도를 평가합니다.

        Args:
            question: 질문
            context: 검색된 컨텍스트 리스트
            expected_output: 기대 출력
            **kwargs: 추가 파라미터

        Returns:
            {"score": float, "reason": str, "is_successful": bool}
        """
        from deepeval.test_case import LLMTestCase

        metric = self._get_metric("contextual_precision", **kwargs)

        test_case = LLMTestCase(
            input=question,
            actual_output="",  # Contextual Precision은 actual_output 불필요
            expected_output=expected_output,
            retrieval_context=context,
        )

        metric.measure(test_case)

        return {
            "score": metric.score,
            "reason": metric.reason if self.include_reason else None,
            "is_successful": metric.is_successful(),
            "threshold": self.threshold,
        }

    def evaluate_contextual_recall(
        self,
        question: str,
        context: List[str],
        expected_output: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Contextual Recall 평가

        검색된 컨텍스트의 재현율을 평가합니다.

        Args:
            question: 질문
            context: 검색된 컨텍스트 리스트
            expected_output: 기대 출력
            **kwargs: 추가 파라미터

        Returns:
            {"score": float, "reason": str, "is_successful": bool}
        """
        from deepeval.test_case import LLMTestCase

        metric = self._get_metric("contextual_recall", **kwargs)

        test_case = LLMTestCase(
            input=question,
            actual_output="",
            expected_output=expected_output,
            retrieval_context=context,
        )

        metric.measure(test_case)

        return {
            "score": metric.score,
            "reason": metric.reason if self.include_reason else None,
            "is_successful": metric.is_successful(),
            "threshold": self.threshold,
        }
