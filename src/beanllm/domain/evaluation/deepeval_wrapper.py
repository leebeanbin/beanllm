"""
DeepEval Wrapper - DeepEval 통합 (2024-2025)

DeepEval은 LLM 평가를 위한 종합 프레임워크로 14+ 메트릭을 제공합니다.

DeepEval 특징:
- LLM-as-a-Judge 접근법
- RAG 평가 특화 (Answer Relevancy, Faithfulness, Contextual Precision/Recall)
- Hallucination 감지
- Toxicity, Bias 평가
- Summarization 평가
- 500K+ downloads/month
- pytest 통합

Requirements:
    pip install deepeval

References:
    - https://github.com/confident-ai/deepeval
    - https://docs.confident-ai.com/
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from beanllm.domain.evaluation.base_framework import BaseEvaluationFramework
from beanllm.domain.evaluation.deepeval import BatchEvaluator, RAGEvaluators, SafetyEvaluators

try:
    from beanllm.utils.logging import get_logger
except ImportError:

    def get_logger(name: str) -> logging.Logger:  # type: ignore[misc]
        return logging.getLogger(name)


logger = get_logger(__name__)


class DeepEvalWrapper(BaseEvaluationFramework):
    """
    DeepEval 통합 래퍼

    DeepEval의 주요 메트릭을 beanLLM 스타일로 사용할 수 있게 합니다.

    지원 메트릭:
    - Answer Relevancy: 답변이 질문과 얼마나 관련있는지
    - Faithfulness: 답변이 컨텍스트에 충실한지 (Hallucination 방지)
    - Contextual Precision: 검색된 컨텍스트의 정밀도
    - Contextual Recall: 검색된 컨텍스트의 재현율
    - Hallucination: 환각 감지
    - Toxicity: 독성 평가
    - Bias: 편향 평가
    - Summarization: 요약 품질
    - G-Eval: 커스텀 평가 기준

    Example:
        ```python
        from beanllm.domain.evaluation import DeepEvalWrapper

        # 기본 사용
        evaluator = DeepEvalWrapper(
            model="gpt-4o-mini",
            api_key="sk-..."
        )

        # Answer Relevancy 평가
        result = evaluator.evaluate_answer_relevancy(
            question="What is AI?",
            answer="AI is artificial intelligence, a field of computer science."
        )
        print(result)  # {"score": 0.95, "reason": "..."}

        # Faithfulness 평가 (RAG)
        result = evaluator.evaluate_faithfulness(
            answer="Paris is the capital of France.",
            context=["Paris is the capital and largest city of France."]
        )
        print(result)  # {"score": 1.0, "reason": "..."}

        # 배치 평가
        results = evaluator.batch_evaluate(
            metric="answer_relevancy",
            data=[
                {"question": "Q1", "answer": "A1"},
                {"question": "Q2", "answer": "A2"},
            ]
        )
        ```
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        threshold: float = 0.5,
        include_reason: bool = True,
        async_mode: bool = True,
        **kwargs,
    ):
        """
        Args:
            model: LLM 모델 (gpt-4o-mini, gpt-4o, claude-3-5-sonnet-20241022 등)
            api_key: API 키 (None이면 환경변수)
            threshold: 통과 임계값 (기본: 0.5)
            include_reason: 평가 이유 포함 여부
            async_mode: 비동기 모드 사용
            **kwargs: 추가 파라미터
        """
        self.model = model
        self.api_key = api_key
        self.threshold = threshold
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.kwargs = kwargs

        # Compose evaluators
        self._rag_evaluators = RAGEvaluators(
            model=model,
            threshold=threshold,
            include_reason=include_reason,
            async_mode=async_mode,
            **kwargs,
        )
        self._safety_evaluators = SafetyEvaluators(
            model=model,
            threshold=threshold,
            include_reason=include_reason,
            async_mode=async_mode,
            **kwargs,
        )
        self._batch_evaluator = BatchEvaluator(
            rag_evaluators=self._rag_evaluators,
            safety_evaluators=self._safety_evaluators,
        )

    def evaluate_answer_relevancy(
        self,
        question: str,
        answer: str,
        **kwargs,
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
        return self._rag_evaluators.evaluate_answer_relevancy(question, answer, **kwargs)

    def evaluate_faithfulness(
        self,
        answer: str,
        context: Union[str, List[str]],
        **kwargs,
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
        return self._rag_evaluators.evaluate_faithfulness(answer, context, **kwargs)

    def evaluate_contextual_precision(
        self,
        question: str,
        context: List[str],
        expected_output: str,
        **kwargs,
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
        return self._rag_evaluators.evaluate_contextual_precision(
            question, context, expected_output, **kwargs
        )

    def evaluate_contextual_recall(
        self,
        question: str,
        context: List[str],
        expected_output: str,
        **kwargs,
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
        return self._rag_evaluators.evaluate_contextual_recall(
            question, context, expected_output, **kwargs
        )

    def evaluate_hallucination(
        self,
        answer: str,
        context: Union[str, List[str]],
        **kwargs,
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
        return self._safety_evaluators.evaluate_hallucination(answer, context, **kwargs)

    def evaluate_toxicity(
        self,
        text: str,
        **kwargs,
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
        return self._safety_evaluators.evaluate_toxicity(text, **kwargs)

    def batch_evaluate(
        self,
        metric: str,
        data: List[Dict[str, Any]],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        배치 평가

        여러 데이터에 대해 동일한 메트릭을 평가합니다.

        Args:
            metric: 메트릭 이름 (answer_relevancy, faithfulness 등)
            data: 평가 데이터 리스트
            **kwargs: 메트릭별 추가 파라미터

        Returns:
            평가 결과 리스트

        Example:
            ```python
            results = evaluator.batch_evaluate(
                metric="answer_relevancy",
                data=[
                    {"question": "What is AI?", "answer": "AI is ..."},
                    {"question": "What is ML?", "answer": "ML is ..."},
                ]
            )
            ```
        """
        return self._batch_evaluator.batch_evaluate(metric, data, **kwargs)

    # BaseEvaluationFramework 추상 메서드 구현

    def evaluate(self, **kwargs: Any) -> Dict[str, Any]:
        """
        평가 실행 (BaseEvaluationFramework 인터페이스)

        Args:
            metric: 메트릭 이름 (answer_relevancy, faithfulness 등)
            data: 평가 데이터 (단일 또는 리스트)
            **kwargs: 메트릭별 추가 파라미터

        Returns:
            평가 결과

        Example:
            ```python
            # 단일 평가
            result = evaluator.evaluate(
                metric="answer_relevancy",
                data={"question": "What is AI?", "answer": "AI is..."}
            )

            # 배치 평가
            results = evaluator.evaluate(
                metric="faithfulness",
                data=[
                    {"answer": "A1", "context": ["C1"]},
                    {"answer": "A2", "context": ["C2"]}
                ]
            )
            ```
        """
        metric: str = kwargs.pop("metric", "")
        data: Any = kwargs.pop("data", None)
        if not metric or data is None:
            raise ValueError("Both 'metric' and 'data' are required kwargs")

        if isinstance(data, list):
            # 배치 평가
            return {"results": self.batch_evaluate(metric=metric, data=data, **kwargs)}
        else:
            data_dict: Dict[str, Any] = data if isinstance(data, dict) else {}
            # 단일 평가
            if metric == "answer_relevancy":
                return self.evaluate_answer_relevancy(**data_dict, **kwargs)
            elif metric == "faithfulness":
                return self.evaluate_faithfulness(**data_dict, **kwargs)
            elif metric == "contextual_precision":
                return self.evaluate_contextual_precision(**data_dict, **kwargs)
            elif metric == "contextual_recall":
                return self.evaluate_contextual_recall(**data_dict, **kwargs)
            elif metric == "hallucination":
                return self.evaluate_hallucination(**data_dict, **kwargs)
            elif metric == "toxicity":
                return self.evaluate_toxicity(**data_dict, **kwargs)
            else:
                raise ValueError(
                    f"Unknown metric: {metric}. Available: {list(self.list_tasks().keys())}"
                )

    def list_tasks(self) -> Dict[str, str]:
        """
        사용 가능한 메트릭 목록 (BaseEvaluationFramework 인터페이스)

        Returns:
            {"metric_name": "description", ...}

        Example:
            ```python
            metrics = evaluator.list_tasks()
            print(metrics)
            # {
            #     "answer_relevancy": "답변이 질문과 얼마나 관련있는지",
            #     "faithfulness": "답변이 컨텍스트에 충실한지",
            #     ...
            # }
            ```
        """
        return {
            "answer_relevancy": "답변이 질문과 얼마나 관련있는지",
            "faithfulness": "답변이 컨텍스트에 충실한지 (Hallucination 방지)",
            "contextual_precision": "검색된 컨텍스트의 정밀도",
            "contextual_recall": "검색된 컨텍스트의 재현율",
            "hallucination": "환각 감지",
            "toxicity": "독성 평가",
            "bias": "편향 평가",
            "summarization": "요약 품질",
            "geval": "커스텀 평가 기준",
        }

    def __repr__(self) -> str:
        return (
            f"DeepEvalWrapper(model={self.model}, threshold={self.threshold}, "
            f"async={self.async_mode})"
        )
