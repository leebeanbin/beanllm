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

import logging
from typing import Any, Dict, List, Optional, Union

from .base_framework import BaseEvaluationFramework

try:
    from beanllm.utils.logger import get_logger
except ImportError:
    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)

# DeepEval 설치 여부 체크
try:
    HAS_DEEPEVAL = True
    # 실제 import는 사용 시점에 수행
except ImportError:
    HAS_DEEPEVAL = False


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

        # Lazy loading
        self._deepeval = None
        self._metrics_cache = {}

    def _check_dependencies(self):
        """의존성 확인"""
        try:
            import deepeval
        except ImportError:
            raise ImportError(
                "deepeval is required for DeepEvalWrapper. "
                "Install it with: pip install deepeval"
            )

        self._deepeval = deepeval

    def _get_metric(self, metric_name: str, **metric_kwargs):
        """
        DeepEval 메트릭 가져오기 (lazy loading + caching)

        Args:
            metric_name: 메트릭 이름
            **metric_kwargs: 메트릭별 추가 파라미터

        Returns:
            DeepEval Metric 객체
        """
        self._check_dependencies()

        # 캐시 키
        cache_key = f"{metric_name}_{str(metric_kwargs)}"
        if cache_key in self._metrics_cache:
            return self._metrics_cache[cache_key]

        # 메트릭 import
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            BiasMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            FaithfulnessMetric,
            GEval,
            HallucinationMetric,
            SummarizationMetric,
            ToxicityMetric,
        )

        # 메트릭 생성
        metric_map = {
            "answer_relevancy": AnswerRelevancyMetric,
            "faithfulness": FaithfulnessMetric,
            "contextual_precision": ContextualPrecisionMetric,
            "contextual_recall": ContextualRecallMetric,
            "hallucination": HallucinationMetric,
            "toxicity": ToxicityMetric,
            "bias": BiasMetric,
            "summarization": SummarizationMetric,
            "geval": GEval,
        }

        if metric_name not in metric_map:
            raise ValueError(
                f"Unknown metric: {metric_name}. "
                f"Available: {list(metric_map.keys())}"
            )

        metric_class = metric_map[metric_name]

        # 메트릭 인스턴스 생성
        metric = metric_class(
            model=self.model,
            threshold=self.threshold,
            include_reason=self.include_reason,
            async_mode=self.async_mode,
            **metric_kwargs,
            **self.kwargs,
        )

        # 캐시 저장
        self._metrics_cache[cache_key] = metric

        logger.info(f"DeepEval metric loaded: {metric_name}")

        return metric

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
        results = []

        for item in data:
            try:
                if metric == "answer_relevancy":
                    result = self.evaluate_answer_relevancy(**item, **kwargs)
                elif metric == "faithfulness":
                    result = self.evaluate_faithfulness(**item, **kwargs)
                elif metric == "contextual_precision":
                    result = self.evaluate_contextual_precision(**item, **kwargs)
                elif metric == "contextual_recall":
                    result = self.evaluate_contextual_recall(**item, **kwargs)
                elif metric == "hallucination":
                    result = self.evaluate_hallucination(**item, **kwargs)
                elif metric == "toxicity":
                    result = self.evaluate_toxicity(**item, **kwargs)
                else:
                    raise ValueError(f"Unknown metric: {metric}")

                results.append(result)

            except Exception as e:
                logger.error(f"DeepEval evaluation failed for item {item}: {e}")
                results.append({
                    "score": 0.0,
                    "reason": f"Error: {e}",
                    "is_successful": False,
                    "error": str(e),
                })

        logger.info(f"DeepEval batch evaluation completed: {len(results)} items")

        return results

    # BaseEvaluationFramework 추상 메서드 구현

    def evaluate(self, metric: str, data: Union[Dict[str, Any], List[Dict[str, Any]]], **kwargs) -> Dict[str, Any]:
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
        if isinstance(data, list):
            # 배치 평가
            return {"results": self.batch_evaluate(metric=metric, data=data, **kwargs)}
        else:
            # 단일 평가
            if metric == "answer_relevancy":
                return self.evaluate_answer_relevancy(**data, **kwargs)
            elif metric == "faithfulness":
                return self.evaluate_faithfulness(**data, **kwargs)
            elif metric == "contextual_precision":
                return self.evaluate_contextual_precision(**data, **kwargs)
            elif metric == "contextual_recall":
                return self.evaluate_contextual_recall(**data, **kwargs)
            elif metric == "hallucination":
                return self.evaluate_hallucination(**data, **kwargs)
            elif metric == "toxicity":
                return self.evaluate_toxicity(**data, **kwargs)
            else:
                raise ValueError(
                    f"Unknown metric: {metric}. "
                    f"Available: {list(self.list_tasks().keys())}"
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
