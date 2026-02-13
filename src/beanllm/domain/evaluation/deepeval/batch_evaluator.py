"""
Batch Evaluator - 배치 평가 모듈
"""

from __future__ import annotations

from typing import Any, Dict, List

from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class BatchEvaluator:
    """배치 평가기"""

    def __init__(
        self,
        rag_evaluators: Any,
        safety_evaluators: Any,
    ):
        """
        Args:
            rag_evaluators: RAG 평가기 인스턴스
            safety_evaluators: 안전성 평가기 인스턴스
        """
        self.rag_evaluators = rag_evaluators
        self.safety_evaluators = safety_evaluators

    def batch_evaluate(
        self,
        metric: str,
        data: List[Dict[str, Any]],
        **kwargs: Any,
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
        """
        results = []

        for item in data:
            try:
                if metric == "answer_relevancy":
                    result = self.rag_evaluators.evaluate_answer_relevancy(**item, **kwargs)
                elif metric == "faithfulness":
                    result = self.rag_evaluators.evaluate_faithfulness(**item, **kwargs)
                elif metric == "contextual_precision":
                    result = self.rag_evaluators.evaluate_contextual_precision(**item, **kwargs)
                elif metric == "contextual_recall":
                    result = self.rag_evaluators.evaluate_contextual_recall(**item, **kwargs)
                elif metric == "hallucination":
                    result = self.safety_evaluators.evaluate_hallucination(**item, **kwargs)
                elif metric == "toxicity":
                    result = self.safety_evaluators.evaluate_toxicity(**item, **kwargs)
                else:
                    raise ValueError(f"Unknown metric: {metric}")

                results.append(result)

            except Exception as e:
                logger.error(f"DeepEval evaluation failed for item {item}: {e}")
                results.append(
                    {
                        "score": 0.0,
                        "reason": f"Error: {e}",
                        "is_successful": False,
                        "error": str(e),
                    }
                )

        logger.info(f"DeepEval batch evaluation completed: {len(results)} items")

        return results
