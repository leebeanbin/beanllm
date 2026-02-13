"""
DeepEval Metrics - 메트릭 생성 및 설정

DeepEval 메트릭의 팩토리 로직과 설정 상수를 제공합니다.
deepeval_wrapper.py에서 평가 실행 시 이 모듈을 사용합니다.
"""

from typing import Any, Dict, Type

# DeepEval 메트릭 이름 → 설명 (list_tasks 등에서 사용)
DEEPEVAL_METRIC_DESCRIPTIONS: Dict[str, str] = {
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


def get_deepeval_metric_map() -> Dict[str, Type[Any]]:
    """
    DeepEval 메트릭 클래스 매핑 반환.

    Returns:
        metric_name → Metric class 매핑
    """
    from deepeval.metrics import (  # type: ignore[import-untyped]
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

    return {
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


def create_deepeval_metric(
    metric_name: str,
    model: str = "gpt-4o-mini",
    threshold: float = 0.5,
    include_reason: bool = True,
    async_mode: bool = True,
    **kwargs: Any,
) -> Any:
    """
    DeepEval 메트릭 인스턴스 생성.

    Args:
        metric_name: 메트릭 이름 (answer_relevancy, faithfulness 등)
        model: LLM 모델
        threshold: 통과 임계값
        include_reason: 평가 이유 포함 여부
        async_mode: 비동기 모드 사용
        **kwargs: 메트릭별 추가 파라미터

    Returns:
        DeepEval Metric 인스턴스

    Raises:
        ImportError: deepeval 미설치 시
        ValueError: 알 수 없는 메트릭 이름
    """
    metric_map = get_deepeval_metric_map()

    if metric_name not in metric_map:
        raise ValueError(f"Unknown metric: {metric_name}. Available: {list(metric_map.keys())}")

    metric_class = metric_map[metric_name]

    return metric_class(
        model=model,
        threshold=threshold,
        include_reason=include_reason,
        async_mode=async_mode,
        **kwargs,
    )
