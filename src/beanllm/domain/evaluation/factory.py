"""
Evaluation Framework Factory - 평가 프레임워크 생성 함수

외부 평가 프레임워크를 쉽게 생성할 수 있는 Factory 함수를 제공합니다.
"""

from typing import Optional

from .base_framework import BaseEvaluationFramework

try:
    from beanllm.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def create_evaluation_framework(
    framework: str,
    **kwargs,
) -> BaseEvaluationFramework:
    """
    평가 프레임워크 생성 (Factory 함수)

    Args:
        framework: 프레임워크 종류
            - "ragas": RAGAS (Reference-free RAG 평가)
            - "deepeval": DeepEval (LLM-as-a-Judge, RAG 평가)
            - "lm-eval" or "lm-eval-harness": LM Evaluation Harness (표준 벤치마크)
        **kwargs: 프레임워크별 초기화 파라미터
            - RAGAS: model="gpt-4o-mini", embeddings="text-embedding-3-small", ...
            - DeepEval: model="gpt-4o-mini", api_key=None, threshold=0.5, ...
            - LM Eval Harness: model="hf", model_args="...", batch_size="auto", ...

    Returns:
        BaseEvaluationFramework 인스턴스

    Raises:
        ValueError: 알 수 없는 프레임워크
        ImportError: 프레임워크가 설치되지 않음

    Example:
        ```python
        from beanllm.domain.evaluation import create_evaluation_framework

        # RAGAS (Reference-free RAG 평가)
        evaluator = create_evaluation_framework(
            framework="ragas",
            model="gpt-4o-mini",
            embeddings="text-embedding-3-small"
        )

        result = evaluator.evaluate(
            metric="faithfulness",
            data={"question": "Q", "answer": "A", "contexts": ["C"]}
        )

        # DeepEval
        evaluator = create_evaluation_framework(
            framework="deepeval",
            model="gpt-4o-mini",
            api_key="sk-..."
        )

        result = evaluator.evaluate(
            metric="answer_relevancy",
            data={"question": "What is AI?", "answer": "AI is..."}
        )

        # LM Eval Harness
        evaluator = create_evaluation_framework(
            framework="lm-eval",
            model="hf",
            model_args="pretrained=meta-llama/Llama-3.2-1B"
        )

        results = evaluator.evaluate(
            tasks=["mmlu", "hellaswag"],
            num_fewshot=5
        )
        ```
    """
    framework = framework.lower()

    if framework == "ragas":
        try:
            from .ragas_wrapper import RAGASWrapper
            logger.info("Creating RAGAS framework")
            return RAGASWrapper(**kwargs)
        except ImportError:
            raise ImportError(
                "ragas is required for RAGASWrapper. "
                "Install it with: pip install ragas"
            )

    elif framework == "deepeval":
        try:
            from .deepeval_wrapper import DeepEvalWrapper
            logger.info("Creating DeepEval framework")
            return DeepEvalWrapper(**kwargs)
        except ImportError:
            raise ImportError(
                "deepeval is required for DeepEvalWrapper. "
                "Install it with: pip install deepeval"
            )

    elif framework in ["lm-eval", "lm-eval-harness", "lm_eval", "lm_eval_harness"]:
        try:
            from .lm_eval_harness_wrapper import LMEvalHarnessWrapper
            logger.info("Creating LM Eval Harness framework")
            return LMEvalHarnessWrapper(**kwargs)
        except ImportError:
            raise ImportError(
                "lm-eval is required for LMEvalHarnessWrapper. "
                "Install it with: pip install lm-eval"
            )

    else:
        raise ValueError(
            f"Unknown framework: {framework}. "
            f"Available: ragas, deepeval, lm-eval"
        )


def list_available_frameworks() -> dict:
    """
    사용 가능한 평가 프레임워크 목록

    Returns:
        {"framework_name": "description", ...}

    Example:
        ```python
        from beanllm.domain.evaluation import list_available_frameworks

        frameworks = list_available_frameworks()
        print(frameworks)
        # {
        #     "deepeval": "DeepEval - LLM-as-a-Judge, RAG 평가 (14+ metrics)",
        #     "lm-eval": "LM Evaluation Harness - 표준 벤치마크 (60+ tasks)"
        # }
        ```
    """
    return {
        "ragas": "RAGAS - Reference-free RAG 평가 (Faithfulness, Answer Relevancy 등)",
        "deepeval": "DeepEval - LLM-as-a-Judge, RAG 평가 (14+ metrics)",
        "lm-eval": "LM Evaluation Harness - 표준 벤치마크 (60+ tasks)",
    }
