"""
RAGAS Wrapper - RAGAS 통합 (2024-2025)

RAGAS (Retrieval Augmented Generation Assessment)는 RAG 시스템을 위한
reference-free 평가 프레임워크입니다.

RAGAS 특징:
- Reference-free 평가 (ground truth 없이도 평가 가능)
- RAG 특화 메트릭 (Faithfulness, Answer Relevancy, Context Precision/Recall)
- LangChain, LlamaIndex 통합
- Component-level 평가 (Retriever, Generator 개별 평가)
- 20K+ stars on GitHub

RAGAS vs DeepEval:
- RAGAS: RAG에 특화, reference-free, 오픈소스
- DeepEval: 더 광범위한 메트릭 (Toxicity, Bias 등), 상용 서비스 연계

Requirements:
    pip install ragas

References:
    - https://github.com/explodinggradients/ragas
    - https://docs.ragas.io/
"""

import logging
from typing import Any, Dict, List, Optional, Union, cast

from .base_framework import BaseEvaluationFramework

try:
    from beanllm.utils.logging import get_logger
except ImportError:

    def get_logger(name: str) -> logging.Logger:  # type: ignore[misc]
        return logging.getLogger(name)


logger = get_logger(__name__)


class RAGASWrapper(BaseEvaluationFramework):
    """
    RAGAS 통합 래퍼

    RAGAS의 주요 메트릭을 beanLLM 스타일로 사용할 수 있게 합니다.

    지원 메트릭: Faithfulness, Answer Relevancy, Context Precision/Recall/Relevancy,
    Answer Similarity, Answer Correctness (일부는 ground_truth 필요).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        embeddings: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            model: LLM 모델 (gpt-4o-mini, gpt-4o, claude-3-5-sonnet-20241022 등)
            embeddings: 임베딩 모델 (text-embedding-3-small, text-embedding-3-large 등)
            api_key: API 키 (None이면 환경변수)
            **kwargs: 추가 파라미터
        """
        self.model = model
        self.embeddings = embeddings
        self.api_key = api_key
        self.kwargs = kwargs

        # Lazy loading
        self._ragas = None
        self._llm = None
        self._embeddings_model = None

    def _check_dependencies(self):
        """의존성 확인"""
        try:
            import ragas
        except ImportError:
            raise ImportError(
                "ragas is required for RAGASWrapper. Install it with: pip install ragas"
            )

        self._ragas = ragas

    def _get_llm(self):
        """LLM 모델 가져오기 (lazy loading)"""
        if self._llm is not None:
            return self._llm

        self._check_dependencies()

        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai is required for RAGAS. "
                "Install it with: pip install langchain-openai"
            )

        # OpenAI 모델 생성
        self._llm = ChatOpenAI(
            model=self.model, api_key=self.api_key if self.api_key else None, **self.kwargs
        )

        logger.info(f"RAGAS LLM loaded: {self.model}")

        return self._llm

    def _get_embeddings(self):
        """임베딩 모델 가져오기 (lazy loading)"""
        if self._embeddings_model is not None:
            return self._embeddings_model

        self._check_dependencies()

        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError:
            raise ImportError(
                "langchain-openai is required for RAGAS. "
                "Install it with: pip install langchain-openai"
            )

        # OpenAI 임베딩 생성
        self._embeddings_model = OpenAIEmbeddings(
            model=self.embeddings, api_key=self.api_key if self.api_key else None
        )

        logger.info(f"RAGAS Embeddings loaded: {self.embeddings}")

        return self._embeddings_model

    def _evaluate_single_metric(
        self,
        metric_name: str,
        data: Dict[str, List[Any]],
        metric_obj: Any,
    ) -> Dict[str, Any]:
        """
        Run RAGAS evaluate on a single-metric dataset.

        Args:
            metric_name: Key for the score in result (e.g. "faithfulness").
            data: Dict with list values: question, answer, contexts; optional ground_truth.
            metric_obj: RAGAS metric instance (e.g. faithfulness, answer_relevancy).

        Returns:
            {metric_name: float (0.0-1.0)}
        """
        self._check_dependencies()
        from datasets import Dataset
        from ragas import evaluate

        dataset = Dataset.from_dict(data)
        result = evaluate(
            dataset,
            metrics=[metric_obj],
            llm=self._get_llm(),
            embeddings=self._get_embeddings(),
        )
        score = result[metric_name]
        logger.info(f"RAGAS {metric_name.replace('_', ' ').title()}: {score:.4f}")
        return {metric_name: score}

    def _single_row_data(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, List[Any]]:
        """Build single-row dict for Dataset.from_dict (list values)."""
        data: Dict[str, List[Any]] = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        if ground_truth is not None:
            data["ground_truth"] = [ground_truth]
        return data

    def evaluate_faithfulness(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Faithfulness 평가 (Reference-free). 답변이 컨텍스트에 충실한지 평가.

        Args:
            question: 질문
            answer: 답변
            contexts: 검색된 컨텍스트 리스트

        Returns:
            {"faithfulness": float (0.0-1.0)}
        """
        from ragas.metrics import faithfulness

        data = self._single_row_data(question, answer, contexts)
        return self._evaluate_single_metric("faithfulness", data, faithfulness)

    def evaluate_answer_relevancy(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Answer Relevancy 평가 (Reference-free). 답변이 질문과 관련있는지 평가.

        Args:
            question: 질문
            answer: 답변
            contexts: 검색된 컨텍스트 리스트

        Returns:
            {"answer_relevancy": float (0.0-1.0)}
        """
        from ragas.metrics import answer_relevancy

        data = self._single_row_data(question, answer, contexts)
        return self._evaluate_single_metric("answer_relevancy", data, answer_relevancy)

    def evaluate_context_precision(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Context Precision 평가 (Requires ground truth). 검색 컨텍스트 정밀도.

        Args:
            question: 질문
            answer: 답변 (RAGAS API 호환용)
            contexts: 검색된 컨텍스트 리스트 (순서 중요)
            ground_truth: 정답

        Returns:
            {"context_precision": float (0.0-1.0)}
        """
        from ragas.metrics import context_precision

        data = self._single_row_data(question, answer, contexts, ground_truth)
        return self._evaluate_single_metric("context_precision", data, context_precision)

    def evaluate_context_recall(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Context Recall 평가 (Requires ground truth). 검색 컨텍스트 재현율.

        Args:
            question: 질문
            answer: 답변 (RAGAS API 호환용)
            contexts: 검색된 컨텍스트 리스트
            ground_truth: 정답

        Returns:
            {"context_recall": float (0.0-1.0)}
        """
        from ragas.metrics import context_recall

        data = self._single_row_data(question, answer, contexts, ground_truth)
        return self._evaluate_single_metric("context_recall", data, context_recall)

    def evaluate_context_relevancy(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Context Relevancy 평가 (Reference-free). 컨텍스트가 질문과 관련있는지 평가.

        Args:
            question: 질문
            answer: 답변 (사용 안 함)
            contexts: 검색된 컨텍스트 리스트

        Returns:
            {"context_relevancy": float (0.0-1.0)}
        """
        self._check_dependencies()
        try:
            from ragas.metrics import context_relevancy
        except ImportError:
            logger.warning(
                "context_relevancy not available in this RAGAS version. "
                "Please upgrade: pip install ragas --upgrade"
            )
            return {"context_relevancy": 0.0, "error": "Metric not available"}
        data = self._single_row_data(question, answer, contexts)
        return self._evaluate_single_metric("context_relevancy", data, context_relevancy)

    def evaluate_answer_similarity(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Answer Similarity 평가 (Requires ground truth). 답변과 정답의 의미적 유사도.

        Args:
            question: 질문 (사용 안 함)
            answer: 답변
            contexts: 컨텍스트 (사용 안 함)
            ground_truth: 정답

        Returns:
            {"answer_similarity": float (0.0-1.0)}
        """
        from ragas.metrics import answer_similarity

        data = self._single_row_data(question, answer, contexts, ground_truth)
        return self._evaluate_single_metric("answer_similarity", data, answer_similarity)

    def evaluate_answer_correctness(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Answer Correctness 평가 (Requires ground truth). 답변 정확도.

        Args:
            question: 질문 (사용 안 함)
            answer: 답변
            contexts: 컨텍스트 (사용 안 함)
            ground_truth: 정답

        Returns:
            {"answer_correctness": float (0.0-1.0)}
        """
        from ragas.metrics import answer_correctness

        data = self._single_row_data(question, answer, contexts, ground_truth)
        return self._evaluate_single_metric("answer_correctness", data, answer_correctness)

    def evaluate_dataset(
        self,
        dataset: Any,
        metrics: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        데이터셋 배치 평가.

        Args:
            dataset: pandas DataFrame 또는 HuggingFace Dataset
                (question, answer, contexts; 선택 ground_truth).
            metrics: 메트릭 리스트 (기본: ["faithfulness", "answer_relevancy"]).
            **kwargs: 추가 파라미터.

        Returns:
            평가 결과 (DataFrame 형태).
        """
        self._check_dependencies()

        from ragas import evaluate
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            answer_similarity,
            context_precision,
            context_recall,
            faithfulness,
        )

        # 메트릭 매핑
        metric_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "answer_similarity": answer_similarity,
            "answer_correctness": answer_correctness,
        }

        # 기본 메트릭 (reference-free)
        if metrics is None:
            metrics = ["faithfulness", "answer_relevancy"]

        # 메트릭 객체 리스트 생성
        metric_objects = []
        for metric_name in metrics:
            if metric_name in metric_map:
                metric_objects.append(metric_map[metric_name])
            else:
                logger.warning(f"Unknown metric: {metric_name}, skipping")

        if not metric_objects:
            raise ValueError(f"No valid metrics found. Available: {list(metric_map.keys())}")

        # Dataset 변환 (pandas → HuggingFace)
        try:
            import pandas as pd
            from datasets import Dataset

            if isinstance(dataset, pd.DataFrame):
                dataset = Dataset.from_pandas(dataset)
        except ImportError:
            pass

        # 평가
        result = evaluate(
            dataset,
            metrics=metric_objects,
            llm=self._get_llm(),
            embeddings=self._get_embeddings(),
        )

        logger.info(f"RAGAS dataset evaluation completed: {len(dataset)} samples")

        return cast(Dict[str, Any], result)

    # BaseEvaluationFramework 추상 메서드 구현

    def evaluate(self, **kwargs: Any) -> Dict[str, Any]:
        """
        평가 실행 (BaseEvaluationFramework 인터페이스).

        Args:
            **kwargs: metric (메트릭 이름), data (평가 데이터), 메트릭별 추가 파라미터.

        Returns:
            평가 결과.
        """
        metric: str = kwargs.pop("metric", "dataset")
        data: Union[Dict[str, Any], Any] = kwargs.pop("data", {})

        # 데이터셋 배치 평가
        if metric == "dataset":
            return cast(Dict[str, Any], self.evaluate_dataset(dataset=data, **kwargs))

        # 단일 평가 (data must be dict for **data)
        if not isinstance(data, dict):
            data = {}
        if metric == "faithfulness":
            return cast(Dict[str, Any], self.evaluate_faithfulness(**data, **kwargs))
        elif metric == "answer_relevancy":
            return cast(Dict[str, Any], self.evaluate_answer_relevancy(**data, **kwargs))
        elif metric == "context_precision":
            return cast(Dict[str, Any], self.evaluate_context_precision(**data, **kwargs))
        elif metric == "context_recall":
            return cast(Dict[str, Any], self.evaluate_context_recall(**data, **kwargs))
        elif metric == "context_relevancy":
            return cast(Dict[str, Any], self.evaluate_context_relevancy(**data, **kwargs))
        elif metric == "answer_similarity":
            return cast(Dict[str, Any], self.evaluate_answer_similarity(**data, **kwargs))
        elif metric == "answer_correctness":
            return cast(Dict[str, Any], self.evaluate_answer_correctness(**data, **kwargs))
        else:
            raise ValueError(
                f"Unknown metric: {metric}. Available: {list(self.list_tasks().keys())}"
            )

    def list_tasks(self) -> Dict[str, str]:
        """사용 가능한 메트릭 목록 (BaseEvaluationFramework 인터페이스)."""
        return {
            "faithfulness": "답변이 컨텍스트에 충실한지 (Reference-free)",
            "answer_relevancy": "답변이 질문과 관련있는지 (Reference-free)",
            "context_precision": "검색된 컨텍스트의 정밀도 (Requires ground truth)",
            "context_recall": "검색된 컨텍스트의 재현율 (Requires ground truth)",
            "context_relevancy": "컨텍스트가 질문과 관련있는지 (Reference-free)",
            "answer_similarity": "답변 유사도 (Requires ground truth)",
            "answer_correctness": "답변 정확도 (Requires ground truth)",
            "dataset": "데이터셋 배치 평가",
        }

    def __repr__(self) -> str:
        return f"RAGASWrapper(model={self.model}, embeddings={self.embeddings})"
