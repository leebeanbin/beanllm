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
from typing import Any, Dict, List, Optional, Union

from .base_framework import BaseEvaluationFramework

try:
    from beanllm.utils.logger import get_logger
except ImportError:

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


class RAGASWrapper(BaseEvaluationFramework):
    """
    RAGAS 통합 래퍼

    RAGAS의 주요 메트릭을 beanLLM 스타일로 사용할 수 있게 합니다.

    지원 메트릭 (2024-2025):
    1. Component Metrics (개별 컴포넌트 평가):
       - Faithfulness: 답변이 컨텍스트에 충실한지 (Hallucination 방지)
       - Answer Relevancy: 답변이 질문과 관련있는지
       - Context Precision: 검색된 컨텍스트의 정밀도
       - Context Recall: 검색된 컨텍스트의 재현율
       - Context Relevancy: 컨텍스트가 질문과 관련있는지
       - Context Entity Recall: 엔티티 기반 재현율

    2. End-to-End Metrics (전체 시스템 평가):
       - Answer Similarity: 답변 유사도 (reference 필요)
       - Answer Correctness: 답변 정확도 (reference 필요)

    Example:
        ```python
        from beanllm.domain.evaluation import RAGASWrapper

        # 기본 사용
        evaluator = RAGASWrapper(
            model="gpt-4o-mini",
            embeddings="text-embedding-3-small"
        )

        # Faithfulness 평가 (Reference-free)
        result = evaluator.evaluate_faithfulness(
            question="What is Paris?",
            answer="Paris is the capital of France.",
            contexts=["Paris is the capital and largest city of France."]
        )
        print(result)  # {"faithfulness": 1.0}

        # Answer Relevancy 평가 (Reference-free)
        result = evaluator.evaluate_answer_relevancy(
            question="What is the capital of France?",
            answer="Paris is the capital of France.",
            contexts=["Paris is a major European city."]
        )
        print(result)  # {"answer_relevancy": 0.95}

        # 배치 평가 (DataFrame 사용)
        import pandas as pd

        data = {
            "question": ["Q1", "Q2"],
            "answer": ["A1", "A2"],
            "contexts": [["C1"], ["C2"]],
            "ground_truth": ["GT1", "GT2"]  # Optional
        }
        df = pd.DataFrame(data)

        results = evaluator.evaluate_dataset(
            dataset=df,
            metrics=["faithfulness", "answer_relevancy"]
        )
        print(results)  # DataFrame with scores
        ```
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
                "ragas is required for RAGASWrapper. " "Install it with: pip install ragas"
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

    def evaluate_faithfulness(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Faithfulness 평가 (Reference-free)

        답변이 제공된 컨텍스트에 충실한지 평가합니다.
        LLM을 사용하여 답변의 각 문장이 컨텍스트에서 지원되는지 확인합니다.

        Args:
            question: 질문
            answer: 답변
            contexts: 검색된 컨텍스트 리스트
            **kwargs: 추가 파라미터

        Returns:
            {"faithfulness": float (0.0-1.0)}

        Example:
            ```python
            result = evaluator.evaluate_faithfulness(
                question="What is Paris?",
                answer="Paris is the capital of France and home to the Eiffel Tower.",
                contexts=["Paris is the capital of France."]
            )
            # {"faithfulness": 0.5}  # 절반만 지원됨 (Eiffel Tower는 컨텍스트에 없음)
            ```
        """
        self._check_dependencies()

        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness

        # Dataset 생성
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        dataset = Dataset.from_dict(data)

        # 평가
        result = evaluate(
            dataset,
            metrics=[faithfulness],
            llm=self._get_llm(),
            embeddings=self._get_embeddings(),
        )

        score = result["faithfulness"]

        logger.info(f"RAGAS Faithfulness: {score:.4f}")

        return {"faithfulness": score}

    def evaluate_answer_relevancy(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Answer Relevancy 평가 (Reference-free)

        답변이 질문과 얼마나 관련있는지 평가합니다.
        LLM을 사용하여 답변에서 역으로 질문을 생성하고,
        원래 질문과의 유사도를 측정합니다.

        Args:
            question: 질문
            answer: 답변
            contexts: 검색된 컨텍스트 리스트
            **kwargs: 추가 파라미터

        Returns:
            {"answer_relevancy": float (0.0-1.0)}

        Example:
            ```python
            result = evaluator.evaluate_answer_relevancy(
                question="What is the capital of France?",
                answer="Paris is the capital of France.",
                contexts=["Paris is a city in France."]
            )
            # {"answer_relevancy": 0.95}
            ```
        """
        self._check_dependencies()

        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy

        # Dataset 생성
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        dataset = Dataset.from_dict(data)

        # 평가
        result = evaluate(
            dataset,
            metrics=[answer_relevancy],
            llm=self._get_llm(),
            embeddings=self._get_embeddings(),
        )

        score = result["answer_relevancy"]

        logger.info(f"RAGAS Answer Relevancy: {score:.4f}")

        return {"answer_relevancy": score}

    def evaluate_context_precision(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Context Precision 평가 (Requires ground truth)

        검색된 컨텍스트의 정밀도를 평가합니다.
        관련있는 컨텍스트가 상위에 랭크되어 있는지 확인합니다.

        Args:
            question: 질문
            answer: 답변 (사용 안 함, RAGAS API 호환용)
            contexts: 검색된 컨텍스트 리스트 (순서 중요)
            ground_truth: 정답
            **kwargs: 추가 파라미터

        Returns:
            {"context_precision": float (0.0-1.0)}

        Example:
            ```python
            result = evaluator.evaluate_context_precision(
                question="What is the capital of France?",
                answer="Paris",
                contexts=["Paris is the capital.", "France is in Europe."],
                ground_truth="Paris is the capital of France."
            )
            # {"context_precision": 1.0}  # 관련 컨텍스트가 첫 번째
            ```
        """
        self._check_dependencies()

        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import context_precision

        # Dataset 생성
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth],
        }
        dataset = Dataset.from_dict(data)

        # 평가
        result = evaluate(
            dataset,
            metrics=[context_precision],
            llm=self._get_llm(),
            embeddings=self._get_embeddings(),
        )

        score = result["context_precision"]

        logger.info(f"RAGAS Context Precision: {score:.4f}")

        return {"context_precision": score}

    def evaluate_context_recall(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Context Recall 평가 (Requires ground truth)

        검색된 컨텍스트의 재현율을 평가합니다.
        ground truth를 생성하는 데 필요한 모든 정보가 검색되었는지 확인합니다.

        Args:
            question: 질문
            answer: 답변 (사용 안 함, RAGAS API 호환용)
            contexts: 검색된 컨텍스트 리스트
            ground_truth: 정답
            **kwargs: 추가 파라미터

        Returns:
            {"context_recall": float (0.0-1.0)}

        Example:
            ```python
            result = evaluator.evaluate_context_recall(
                question="What is the capital of France?",
                answer="Paris",
                contexts=["Paris is the capital of France."],
                ground_truth="Paris is the capital of France."
            )
            # {"context_recall": 1.0}  # 모든 정보가 검색됨
            ```
        """
        self._check_dependencies()

        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import context_recall

        # Dataset 생성
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth],
        }
        dataset = Dataset.from_dict(data)

        # 평가
        result = evaluate(
            dataset,
            metrics=[context_recall],
            llm=self._get_llm(),
            embeddings=self._get_embeddings(),
        )

        score = result["context_recall"]

        logger.info(f"RAGAS Context Recall: {score:.4f}")

        return {"context_recall": score}

    def evaluate_context_relevancy(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Context Relevancy 평가 (Reference-free)

        검색된 컨텍스트가 질문과 얼마나 관련있는지 평가합니다.

        Args:
            question: 질문
            answer: 답변 (사용 안 함)
            contexts: 검색된 컨텍스트 리스트
            **kwargs: 추가 파라미터

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

        from datasets import Dataset
        from ragas import evaluate

        # Dataset 생성
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        dataset = Dataset.from_dict(data)

        # 평가
        result = evaluate(
            dataset,
            metrics=[context_relevancy],
            llm=self._get_llm(),
            embeddings=self._get_embeddings(),
        )

        score = result["context_relevancy"]

        logger.info(f"RAGAS Context Relevancy: {score:.4f}")

        return {"context_relevancy": score}

    def evaluate_answer_similarity(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Answer Similarity 평가 (Requires ground truth)

        생성된 답변과 정답의 의미적 유사도를 평가합니다.

        Args:
            question: 질문 (사용 안 함)
            answer: 답변
            contexts: 컨텍스트 (사용 안 함)
            ground_truth: 정답
            **kwargs: 추가 파라미터

        Returns:
            {"answer_similarity": float (0.0-1.0)}
        """
        self._check_dependencies()

        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_similarity

        # Dataset 생성
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth],
        }
        dataset = Dataset.from_dict(data)

        # 평가
        result = evaluate(
            dataset,
            metrics=[answer_similarity],
            llm=self._get_llm(),
            embeddings=self._get_embeddings(),
        )

        score = result["answer_similarity"]

        logger.info(f"RAGAS Answer Similarity: {score:.4f}")

        return {"answer_similarity": score}

    def evaluate_answer_correctness(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Answer Correctness 평가 (Requires ground truth)

        답변의 정확도를 평가합니다.
        Factual similarity와 Semantic similarity를 모두 고려합니다.

        Args:
            question: 질문 (사용 안 함)
            answer: 답변
            contexts: 컨텍스트 (사용 안 함)
            ground_truth: 정답
            **kwargs: 추가 파라미터

        Returns:
            {"answer_correctness": float (0.0-1.0)}
        """
        self._check_dependencies()

        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_correctness

        # Dataset 생성
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth],
        }
        dataset = Dataset.from_dict(data)

        # 평가
        result = evaluate(
            dataset,
            metrics=[answer_correctness],
            llm=self._get_llm(),
            embeddings=self._get_embeddings(),
        )

        score = result["answer_correctness"]

        logger.info(f"RAGAS Answer Correctness: {score:.4f}")

        return {"answer_correctness": score}

    def evaluate_dataset(
        self,
        dataset,
        metrics: Optional[List[str]] = None,
        **kwargs,
    ) -> Any:
        """
        데이터셋 배치 평가

        Args:
            dataset: pandas DataFrame 또는 HuggingFace Dataset
                필수 컬럼: question, answer, contexts
                선택 컬럼: ground_truth (일부 메트릭 필요)
            metrics: 평가할 메트릭 리스트
                - Reference-free: ["faithfulness", "answer_relevancy"]
                - With reference: ["context_precision", "context_recall",
                                   "answer_similarity", "answer_correctness"]
            **kwargs: 추가 파라미터

        Returns:
            평가 결과 (DataFrame 형태)

        Example:
            ```python
            import pandas as pd

            data = {
                "question": ["What is AI?", "What is ML?"],
                "answer": ["AI is...", "ML is..."],
                "contexts": [["Context 1"], ["Context 2"]],
                "ground_truth": ["GT 1", "GT 2"]  # Optional
            }
            df = pd.DataFrame(data)

            # Reference-free 평가
            results = evaluator.evaluate_dataset(
                dataset=df,
                metrics=["faithfulness", "answer_relevancy"]
            )

            # Reference 필요한 평가
            results = evaluator.evaluate_dataset(
                dataset=df,
                metrics=["context_precision", "answer_correctness"]
            )
            ```
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

        return result

    # BaseEvaluationFramework 추상 메서드 구현

    def evaluate(
        self, metric: str, data: Union[Dict[str, Any], Any], **kwargs
    ) -> Dict[str, Any]:
        """
        평가 실행 (BaseEvaluationFramework 인터페이스)

        Args:
            metric: 메트릭 이름 (faithfulness, answer_relevancy 등)
            data: 평가 데이터 (dict 또는 DataFrame)
            **kwargs: 메트릭별 추가 파라미터

        Returns:
            평가 결과

        Example:
            ```python
            # 단일 평가
            result = evaluator.evaluate(
                metric="faithfulness",
                data={
                    "question": "What is AI?",
                    "answer": "AI is...",
                    "contexts": ["Context 1"]
                }
            )

            # 배치 평가
            result = evaluator.evaluate(
                metric="dataset",
                data=df,  # pandas DataFrame
                metrics=["faithfulness", "answer_relevancy"]
            )
            ```
        """
        # 데이터셋 배치 평가
        if metric == "dataset":
            return self.evaluate_dataset(dataset=data, **kwargs)

        # 단일 평가
        if metric == "faithfulness":
            return self.evaluate_faithfulness(**data, **kwargs)
        elif metric == "answer_relevancy":
            return self.evaluate_answer_relevancy(**data, **kwargs)
        elif metric == "context_precision":
            return self.evaluate_context_precision(**data, **kwargs)
        elif metric == "context_recall":
            return self.evaluate_context_recall(**data, **kwargs)
        elif metric == "context_relevancy":
            return self.evaluate_context_relevancy(**data, **kwargs)
        elif metric == "answer_similarity":
            return self.evaluate_answer_similarity(**data, **kwargs)
        elif metric == "answer_correctness":
            return self.evaluate_answer_correctness(**data, **kwargs)
        else:
            raise ValueError(
                f"Unknown metric: {metric}. " f"Available: {list(self.list_tasks().keys())}"
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
            #     "faithfulness": "답변이 컨텍스트에 충실한지 (Reference-free)",
            #     "answer_relevancy": "답변이 질문과 관련있는지 (Reference-free)",
            #     ...
            # }
            ```
        """
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
