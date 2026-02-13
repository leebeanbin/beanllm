"""
RAG-specific metrics: AnswerRelevance, ContextPrecision, Faithfulness, ContextRecall.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING, Callable, List, Optional

from beanllm.domain.evaluation.base_metric import BaseMetric
from beanllm.domain.evaluation.enums import MetricType
from beanllm.domain.evaluation.metrics.llm_judge import LLMJudgeMetric
from beanllm.domain.evaluation.results import EvaluationResult

if TYPE_CHECKING:
    from beanllm.domain.evaluation.protocols import LLMClientProtocol


class AnswerRelevanceMetric(BaseMetric):
    """
    Answer Relevance (RAG)

    생성된 답변이 질문과 얼마나 관련있는지 평가
    """

    def __init__(self, client: Optional["LLMClientProtocol"] = None) -> None:
        super().__init__("answer_relevance", MetricType.RAG)
        self.client = client

    def compute(self, prediction: str, reference: str, **kwargs) -> EvaluationResult:
        """
        Args:
            prediction: 생성된 답변
            reference: 원래 질문
        """
        question = reference
        answer = prediction

        # LLM-as-judge 사용
        judge = LLMJudgeMetric(
            client=self.client,
            criterion="relevance",
            use_reference=True,
        )

        result = judge.compute(answer, question)
        return replace(result, metric_name=self.name)


class ContextPrecisionMetric(BaseMetric):
    """
    Context Precision (RAG)

    검색된 컨텍스트가 질문에 대한 답변과 얼마나 관련있는지 평가
    """

    def __init__(self) -> None:
        super().__init__("context_precision", MetricType.RAG)

    def compute(
        self,
        prediction: str,
        reference: str,
        contexts: Optional[List[str]] = None,
        **kwargs,
    ) -> EvaluationResult:
        """
        Args:
            prediction: 생성된 답변
            reference: 원래 질문
            contexts: 검색된 컨텍스트 리스트
        """
        if not contexts:
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                metadata={"error": "No contexts provided"},
            )

        # 각 컨텍스트가 답변 생성에 사용되었는지 확인
        # 간단한 휴리스틱: 답변에 컨텍스트의 단어가 포함되어 있는지
        answer_tokens = set(prediction.lower().split())
        relevant_count = 0

        for ctx in contexts:
            ctx_tokens = set(ctx.lower().split())
            overlap = len(answer_tokens & ctx_tokens)
            # 충분한 오버랩이 있으면 관련있다고 판단
            if overlap >= min(3, len(ctx_tokens) * 0.3):
                relevant_count += 1

        precision = relevant_count / len(contexts)

        return EvaluationResult(
            metric_name=self.name,
            score=precision,
            metadata={
                "total_contexts": len(contexts),
                "relevant_contexts": relevant_count,
            },
        )


class FaithfulnessMetric(BaseMetric):
    """
    Faithfulness (RAG)

    생성된 답변이 제공된 컨텍스트에 충실한지 평가 (환각 검출)
    """

    def __init__(self, client: Optional["LLMClientProtocol"] = None) -> None:
        super().__init__("faithfulness", MetricType.RAG)
        self.client = client

    def _get_client(self):
        """클라이언트 반환 (생성자에서 주입 필수)"""
        if self.client is None:
            raise RuntimeError(
                "LLM client not available. "
                "Please provide a client via constructor: "
                "FaithfulnessMetric(client=your_client)"
            )
        return self.client

    def compute(
        self,
        prediction: str,
        reference: str,
        contexts: Optional[List[str]] = None,
        **kwargs,
    ) -> EvaluationResult:
        """
        Args:
            prediction: 생성된 답변
            reference: (사용안함)
            contexts: 검색된 컨텍스트 리스트
        """
        if not contexts:
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                metadata={"error": "No contexts provided"},
            )

        client = self._get_client()

        # Faithfulness 평가 프롬프트
        context_text = "\n\n".join(contexts)
        prompt = (
            f"Given the following context:\n{context_text}\n\n"
            f"Evaluate if the following statement is faithful to the context "
            f"(i.e., all information is supported by the context):\n{prediction}\n\n"
            f"Respond with a score from 0 to 1, where 1 means fully faithful.\n"
            f"Format: SCORE: <number>"
        )

        response = client.chat([{"role": "user", "content": prompt}])
        output = response.content

        # 점수 추출
        score_match = re.search(r"SCORE:\s*([\d.]+)", output)
        score = float(score_match.group(1)) if score_match else 0.5

        return EvaluationResult(
            metric_name=self.name,
            score=score,
            metadata={"contexts_count": len(contexts)},
        )


class ContextRecallMetric(BaseMetric):
    """
    Context Recall (RAG)

    모든 관련 문서가 검색되었는지 평가
    검색된 컨텍스트가 ground truth 컨텍스트를 얼마나 포함하는지 측정
    """

    def __init__(
        self,
        embedding_function: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            embedding_function: 임베딩 함수 (선택적, 없으면 토큰 기반 매칭 사용)
        """
        super().__init__("context_recall", MetricType.RAG)
        self.embedding_function = embedding_function

    def compute(
        self,
        prediction: str,
        reference: str,
        contexts: Optional[List[str]] = None,
        ground_truth_contexts: Optional[List[str]] = None,
        **kwargs,
    ) -> EvaluationResult:
        """
        Args:
            prediction: 생성된 답변 (사용 안 함)
            reference: 질문 (사용 안 함)
            contexts: 검색된 컨텍스트 리스트
            ground_truth_contexts: 실제 관련 컨텍스트 리스트 (필수)
        """
        if not contexts:
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                metadata={"error": "No contexts provided"},
            )

        if not ground_truth_contexts:
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                metadata={"error": "No ground truth contexts provided"},
            )

        # 임베딩 기반 유사도 계산 (가능한 경우)
        if self.embedding_function:
            recall = self._compute_recall_with_embeddings(contexts, ground_truth_contexts)
        else:
            # 토큰 기반 매칭 (간단한 방법)
            recall = self._compute_recall_with_tokens(contexts, ground_truth_contexts)

        return EvaluationResult(
            metric_name=self.name,
            score=recall,
            metadata={
                "retrieved_count": len(contexts),
                "ground_truth_count": len(ground_truth_contexts),
            },
        )

    def _compute_recall_with_embeddings(
        self,
        contexts: List[str],
        ground_truth_contexts: List[str],
    ) -> float:
        """임베딩 기반 재현율 계산"""
        if self.embedding_function is None:
            return self._compute_recall_with_tokens(contexts, ground_truth_contexts)
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity

            # 임베딩 생성
            retrieved_embeddings = np.array(self.embedding_function(contexts))
            gt_embeddings = np.array(self.embedding_function(ground_truth_contexts))

            # 유사도 행렬 계산
            similarity_matrix = cosine_similarity(gt_embeddings, retrieved_embeddings)

            # 각 ground truth에 대해 가장 유사한 retrieved context 찾기
            max_similarities = similarity_matrix.max(axis=1)

            # 임계값 이상인 것만 관련있다고 판단 (0.7 이상)
            threshold = 0.7
            relevant_count = sum(1 for sim in max_similarities if sim >= threshold)

            recall = relevant_count / len(ground_truth_contexts) if ground_truth_contexts else 0.0

            return recall

        except ImportError:
            # scikit-learn이 없으면 토큰 기반으로 폴백
            return self._compute_recall_with_tokens(contexts, ground_truth_contexts)

    def _compute_recall_with_tokens(
        self,
        contexts: List[str],
        ground_truth_contexts: List[str],
    ) -> float:
        """토큰 기반 재현율 계산"""
        # 각 ground truth 컨텍스트가 retrieved 컨텍스트에 포함되어 있는지 확인
        relevant_count = 0

        for gt_ctx in ground_truth_contexts:
            gt_tokens = set(gt_ctx.lower().split())

            # retrieved 컨텍스트 중 하나라도 충분한 오버랩이 있으면 관련있다고 판단
            found = False
            for ctx in contexts:
                ctx_tokens = set(ctx.lower().split())
                overlap = len(gt_tokens & ctx_tokens)
                # 30% 이상 오버랩이 있으면 관련있다고 판단
                if overlap >= len(gt_tokens) * 0.3:
                    found = True
                    break

            if found:
                relevant_count += 1

        recall = relevant_count / len(ground_truth_contexts) if ground_truth_contexts else 0.0

        return recall
