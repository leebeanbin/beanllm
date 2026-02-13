"""
Semantic similarity metric (embedding-based).
"""

from __future__ import annotations

import math
from typing import List

from beanllm.domain.evaluation.base_metric import BaseMetric
from beanllm.domain.evaluation.enums import MetricType
from beanllm.domain.evaluation.results import EvaluationResult


class SemanticSimilarityMetric(BaseMetric):
    """
    의미론적 유사도 (Embedding 기반)

    두 텍스트의 의미적 유사성을 임베딩 벡터의 코사인 유사도로 측정
    """

    def __init__(self, embedding_model=None) -> None:
        super().__init__("semantic_similarity", MetricType.SEMANTIC)
        self.embedding_model = embedding_model

    def _get_embedding_model(self):
        """임베딩 모델 lazy loading"""
        if self.embedding_model is None:
            try:
                from beanllm.domain.embeddings import OpenAIEmbedding

                self.embedding_model = OpenAIEmbedding()
            except Exception:
                raise RuntimeError(
                    "Embedding model not available. Please provide an embedding model."
                )
        return self.embedding_model

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """코사인 유사도"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def compute(self, prediction: str, reference: str, **kwargs) -> EvaluationResult:
        model = self._get_embedding_model()

        # 임베딩 생성
        pred_emb = model.embed(prediction)
        ref_emb = model.embed(reference)

        # 코사인 유사도
        similarity = self._cosine_similarity(pred_emb, ref_emb)

        return EvaluationResult(
            metric_name=self.name,
            score=similarity,
            metadata={"embedding_model": str(type(model).__name__)},
        )
