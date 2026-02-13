"""
Position Engineering Reranker implementation.
"""

from __future__ import annotations

from typing import List, Optional

from beanllm.domain.retrieval.base import BaseReranker
from beanllm.domain.retrieval.types import RerankResult

try:
    from beanllm.utils.logging import get_logger
except ImportError:
    import logging

    def get_logger(name: str) -> logging.Logger:  # type: ignore[misc]
        return logging.getLogger(name)


logger = get_logger(__name__)


class PositionEngineeringReranker(BaseReranker):
    """
    Position Engineering Reranker (2024-2025).
    전략: head, tail, head_tail, side.
    """

    def __init__(
        self,
        base_reranker: Optional[BaseReranker] = None,
        strategy: str = "head",
        **kwargs,
    ):
        self.base_reranker = base_reranker
        self.strategy = strategy.lower()
        self.kwargs = kwargs
        valid_strategies = ["head", "tail", "head_tail", "side"]
        if self.strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {self.strategy}. Available: {valid_strategies}")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        scores: Optional[List[float]] = None,
    ) -> List[RerankResult]:
        if self.base_reranker is not None:
            ranked_results = self.base_reranker.rerank(
                query=query, documents=documents, top_k=top_k
            )
        elif scores is not None:
            if len(scores) != len(documents):
                raise ValueError("scores와 documents의 길이가 일치하지 않습니다.")
            ranked_results = [
                RerankResult(text=doc, score=float(score), index=idx)
                for idx, (doc, score) in enumerate(zip(documents, scores))
            ]
            ranked_results.sort(key=lambda x: x.score, reverse=True)
            if top_k is not None:
                ranked_results = ranked_results[:top_k]
        else:
            ranked_results = [
                RerankResult(text=doc, score=1.0 / (idx + 1), index=idx)
                for idx, doc in enumerate(documents)
            ]
            if top_k is not None:
                ranked_results = ranked_results[:top_k]
        reordered = self._apply_position_engineering(ranked_results)
        logger.info(
            f"Position Engineering applied: strategy={self.strategy}, count={len(reordered)}"
        )
        return reordered

    def _apply_position_engineering(self, results: List[RerankResult]) -> List[RerankResult]:
        n = len(results)
        if n == 0:
            return results
        if self.strategy == "head":
            return results
        if self.strategy == "tail":
            return results[::-1]
        if self.strategy == "head_tail":
            left = [r for i, r in enumerate(results) if i % 2 == 0]
            right = [r for i, r in enumerate(results) if i % 2 == 1]
            return left + right[::-1]
        if self.strategy == "side":
            reordered = [None] * n
            front_idx, back_idx = 0, n - 1
            for i, result in enumerate(results):
                if i % 2 == 0:
                    reordered[front_idx] = result
                    front_idx += 1
                else:
                    reordered[back_idx] = result
                    back_idx -= 1
            return reordered
        return results

    def __repr__(self) -> str:
        base_name = self.base_reranker.__class__.__name__ if self.base_reranker else "None"
        return f"PositionEngineeringReranker(base={base_name}, strategy={self.strategy})"
