"""
Cohere Reranker implementation.
"""

from __future__ import annotations

import os
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


class CohereReranker(BaseReranker):
    """
    Cohere Rerank (2024-2025)

    Cohere의 최신 재순위화 모델로 100개 이상의 언어를 지원합니다.

    모델:
    - rerank-3-nimble: 프로덕션용 고속 (기본값)
    - rerank-4: 32K context (2024년 12월 최신)

    Features:
    - 100+ 언어 지원
    - 32K context window (rerank-4)
    - Self-learning (rerank-4)
    - 프로덕션급 속도 (nimble)
    """

    def __init__(
        self,
        model: str = "rerank-3-nimble",
        api_key: Optional[str] = None,
        max_chunks_per_doc: Optional[int] = None,
        **kwargs,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.max_chunks_per_doc = max_chunks_per_doc
        self.kwargs = kwargs

        if not self.api_key:
            raise ValueError("COHERE_API_KEY not found in environment variables")

        self._client = None

    def _get_client(self):
        """Cohere 클라이언트 가져오기 (lazy)"""
        if self._client is not None:
            return self._client

        try:
            import cohere
        except ImportError:
            raise ImportError("cohere required for CohereReranker. Install: pip install cohere")

        self._client = cohere.Client(api_key=self.api_key)
        return self._client

    def rerank(
        self, query: str, documents: List[str], top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """문서를 재순위화"""
        client = self._get_client()

        try:
            response = client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=top_k if top_k else len(documents),
                max_chunks_per_doc=self.max_chunks_per_doc,
                **self.kwargs,
            )

            results = [
                RerankResult(
                    text=documents[result.index],
                    score=result.relevance_score,
                    index=result.index,
                )
                for result in response.results
            ]

            logger.info(
                f"Reranked {len(documents)} documents with Cohere {self.model}, "
                f"top score: {results[0].score:.4f}"
            )

            return results

        except Exception as e:
            logger.error(f"Cohere Reranker failed: {e}")
            raise
