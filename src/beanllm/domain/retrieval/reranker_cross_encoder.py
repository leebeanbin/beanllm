"""
Cross-Encoder Reranker implementation.
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


class CrossEncoderReranker(BaseReranker):
    """
    범용 Cross-Encoder Reranker

    HuggingFace의 모든 cross-encoder 모델을 지원합니다.

    추천 모델:
    - cross-encoder/ms-marco-MiniLM-L-6-v2: 경량 (빠름)
    - cross-encoder/ms-marco-MiniLM-L-12-v2: 균형
    - cross-encoder/ms-marco-electra-base: 고성능

    Example:
        ```python
        from beanllm.domain.retrieval import CrossEncoderReranker

        reranker = CrossEncoderReranker(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        results = reranker.rerank(
            query="What is Python?",
            documents=["Python is a language...", "Java is..."],
            top_k=1
        )
        ```
    """

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_gpu: bool = True,
        batch_size: int = 32,
        **kwargs,
    ):
        """
        Args:
            model: HuggingFace cross-encoder 모델
            use_gpu: GPU 사용 여부
            batch_size: 배치 크기
            **kwargs: 추가 파라미터
        """
        self.model_name = model
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.kwargs = kwargs

        # Lazy loading
        self._model = None

    def _load_model(self):
        """모델 로딩"""
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers required. Install: pip install sentence-transformers"
            )

        device = "cuda" if self.use_gpu else "cpu"
        self._model = CrossEncoder(self.model_name, device=device)

        logger.info(f"CrossEncoder loaded: {self.model_name} on {device}")

    def rerank(
        self, query: str, documents: List[str], top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """문서를 재순위화"""
        self._load_model()
        assert self._model is not None

        try:
            # 쿼리-문서 페어
            pairs = [[query, doc] for doc in documents]

            # 점수 계산
            scores = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

            # RerankResult 생성
            results = [
                RerankResult(text=doc, score=float(score), index=idx)
                for idx, (doc, score) in enumerate(zip(documents, scores))
            ]

            # 정렬
            results.sort(key=lambda x: x.score, reverse=True)

            # Top-k
            if top_k is not None:
                results = results[:top_k]

            logger.info(
                f"Reranked {len(documents)} documents with CrossEncoder, "
                f"top score: {results[0].score:.4f}"
            )

            return results

        except Exception as e:
            logger.error(f"CrossEncoder Reranker failed: {e}")
            raise
