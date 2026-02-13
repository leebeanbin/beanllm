"""
BGE Reranker implementation.
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


class BGEReranker(BaseReranker):
    """
    BGE Reranker v2 (BAAI, 2024-2025)

    BAAI의 최신 재순위화 모델로 BEIR, MIRACL 등 벤치마크에서 대폭 개선되었습니다.

    모델 라인업 (추천 순):
    - BAAI/bge-reranker-v2-m3: 다국어 최강 (100+ 언어)
    - BAAI/bge-reranker-v2-gemma: LLM 백본 (높은 성능)
    - BAAI/bge-reranker-v2-minicpm-layerwise: 중국어/영어 특화
    - BAAI/bge-reranker-base: 경량 (빠른 속도)
    - BAAI/bge-reranker-large: 고성능

    Features:
    - Cross-encoder 아키텍처 (bi-encoder보다 깊은 이해)
    - 다국어 지원 (m3 모델)
    - 최대 입력 크기 확장
    - BEIR, C-MTEB 벤치마크 SOTA

    Example:
        ```python
        from beanllm.domain.retrieval import BGEReranker

        # 다국어 모델 (추천)
        reranker = BGEReranker(model="BAAI/bge-reranker-v2-m3")
        results = reranker.rerank(
            query="What is machine learning?",
            documents=[
                "ML is a subset of AI...",
                "Python is a programming language...",
                "Deep learning uses neural networks..."
            ],
            top_k=2
        )

        for result in results:
            print(f"Score: {result.score:.4f}, Text: {result.text[:50]}")
        # Score: 0.9823, Text: ML is a subset of AI...
        # Score: 0.7654, Text: Deep learning uses neural networks...
        ```
    """

    def __init__(
        self,
        model: str = "BAAI/bge-reranker-v2-m3",
        use_gpu: bool = True,
        batch_size: int = 32,
        max_length: int = 512,
        **kwargs,
    ):
        """
        Args:
            model: BGE Reranker 모델
                - BAAI/bge-reranker-v2-m3: 다국어 (기본값, 추천)
                - BAAI/bge-reranker-v2-gemma: LLM 백본
                - BAAI/bge-reranker-v2-minicpm-layerwise: 중국어/영어
                - BAAI/bge-reranker-base: 경량
                - BAAI/bge-reranker-large: 고성능
            use_gpu: GPU 사용 여부 (기본: True)
            batch_size: 배치 크기 (기본: 32)
            max_length: 최대 토큰 길이 (기본: 512)
            **kwargs: 추가 파라미터
        """
        self.model_name = model
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.max_length = max_length
        self.kwargs = kwargs

        # Lazy loading
        self._model = None
        self._tokenizer = None
        self._device = None

    def _load_model(self):
        """모델 로딩 (lazy loading)"""
        if self._model is not None:
            return

        try:
            import torch  # type: ignore[import-untyped]
            from transformers import (  # type: ignore[import-untyped]
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError:
            raise ImportError(
                "transformers and torch required for BGEReranker. "
                "Install: pip install transformers torch"
            )

        # Device 설정
        if self.use_gpu and torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        logger.info(f"Loading BGE Reranker: {self.model_name} on {self._device}")

        # 모델 및 토크나이저 로드
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self._device)
        self._model.eval()

        logger.info(f"BGE Reranker loaded: {self.model_name}")

    def rerank(
        self, query: str, documents: List[str], top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        문서를 재순위화

        Args:
            query: 검색 쿼리
            documents: 재순위화할 문서 리스트
            top_k: 반환할 상위 k개 (None이면 전체)

        Returns:
            재순위화된 결과 (점수 내림차순)
        """
        # 모델 로드
        self._load_model()
        assert self._tokenizer is not None
        assert self._model is not None

        try:
            import torch

            # 쿼리-문서 페어 생성
            pairs = [[query, doc] for doc in documents]

            # 배치 처리
            all_scores = []

            for i in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[i : i + self.batch_size]

                # Tokenization
                inputs = self._tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self._device)

                # Forward pass
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    scores = outputs.logits.squeeze(-1)

                    # Sigmoid (확률로 변환)
                    if scores.dim() == 0:
                        scores = scores.unsqueeze(0)

                    # CPU로 이동
                    scores = scores.cpu().tolist()
                    if isinstance(scores, float):
                        scores = [scores]

                    all_scores.extend(scores)

            # RerankResult 생성
            results = [
                RerankResult(text=doc, score=float(score), index=idx)
                for idx, (doc, score) in enumerate(zip(documents, all_scores))
            ]

            # 점수로 정렬
            results.sort(key=lambda x: x.score, reverse=True)

            # Top-k 선택
            if top_k is not None:
                results = results[:top_k]

            logger.info(f"Reranked {len(documents)} documents, top score: {results[0].score:.4f}")

            return results

        except Exception as e:
            logger.error(f"BGE Reranker failed: {e}")
            raise
