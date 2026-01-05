"""
Hybrid Search - BM25 + Dense Embeddings (2024-2025)

Sparse (BM25) + Dense (Embeddings) 검색을 결합하여 최적의 검색 성능을 제공합니다.

Hybrid Search 장점:
- BM25: 키워드 매칭, 고유 용어에 강함 (예: 제품명, 고유명사)
- Dense: 의미 기반 매칭, 동의어와 paraphrasing에 강함
- 결합: 두 방식의 장점을 모두 활용하여 30-50% 검색 품질 향상

Fusion Methods:
- RRF (Reciprocal Rank Fusion): 순위 기반 결합 (기본값, 추천)
- Weighted Sum: 점수 가중 평균
- Distribution-Based: 점수 분포 정규화 후 결합

Requirements:
    pip install rank-bm25

References:
    - Cormack et al. (2009): "Reciprocal Rank Fusion"
    - Robertson & Zaragoza (2009): "The Probabilistic Relevance Framework: BM25"
"""

import heapq
import logging
from typing import Callable, Dict, List, Optional, Tuple

from .types import SearchResult

try:
    from beanllm.utils.logger import get_logger
except ImportError:

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


class HybridRetriever:
    """
    Hybrid Retrieval (BM25 + Dense Embeddings)

    BM25와 Dense Embeddings를 결합하여 검색 품질을 향상시킵니다.

    Features:
    - BM25 (Sparse): 키워드 매칭, 통계 기반
    - Dense (Embeddings): 의미 기반 매칭
    - Fusion: RRF, Weighted Sum, Distribution-Based
    - 30-50% 검색 품질 향상

    Example:
        ```python
        from beanllm.domain.retrieval import HybridRetriever
        from beanllm.domain.embeddings import OpenAIEmbedding

        # Hybrid Retriever 생성
        embedding_model = OpenAIEmbedding(model="text-embedding-3-small")
        retriever = HybridRetriever(
            documents=["Doc 1", "Doc 2", "Doc 3"],
            embedding_function=embedding_model.embed,
            fusion_method="rrf",
            bm25_weight=0.5,
            dense_weight=0.5
        )

        # 검색
        results = retriever.search(
            query="What is machine learning?",
            top_k=3
        )

        for result in results:
            print(f"Score: {result.score:.4f}, Text: {result.text[:50]}")
        ```
    """

    def __init__(
        self,
        documents: List[str],
        embedding_function: Callable[[str], List[float]],
        fusion_method: str = "rrf",
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        rrf_k: int = 60,
        **kwargs,
    ):
        """
        Args:
            documents: 검색 대상 문서 리스트
            embedding_function: 임베딩 함수 (str -> List[float])
            fusion_method: Fusion 방법
                - "rrf": Reciprocal Rank Fusion (기본값, 추천)
                - "weighted_sum": 가중 평균
                - "distribution_based": 분포 정규화 후 결합
            bm25_weight: BM25 가중치 (fusion_method="weighted_sum"일 때)
            dense_weight: Dense 가중치 (fusion_method="weighted_sum"일 때)
            bm25_k1: BM25 k1 파라미터 (기본: 1.5)
            bm25_b: BM25 b 파라미터 (기본: 0.75)
            rrf_k: RRF k 파라미터 (기본: 60)
            **kwargs: 추가 파라미터
        """
        self.documents = documents
        self.embedding_function = embedding_function
        self.fusion_method = fusion_method.lower()
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.rrf_k = rrf_k
        self.kwargs = kwargs

        # Fusion method 검증
        valid_methods = ["rrf", "weighted_sum", "distribution_based"]
        if self.fusion_method not in valid_methods:
            raise ValueError(
                f"Invalid fusion_method: {self.fusion_method}. "
                f"Available: {valid_methods}"
            )

        # BM25 초기화
        self._bm25 = None
        self._init_bm25()

        # Dense 임베딩 초기화
        self._document_embeddings = None
        self._init_embeddings()

        logger.info(
            f"HybridRetriever initialized: {len(documents)} documents, "
            f"fusion={self.fusion_method}"
        )

    def _init_bm25(self):
        """BM25 인덱스 초기화"""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank-bm25 is required for HybridRetriever. "
                "Install it with: pip install rank-bm25"
            )

        # 문서 토큰화 (간단한 공백 기반)
        tokenized_docs = [doc.lower().split() for doc in self.documents]

        # BM25 인덱스 생성
        self._bm25 = BM25Okapi(
            tokenized_docs,
            k1=self.bm25_k1,
            b=self.bm25_b,
        )

        logger.info("BM25 index created")

    def _init_embeddings(self):
        """Dense 임베딩 생성"""
        logger.info("Generating dense embeddings...")

        # 모든 문서 임베딩
        self._document_embeddings = []
        for doc in self.documents:
            emb = self.embedding_function(doc)
            self._document_embeddings.append(emb)

        logger.info(f"Dense embeddings created: {len(self._document_embeddings)} docs")

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """
        Hybrid Search 수행

        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 k개

        Returns:
            검색 결과 (점수 내림차순)
        """
        # 1. BM25 검색
        bm25_scores = self._bm25_search(query, top_k=top_k * 2)  # 더 많이 검색

        # 2. Dense 검색
        dense_scores = self._dense_search(query, top_k=top_k * 2)

        # 3. Fusion
        if self.fusion_method == "rrf":
            final_scores = self._reciprocal_rank_fusion(bm25_scores, dense_scores)
        elif self.fusion_method == "weighted_sum":
            final_scores = self._weighted_sum_fusion(bm25_scores, dense_scores)
        elif self.fusion_method == "distribution_based":
            final_scores = self._distribution_based_fusion(bm25_scores, dense_scores)
        else:
            # Fallback (안전장치)
            final_scores = self._reciprocal_rank_fusion(bm25_scores, dense_scores)

        # 4. Top-k 선택 (heapq.nlargest 최적화: O(n log n) → O(n log k))
        # 전체 정렬 대신 상위 k개만 선택하여 성능 향상 (k << n일 때 효과적)
        top_results = heapq.nlargest(
            top_k,
            final_scores.items(),
            key=lambda x: x[1]
        )

        # SearchResult 생성
        results = [
            SearchResult(
                text=self.documents[idx],
                score=score,
                metadata={"index": idx, "fusion_method": self.fusion_method},
            )
            for idx, score in top_results
        ]

        logger.info(
            f"Hybrid search completed: query_length={len(query)}, "
            f"top_k={top_k}, fusion={self.fusion_method}"
        )

        return results

    def _bm25_search(self, query: str, top_k: int) -> Dict[int, float]:
        """
        BM25 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 k개

        Returns:
            {문서 인덱스: BM25 점수}
        """
        # 쿼리 토큰화
        query_tokens = query.lower().split()

        # BM25 점수 계산
        scores = self._bm25.get_scores(query_tokens)

        # 상위 k개 선택
        top_indices = scores.argsort()[::-1][:top_k]

        # 딕셔너리 생성
        bm25_scores = {int(idx): float(scores[idx]) for idx in top_indices if scores[idx] > 0}

        return bm25_scores

    def _dense_search(self, query: str, top_k: int) -> Dict[int, float]:
        """
        Dense 임베딩 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 k개

        Returns:
            {문서 인덱스: 코사인 유사도}
        """
        # 쿼리 임베딩
        query_emb = self.embedding_function(query)

        # 코사인 유사도 계산
        similarities = []
        for doc_emb in self._document_embeddings:
            sim = self._cosine_similarity(query_emb, doc_emb)
            similarities.append(sim)

        # 상위 k개 선택
        import numpy as np

        similarities = np.array(similarities)
        top_indices = similarities.argsort()[::-1][:top_k]

        # 딕셔너리 생성
        dense_scores = {
            int(idx): float(similarities[idx]) for idx in top_indices if similarities[idx] > 0
        }

        return dense_scores

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """코사인 유사도 계산"""
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def _reciprocal_rank_fusion(
        self, bm25_scores: Dict[int, float], dense_scores: Dict[int, float]
    ) -> Dict[int, float]:
        """
        Reciprocal Rank Fusion (RRF)

        순위 기반 결합 방식으로, 점수 스케일에 영향을 덜 받습니다.

        RRF(d) = Σ 1 / (k + rank(d))

        Args:
            bm25_scores: {문서 인덱스: BM25 점수}
            dense_scores: {문서 인덱스: Dense 점수}

        Returns:
            {문서 인덱스: RRF 점수}
        """
        # BM25 순위
        bm25_ranked = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
        bm25_ranks = {idx: rank for rank, (idx, _) in enumerate(bm25_ranked)}

        # Dense 순위
        dense_ranked = sorted(dense_scores.items(), key=lambda x: x[1], reverse=True)
        dense_ranks = {idx: rank for rank, (idx, _) in enumerate(dense_ranked)}

        # 모든 문서 인덱스
        all_indices = set(bm25_scores.keys()) | set(dense_scores.keys())

        # RRF 점수 계산
        rrf_scores = {}
        for idx in all_indices:
            score = 0.0

            # BM25 기여
            if idx in bm25_ranks:
                score += 1.0 / (self.rrf_k + bm25_ranks[idx])

            # Dense 기여
            if idx in dense_ranks:
                score += 1.0 / (self.rrf_k + dense_ranks[idx])

            rrf_scores[idx] = score

        return rrf_scores

    def _weighted_sum_fusion(
        self, bm25_scores: Dict[int, float], dense_scores: Dict[int, float]
    ) -> Dict[int, float]:
        """
        가중 평균 Fusion

        점수를 정규화한 후 가중 평균을 계산합니다.

        Args:
            bm25_scores: {문서 인덱스: BM25 점수}
            dense_scores: {문서 인덱스: Dense 점수}

        Returns:
            {문서 인덱스: 가중 평균 점수}
        """
        # 정규화
        bm25_normalized = self._normalize_scores(bm25_scores)
        dense_normalized = self._normalize_scores(dense_scores)

        # 모든 문서 인덱스
        all_indices = set(bm25_scores.keys()) | set(dense_scores.keys())

        # 가중 평균
        weighted_scores = {}
        for idx in all_indices:
            bm25_score = bm25_normalized.get(idx, 0.0)
            dense_score = dense_normalized.get(idx, 0.0)

            weighted_scores[idx] = (
                self.bm25_weight * bm25_score + self.dense_weight * dense_score
            )

        return weighted_scores

    def _distribution_based_fusion(
        self, bm25_scores: Dict[int, float], dense_scores: Dict[int, float]
    ) -> Dict[int, float]:
        """
        Distribution-Based Fusion

        점수 분포를 고려하여 정규화 후 결합합니다.

        Args:
            bm25_scores: {문서 인덱스: BM25 점수}
            dense_scores: {문서 인덱스: Dense 점수}

        Returns:
            {문서 인덱스: 결합 점수}
        """
        import numpy as np

        # 점수를 리스트로 변환
        bm25_values = list(bm25_scores.values())
        dense_values = list(dense_scores.values())

        # 평균과 표준편차 계산
        bm25_mean = np.mean(bm25_values) if bm25_values else 0.0
        bm25_std = np.std(bm25_values) if bm25_values else 1.0
        dense_mean = np.mean(dense_values) if dense_values else 0.0
        dense_std = np.std(dense_values) if dense_values else 1.0

        # Z-score 정규화
        bm25_normalized = {
            idx: (score - bm25_mean) / (bm25_std + 1e-10)
            for idx, score in bm25_scores.items()
        }
        dense_normalized = {
            idx: (score - dense_mean) / (dense_std + 1e-10)
            for idx, score in dense_scores.items()
        }

        # 모든 문서 인덱스
        all_indices = set(bm25_scores.keys()) | set(dense_scores.keys())

        # 결합
        combined_scores = {}
        for idx in all_indices:
            bm25_z = bm25_normalized.get(idx, 0.0)
            dense_z = dense_normalized.get(idx, 0.0)

            combined_scores[idx] = (
                self.bm25_weight * bm25_z + self.dense_weight * dense_z
            )

        return combined_scores

    def _normalize_scores(self, scores: Dict[int, float]) -> Dict[int, float]:
        """
        Min-Max 정규화

        Args:
            scores: {문서 인덱스: 점수}

        Returns:
            {문서 인덱스: 정규화된 점수 (0-1)}
        """
        if not scores:
            return {}

        values = list(scores.values())
        min_score = min(values)
        max_score = max(values)

        # 모든 점수가 같은 경우
        if max_score == min_score:
            return {idx: 1.0 for idx in scores.keys()}

        # Min-Max 정규화
        normalized = {
            idx: (score - min_score) / (max_score - min_score)
            for idx, score in scores.items()
        }

        return normalized

    def add_documents(self, new_documents: List[str]):
        """
        새 문서 추가

        Args:
            new_documents: 추가할 문서 리스트
        """
        self.documents.extend(new_documents)

        # BM25 재초기화
        self._init_bm25()

        # Dense 임베딩 추가
        for doc in new_documents:
            emb = self.embedding_function(doc)
            self._document_embeddings.append(emb)

        logger.info(f"Added {len(new_documents)} documents, total: {len(self.documents)}")

    def __repr__(self) -> str:
        return (
            f"HybridRetriever("
            f"docs={len(self.documents)}, "
            f"fusion={self.fusion_method}, "
            f"bm25_weight={self.bm25_weight}, "
            f"dense_weight={self.dense_weight})"
        )
