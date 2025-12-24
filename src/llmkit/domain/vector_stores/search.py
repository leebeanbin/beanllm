"""
Advanced search algorithms
Hybrid, MMR, Re-ranking 등
"""

from typing import Dict, List, Optional, Tuple

from .base import VectorSearchResult


class SearchAlgorithms:
    """고급 검색 알고리즘 모음"""

    @staticmethod
    def hybrid_search(
        vector_store, query: str, k: int = 4, alpha: float = 0.5, **kwargs
    ) -> List[VectorSearchResult]:
        """
        Hybrid Search (벡터 + 키워드 검색)

        Args:
            vector_store: VectorStore 인스턴스
            query: 검색 쿼리
            k: 반환할 결과 수
            alpha: 벡터 검색 가중치 (0.0 ~ 1.0)
                   0.0 = 키워드만, 1.0 = 벡터만, 0.5 = 균형

        Returns:
            검색 결과 리스트
        """
        # 1. 벡터 검색
        vector_results = vector_store.similarity_search(query, k=k * 2, **kwargs)

        # 2. 키워드 검색
        keyword_results = SearchAlgorithms._keyword_search(vector_store, query, k=k * 2)

        # 3. 점수 결합 (RRF)
        combined = SearchAlgorithms._combine_results(vector_results, keyword_results, alpha=alpha)

        return combined[:k]

    @staticmethod
    def _keyword_search(vector_store, query: str, k: int = 10) -> List[VectorSearchResult]:
        """
        키워드 기반 검색 (BM25 스타일)

        Note: 기본 구현은 빈 리스트 반환.
        각 provider에서 override 필요.
        """
        # Provider별로 구현해야 함
        return []

    @staticmethod
    def _combine_results(
        vector_results: List[VectorSearchResult],
        keyword_results: List[VectorSearchResult],
        alpha: float = 0.5,
    ) -> List[VectorSearchResult]:
        """
        벡터와 키워드 결과 결합 (RRF - Reciprocal Rank Fusion)

        Args:
            vector_results: 벡터 검색 결과
            keyword_results: 키워드 검색 결과
            alpha: 벡터 검색 가중치

        Returns:
            결합된 결과
        """
        # 문서 ID -> (결과, 벡터 순위, 키워드 순위)
        results_map: Dict[str, Tuple[VectorSearchResult, Optional[int], Optional[int]]] = {}

        # 벡터 검색 결과
        for rank, result in enumerate(vector_results, 1):
            doc_id = id(result.document)
            results_map[doc_id] = (result, rank, None)

        # 키워드 검색 결과
        for rank, result in enumerate(keyword_results, 1):
            doc_id = id(result.document)
            if doc_id in results_map:
                prev_result, vec_rank, _ = results_map[doc_id]
                results_map[doc_id] = (prev_result, vec_rank, rank)
            else:
                results_map[doc_id] = (result, None, rank)

        # RRF 점수 계산
        k_constant = 60  # RRF constant
        scored_results = []

        for doc_id, (result, vec_rank, key_rank) in results_map.items():
            vec_score = alpha / (k_constant + vec_rank) if vec_rank else 0
            key_score = (1 - alpha) / (k_constant + key_rank) if key_rank else 0
            total_score = vec_score + key_score

            scored_results.append(
                VectorSearchResult(
                    document=result.document, score=total_score, metadata=result.metadata
                )
            )

        # 점수로 정렬
        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results

    @staticmethod
    def rerank(
        query: str,
        results: List[VectorSearchResult],
        model: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[VectorSearchResult]:
        """
        Re-ranking with Cross-encoder

        Args:
            query: 쿼리
            results: 초기 검색 결과
            model: Cross-encoder 모델
            top_k: 재순위화 후 반환할 개수

        Returns:
            재순위화된 결과
        """
        if not results:
            return []

        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError("sentence-transformers 필요:\npip install sentence-transformers")

        # 모델 로드
        model_name = model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        cross_encoder = CrossEncoder(model_name)

        # (query, document) 쌍 생성
        pairs = [[query, result.document.content] for result in results]

        # Cross-encoder로 점수 계산
        scores = cross_encoder.predict(pairs)

        # 점수로 재정렬
        reranked_results = []
        for result, score in zip(results, scores):
            reranked_results.append(
                VectorSearchResult(
                    document=result.document, score=float(score), metadata=result.metadata
                )
            )

        reranked_results.sort(key=lambda x: x.score, reverse=True)

        if top_k:
            return reranked_results[:top_k]
        return reranked_results

    @staticmethod
    def mmr_search(
        vector_store, query: str, k: int = 4, fetch_k: int = 20, lambda_param: float = 0.5, **kwargs
    ) -> List[VectorSearchResult]:
        """
        MMR (Maximal Marginal Relevance) 검색 - 다양성 고려

        Args:
            vector_store: VectorStore 인스턴스
            query: 검색 쿼리
            k: 최종 반환 개수
            fetch_k: 초기 가져올 개수
            lambda_param: 관련성 vs 다양성 (0.0 ~ 1.0)

        Returns:
            다양성을 고려한 검색 결과
        """
        # 초기 검색
        candidates = vector_store.similarity_search(query, k=fetch_k, **kwargs)

        if not candidates or len(candidates) <= k:
            return candidates

        # 임베딩 함수 체크
        if not vector_store.embedding_function:
            return candidates[:k]

        # 쿼리 임베딩
        query_vec = vector_store.embedding_function([query])[0]

        # 후보 벡터들
        candidate_vecs = [
            vector_store.embedding_function([c.document.content])[0] for c in candidates
        ]

        # MMR 알고리즘
        selected_indices = []
        remaining_indices = list(range(len(candidates)))

        for _ in range(min(k, len(candidates))):
            best_score = float("-inf")
            best_idx = None

            for idx in remaining_indices:
                # 관련성 점수
                relevance = vector_store._cosine_similarity(query_vec, candidate_vecs[idx])

                # 다양성 점수
                if selected_indices:
                    diversity = max(
                        vector_store._cosine_similarity(
                            candidate_vecs[idx], candidate_vecs[selected_idx]
                        )
                        for selected_idx in selected_indices
                    )
                else:
                    diversity = 0

                # MMR 점수
                mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

        return [candidates[idx] for idx in selected_indices]


# Mixin class for vector stores
class AdvancedSearchMixin:
    """
    고급 검색 기능을 BaseVectorStore에 추가하는 Mixin

    이 Mixin을 사용하면 hybrid_search, mmr_search, rerank를
    자동으로 사용할 수 있습니다.
    """

    def hybrid_search(
        self, query: str, k: int = 4, alpha: float = 0.5, **kwargs
    ) -> List[VectorSearchResult]:
        """Hybrid Search (벡터 + 키워드)"""
        return SearchAlgorithms.hybrid_search(self, query, k, alpha, **kwargs)

    def rerank(
        self,
        query: str,
        results: List[VectorSearchResult],
        model: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[VectorSearchResult]:
        """Re-ranking with Cross-encoder"""
        return SearchAlgorithms.rerank(query, results, model, top_k)

    def mmr_search(
        self, query: str, k: int = 4, fetch_k: int = 20, lambda_param: float = 0.5, **kwargs
    ) -> List[VectorSearchResult]:
        """MMR 검색 (다양성 고려)"""
        return SearchAlgorithms.mmr_search(self, query, k, fetch_k, lambda_param, **kwargs)
