"""
Embeddings Advanced - 고급 임베딩 기법들
"""

from typing import List, Optional

from .base import BaseEmbedding
from .utils import batch_cosine_similarity

try:
    from ...utils.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


def find_hard_negatives(
    query_vec: List[float],
    candidate_vecs: List[List[float]],
    positive_vecs: Optional[List[List[float]]] = None,
    similarity_threshold: tuple = (0.3, 0.7),
    top_k: Optional[int] = None,
) -> List[int]:
    """
    Hard Negative Mining: 학습에 유용한 어려운 negative 샘플 찾기

    Hard Negative는 쿼리와 관련 없어 보이지만 실제로는 관련 있는 샘플로,
    모델 학습 시 중요한 역할을 합니다.

    Args:
        query_vec: 쿼리 임베딩 벡터
        candidate_vecs: 후보 임베딩 벡터들의 리스트
        positive_vecs: Positive 샘플 벡터들 (선택적, 제외용)
        similarity_threshold: (min, max) 유사도 범위 (이 범위 안이 Hard Negative)
        top_k: 반환할 Hard Negative 개수 (None이면 모두)

    Returns:
        Hard Negative 인덱스 리스트

    Example:
        ```python
        from beanllm.domain.embeddings import embed_sync, find_hard_negatives

        query = embed_sync("고양이 사료")[0]
        candidates = embed_sync([
            "강아지 사료",  # Hard Negative (비슷하지만 다름)
            "고양이 장난감",  # Hard Negative
            "자동차",  # Easy Negative (너무 다름)
            "고양이 먹이"  # Positive (같음)
        ])

        hard_neg_indices = find_hard_negatives(
            query, candidates,
            similarity_threshold=(0.3, 0.7)
        )
        # → [0, 1] (강아지 사료, 고양이 장난감)
        ```

    수학적 원리:
        - Easy Negative: 유사도 < 0.3 (너무 다름, 학습에 도움 안 됨)
        - Hard Negative: 0.3 < 유사도 < 0.7 (비슷하지만 다름, 학습에 중요!)
        - Positive: 유사도 > 0.7 (같음, 제외)
    """
    # 모든 후보와의 유사도 계산
    similarities = batch_cosine_similarity(query_vec, candidate_vecs)

    # Positive 제외 (제공된 경우)
    if positive_vecs:
        [
            max(batch_cosine_similarity(query_vec, [pv])[0] for pv in positive_vecs)
            for _ in candidate_vecs
        ]
        # Positive와 유사한 것 제외
        similarities = [s if s < 0.7 else -1.0 for s in similarities]

    # Hard Negative 찾기 (유사도 범위 내)
    min_sim, max_sim = similarity_threshold
    hard_neg_indices = [i for i, sim in enumerate(similarities) if min_sim < sim < max_sim]

    # 유사도 순으로 정렬
    hard_neg_with_sim = [(i, similarities[i]) for i in hard_neg_indices]
    hard_neg_with_sim.sort(key=lambda x: x[1], reverse=True)

    # Top-k 선택
    if top_k is not None:
        hard_neg_with_sim = hard_neg_with_sim[:top_k]

    return [i for i, _ in hard_neg_with_sim]


def mmr_search(
    query_vec: List[float],
    candidate_vecs: List[List[float]],
    k: int = 5,
    lambda_param: float = 0.6,
) -> List[int]:
    """
    MMR (Maximal Marginal Relevance) 검색: 다양성을 고려한 검색

    관련성과 다양성을 균형있게 고려하여 검색 결과를 선택합니다.

    Args:
        query_vec: 쿼리 임베딩 벡터
        candidate_vecs: 후보 임베딩 벡터들의 리스트
        k: 반환할 결과 개수
        lambda_param: 관련성 vs 다양성 균형 (0.0-1.0, 높을수록 관련성 중시)

    Returns:
        선택된 후보 인덱스 리스트 (다양성 고려)

    Example:
        ```python
        from beanllm.domain.embeddings import embed_sync, mmr_search

        query = embed_sync("고양이")[0]
        candidates = embed_sync([
            "고양이 사료", "고양이 사료 추천", "고양이 사료 종류",  # 모두 비슷함
            "고양이 건강", "고양이 행동"  # 다른 주제
        ])

        # 일반 검색: 모두 "사료" 관련
        # MMR 검색: 다양한 주제 포함
        selected = mmr_search(query, candidates, k=3, lambda_param=0.6)
        # → [0, 3, 4] (사료, 건강, 행동 - 다양함!)
        ```

    수학적 원리:
        MMR = argmax[λ × sim(q, d) - (1-λ) × max(sim(d, d_selected))]
        - λ × sim(q, d): 쿼리와의 관련성
        - (1-λ) × max(sim(d, d_selected)): 이미 선택된 문서와의 차이 (다양성)
    """
    if k >= len(candidate_vecs):
        return list(range(len(candidate_vecs)))

    # 쿼리와 모든 후보의 유사도
    query_similarities = batch_cosine_similarity(query_vec, candidate_vecs)

    # 첫 번째: 가장 관련성 높은 것
    selected = [query_similarities.index(max(query_similarities))]
    remaining = set(range(len(candidate_vecs))) - set(selected)

    # 나머지 k-1개 선택
    for _ in range(k - 1):
        if not remaining:
            break

        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            # 관련성 점수
            relevance = query_similarities[idx]

            # 다양성 점수 (이미 선택된 것과의 최대 유사도)
            diversity = 0.0
            if selected:
                selected_vecs = [candidate_vecs[i] for i in selected]
                candidate_sims = batch_cosine_similarity(candidate_vecs[idx], selected_vecs)
                diversity = max(candidate_sims) if candidate_sims else 0.0

            # MMR 점수
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return selected


def query_expansion(
    query: str,
    embedding: BaseEmbedding,
    expansion_candidates: Optional[List[str]] = None,
    top_k: int = 3,
    similarity_threshold: float = 0.7,
) -> List[str]:
    """
    Query Expansion: 쿼리를 유사어로 확장하여 검색 범위 확대

    원본 쿼리와 유사한 용어를 추가하여 검색 리콜을 향상시킵니다.

    Args:
        query: 원본 쿼리
        embedding: 임베딩 인스턴스
        expansion_candidates: 확장 후보 단어/구 리스트 (None이면 자동 생성 불가)
        top_k: 추가할 확장어 개수
        similarity_threshold: 유사도 임계값 (이 이상만 추가)

    Returns:
        확장된 쿼리 리스트 [원본, 확장1, 확장2, ...]

    Example:
        ```python
        from beanllm.domain.embeddings import Embedding, query_expansion

        emb = Embedding(model="text-embedding-3-small")

        # 후보 단어 제공
        candidates = ["고양이", "냥이", "고양이과", "cat", "feline", "강아지"]

        expanded = query_expansion("고양이", emb, candidates, top_k=3)
        # → ["고양이", "냥이", "고양이과", "cat"]
        ```

    언어학적 원리:
        - 동의어/유사어 추가로 검색 범위 확대
        - 예: "고양이" → "고양이", "냥이", "cat", "feline"
        - 리콜 향상 (더 많은 관련 문서 발견)
    """
    expanded = [query]

    if not expansion_candidates:
        logger.warning("expansion_candidates가 없으면 확장 불가. 원본만 반환합니다.")
        return expanded

    # 원본 쿼리 임베딩
    query_vec = embedding.embed_sync([query])[0]

    # 후보 임베딩
    candidate_vecs = embedding.embed_sync(expansion_candidates)

    # 유사도 계산
    similarities = batch_cosine_similarity(query_vec, candidate_vecs)

    # 유사도가 높은 순으로 정렬
    candidate_with_sim = list(zip(expansion_candidates, similarities))
    candidate_with_sim.sort(key=lambda x: x[1], reverse=True)

    # 임계값 이상이고 원본과 다른 것만 추가
    for candidate, sim in candidate_with_sim:
        if sim >= similarity_threshold and candidate.lower() != query.lower():
            expanded.append(candidate)
            if len(expanded) >= top_k + 1:  # +1은 원본 포함
                break

    return expanded
