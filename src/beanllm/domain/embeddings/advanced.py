"""
Embeddings Advanced - 고급 임베딩 기법들
"""

from typing import List, Optional

from .base import BaseEmbedding
from .utils import batch_cosine_similarity

try:
    from beanllm.utils.logger import get_logger
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


def truncate_embedding(
    embedding: List[float],
    dimension: int,
) -> List[float]:
    """
    Matryoshka Representation Learning: 임베딩 차원 축소

    Matryoshka 임베딩은 하나의 큰 벡터를 여러 작은 차원으로 축소할 수 있습니다.
    이를 통해 저장 공간과 계산 비용을 줄이면서도 성능을 유지할 수 있습니다.

    Args:
        embedding: 원본 임베딩 벡터
        dimension: 축소할 차원 (원본 차원보다 작아야 함)

    Returns:
        축소된 임베딩 벡터

    Example:
        ```python
        from beanllm.domain.embeddings import OpenAIEmbedding, truncate_embedding

        # 1536차원 임베딩 생성
        emb = OpenAIEmbedding(model="text-embedding-3-large")
        vectors = emb.embed_sync(["Hello world"])

        # 768차원으로 축소 (50% 저장 공간 절약)
        truncated = truncate_embedding(vectors[0], dimension=768)

        # 256차원으로 축소 (83% 저장 공간 절약)
        small = truncate_embedding(vectors[0], dimension=256)
        ```

    References:
        - "Matryoshka Representation Learning" (NeurIPS 2022)
        - https://arxiv.org/abs/2205.13147
    """
    if dimension > len(embedding):
        logger.warning(
            f"Requested dimension ({dimension}) is larger than "
            f"embedding dimension ({len(embedding)}). Returning original."
        )
        return embedding

    truncated = embedding[:dimension]

    logger.info(
        f"Truncated embedding: {len(embedding)} -> {dimension} "
        f"({100 * (1 - dimension / len(embedding)):.1f}% reduction)"
    )

    return truncated


def batch_truncate_embeddings(
    embeddings: List[List[float]],
    dimension: int,
) -> List[List[float]]:
    """
    배치 임베딩 차원 축소

    Args:
        embeddings: 임베딩 벡터 리스트
        dimension: 축소할 차원

    Returns:
        축소된 임베딩 벡터 리스트

    Example:
        ```python
        from beanllm.domain.embeddings import embed_sync, batch_truncate_embeddings

        # 여러 텍스트 임베딩
        vectors = embed_sync(["text1", "text2", "text3"])

        # 모두 256차원으로 축소
        truncated_vectors = batch_truncate_embeddings(vectors, dimension=256)
        ```
    """
    return [truncate_embedding(emb, dimension) for emb in embeddings]


class MatryoshkaEmbedding(BaseEmbedding):
    """
    Matryoshka Embedding Wrapper

    기존 임베딩 모델을 Matryoshka 방식으로 사용할 수 있게 래핑합니다.
    차원을 동적으로 축소하여 저장 공간과 계산 비용을 절감합니다.

    지원 차원:
    - 1536 -> 768: 50% 절약, ~5% 성능 손실
    - 1536 -> 512: 67% 절약, ~10% 성능 손실
    - 1536 -> 256: 83% 절약, ~15% 성능 손실

    Example:
        ```python
        from beanllm.domain.embeddings import MatryoshkaEmbedding, OpenAIEmbedding

        # 기존 임베딩 모델
        base_emb = OpenAIEmbedding(model="text-embedding-3-large")

        # Matryoshka 래퍼 (512차원으로 축소)
        mat_emb = MatryoshkaEmbedding(
            base_embedding=base_emb,
            output_dimension=512
        )

        # 사용 (자동으로 512차원으로 축소됨)
        vectors = mat_emb.embed_sync(["text1", "text2"])
        print(len(vectors[0]))  # 512
        ```
    """

    def __init__(
        self,
        base_embedding: BaseEmbedding,
        output_dimension: int = 768,
        **kwargs,
    ):
        """
        Args:
            base_embedding: 기본 임베딩 모델
            output_dimension: 출력 차원 (축소할 차원)
            **kwargs: 추가 파라미터
        """
        super().__init__(model=base_embedding.model, **kwargs)

        self.base_embedding = base_embedding
        self.output_dimension = output_dimension

        logger.info(
            f"MatryoshkaEmbedding initialized: "
            f"model={base_embedding.model}, output_dim={output_dimension}"
        )

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 후 차원 축소 (비동기)"""
        # 기본 임베딩 생성
        embeddings = await self.base_embedding.embed(texts)

        # 차원 축소
        truncated = batch_truncate_embeddings(embeddings, self.output_dimension)

        return truncated

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 후 차원 축소 (동기)"""
        # 기본 임베딩 생성
        embeddings = self.base_embedding.embed_sync(texts)

        # 차원 축소
        truncated = batch_truncate_embeddings(embeddings, self.output_dimension)

        return truncated
