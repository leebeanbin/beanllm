"""
Semantic Similarity - 유사도 계산 및 분할 지점 탐색

임베딩 기반 코사인 유사도 계산 및 의미 경계(breakpoint) 탐지.
"""

import math
from typing import List


def compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    두 벡터 간 코사인 유사도를 계산합니다.

    Args:
        vec1: 첫 번째 임베딩 벡터
        vec2: 두 번째 임베딩 벡터

    Returns:
        0.0 ~ 1.0 범위의 코사인 유사도 (0: 직교, 1: 동일 방향)

    Raises:
        ValueError: 벡터 길이가 다른 경우
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have same length")

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def find_breakpoints(
    embeddings: List[List[float]],
    threshold: float,
    buffer_size: int = 1,
) -> List[int]:
    """
    인접 문장 간 유사도가 임계값 이하인 분할 지점을 찾습니다.

    버퍼를 사용하여 좌/우 문장 그룹의 평균 임베딩 간 유사도를 계산합니다.

    Args:
        embeddings: 문장별 임베딩 리스트
        threshold: 분할 임계값 (이 값 미만이면 분할 지점)
        buffer_size: 유사도 계산 시 사용할 인접 문장 수

    Returns:
        분할 지점 인덱스 리스트 (다음 문장 시작 인덱스)

    Example:
        >>> embs = [[0.1, 0.2], [0.15, 0.25], [0.9, 0.8]]  # 3번째가 급격히 다름
        >>> find_breakpoints(embs, threshold=0.5)
        [2]
    """
    if len(embeddings) <= 1:
        return []

    breakpoints: List[int] = []
    similarities: List[float] = []

    for i in range(len(embeddings) - 1):
        left_start = max(0, i - buffer_size + 1)
        right_end = min(len(embeddings), i + buffer_size + 1)

        left_embeddings = embeddings[left_start : i + 1]
        right_embeddings = embeddings[i + 1 : right_end]

        left_avg = [
            sum(e[j] for e in left_embeddings) / len(left_embeddings)
            for j in range(len(left_embeddings[0]))
        ]
        right_avg = [
            sum(e[j] for e in right_embeddings) / len(right_embeddings)
            for j in range(len(right_embeddings[0]))
        ]

        similarity = compute_cosine_similarity(left_avg, right_avg)
        similarities.append(similarity)

        if similarity < threshold:
            breakpoints.append(i + 1)

    return breakpoints
