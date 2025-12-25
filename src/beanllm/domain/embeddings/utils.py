"""
Embeddings Utils - 임베딩 유틸리티 함수들
"""

from typing import List

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    from ...utils.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    두 벡터 간의 코사인 유사도 계산

    코사인 유사도는 벡터의 방향(의미)을 측정하므로,
    텍스트 임베딩의 의미적 유사도를 비교할 때 적합합니다.

    Args:
        vec1: 첫 번째 임베딩 벡터
        vec2: 두 번째 임베딩 벡터

    Returns:
        코사인 유사도 값 (-1 ~ 1, 1에 가까울수록 유사)

    Example:
        ```python
        from beanllm.domain.embeddings import embed_sync, cosine_similarity

        vec1 = embed_sync("고양이는 귀여워")[0]
        vec2 = embed_sync("강아지는 귀여워")[0]
        similarity = cosine_similarity(vec1, vec2)
        print(f"유사도: {similarity:.3f}")  # 0.8 정도
        ```

    수학적 고려사항:
        - 벡터가 이미 정규화되어 있으면 내적만으로 계산 가능
        - 정규화되지 않은 벡터는 자동으로 정규화하여 계산
        - 코사인 유사도는 벡터의 크기(길이)에 영향을 받지 않음
    """
    if not HAS_NUMPY:
        # numpy가 없는 경우 순수 Python 구현
        if len(vec1) != len(vec2):
            raise ValueError(
                f"벡터 차원이 다릅니다: {len(vec1)} vs {len(vec2)}. "
                "같은 모델로 생성한 임베딩을 사용해야 합니다."
            )

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            logger.warning("영벡터가 감지되었습니다. 유사도는 0으로 반환합니다.")
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return max(-1.0, min(1.0, similarity))

    try:
        v1 = np.array(vec1, dtype=np.float32)
        v2 = np.array(vec2, dtype=np.float32)

        # 차원 확인
        if len(v1) != len(v2):
            raise ValueError(
                f"벡터 차원이 다릅니다: {len(v1)} vs {len(v2)}. "
                "같은 모델로 생성한 임베딩을 사용해야 합니다."
            )

        # L2 정규화 (코사인 유사도 계산을 위해)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            logger.warning("영벡터가 감지되었습니다. 유사도는 0으로 반환합니다.")
            return 0.0

        # 코사인 유사도 = (A · B) / (||A|| * ||B||)
        similarity = np.dot(v1, v2) / (norm1 * norm2)

        # 수치 안정성을 위해 -1과 1 사이로 클리핑
        return float(np.clip(similarity, -1.0, 1.0))

    except Exception as e:
        logger.error(f"코사인 유사도 계산 중 오류: {e}")
        raise


def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
    """
    두 벡터 간의 유클리드 거리 계산

    유클리드 거리는 벡터의 크기와 방향을 모두 고려하므로,
    벡터의 절대적 차이를 측정할 때 사용합니다.

    Args:
        vec1: 첫 번째 임베딩 벡터
        vec2: 두 번째 임베딩 벡터

    Returns:
        유클리드 거리 (0에 가까울수록 유사)

    Example:
        ```python
        from beanllm.domain.embeddings import embed_sync, euclidean_distance

        vec1 = embed_sync("고양이는 귀여워")[0]
        vec2 = embed_sync("강아지는 귀여워")[0]
        distance = euclidean_distance(vec1, vec2)
        print(f"거리: {distance:.3f}")  # 작을수록 유사
        ```

    수학적 고려사항:
        - 거리가 작을수록 유사도가 높음
        - 벡터의 크기(스케일)에 영향을 받음
        - 코사인 유사도와 달리 벡터의 절대적 위치를 비교
    """
    if not HAS_NUMPY:
        # numpy가 없는 경우 순수 Python 구현
        if len(vec1) != len(vec2):
            raise ValueError(f"벡터 차원이 다릅니다: {len(vec1)} vs {len(vec2)}")

        distance = sum((a - b) ** 2 for a, b in zip(vec1, vec2)) ** 0.5
        return distance

    try:
        v1 = np.array(vec1, dtype=np.float32)
        v2 = np.array(vec2, dtype=np.float32)

        if len(v1) != len(v2):
            raise ValueError(f"벡터 차원이 다릅니다: {len(v1)} vs {len(v2)}")

        # 유클리드 거리 = sqrt(sum((a_i - b_i)^2))
        distance = np.linalg.norm(v1 - v2)
        return float(distance)

    except Exception as e:
        logger.error(f"유클리드 거리 계산 중 오류: {e}")
        raise


def normalize_vector(vec: List[float]) -> List[float]:
    """
    벡터를 L2 정규화 (단위 벡터로 변환)

    정규화된 벡터는 크기가 1이 되어 코사인 유사도 계산이 간단해집니다.
    많은 임베딩 모델은 이미 정규화된 벡터를 반환하지만,
    필요시 명시적으로 정규화할 수 있습니다.

    Args:
        vec: 정규화할 벡터

    Returns:
        L2 정규화된 벡터 (크기 = 1)

    Example:
        ```python
        from beanllm.domain.embeddings import embed_sync, normalize_vector

        vec = embed_sync("Hello world")[0]
        normalized = normalize_vector(vec)

        # 정규화 확인
        import math
        norm = math.sqrt(sum(x**2 for x in normalized))
        print(f"정규화 후 크기: {norm:.6f}")  # 1.0에 가까움
        ```

    수학적 고려사항:
        - L2 정규화: v / ||v||
        - 영벡터는 정규화할 수 없음 (원본 반환)
        - 정규화 후 벡터의 방향은 유지되고 크기만 1로 변경
    """
    if not HAS_NUMPY:
        # numpy가 없는 경우 순수 Python 구현
        norm = sum(x * x for x in vec) ** 0.5

        if norm == 0:
            logger.warning("영벡터는 정규화할 수 없습니다. 원본을 반환합니다.")
            return vec

        return [x / norm for x in vec]

    try:
        v = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(v)

        if norm == 0:
            logger.warning("영벡터는 정규화할 수 없습니다. 원본을 반환합니다.")
            return vec

        normalized = v / norm
        return normalized.tolist()

    except Exception as e:
        logger.error(f"벡터 정규화 중 오류: {e}")
        raise


def batch_cosine_similarity(
    query_vec: List[float], candidate_vecs: List[List[float]]
) -> List[float]:
    """
    하나의 쿼리 벡터와 여러 후보 벡터들 간의 코사인 유사도를 일괄 계산

    검색이나 유사도 기반 랭킹에 유용합니다.

    Args:
        query_vec: 쿼리 임베딩 벡터
        candidate_vecs: 후보 임베딩 벡터들의 리스트

    Returns:
        각 후보 벡터와의 코사인 유사도 리스트

    Example:
        ```python
        from beanllm.domain.embeddings import embed_sync, batch_cosine_similarity

        query = embed_sync("고양이")[0]
        candidates = embed_sync(["강아지", "고양이", "자동차"])
        similarities = batch_cosine_similarity(query, candidates)

        # 가장 유사한 것 찾기
        best_idx = similarities.index(max(similarities))
        print(f"가장 유사한 것: {['강아지', '고양이', '자동차'][best_idx]}")
        ```

    수학적 고려사항:
        - 배치 처리로 효율적인 계산
        - 모든 벡터는 같은 차원이어야 함
        - 정규화된 벡터를 사용하면 내적만으로 계산 가능 (더 빠름)
    """
    if not HAS_NUMPY:
        # numpy가 없는 경우 순수 Python 구현
        return [cosine_similarity(query_vec, candidate) for candidate in candidate_vecs]

    try:
        query = np.array(query_vec, dtype=np.float32)
        candidates = np.array(candidate_vecs, dtype=np.float32)

        if len(query) != candidates.shape[1]:
            raise ValueError(
                f"벡터 차원이 다릅니다: 쿼리 {len(query)} vs 후보 {candidates.shape[1]}"
            )

        # 정규화
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return [0.0] * len(candidate_vecs)

        candidate_norms = np.linalg.norm(candidates, axis=1, keepdims=True)

        # 코사인 유사도 계산 (배치)
        similarities = np.dot(candidates, query) / (candidate_norms.flatten() * query_norm)

        # 클리핑
        similarities = np.clip(similarities, -1.0, 1.0)

        return similarities.tolist()

    except Exception as e:
        logger.error(f"배치 코사인 유사도 계산 중 오류: {e}")
        raise
