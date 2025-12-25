"""
Base classes for vector stores
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# 순환 참조 방지를 위해 TYPE_CHECKING 사용
if TYPE_CHECKING:
    from ...domain.loaders import Document
else:
    # 런타임에만 import
    try:
        from ...domain.loaders import Document
    except ImportError:
        Document = Any  # type: ignore

# AdvancedSearchMixin은 순환 참조 방지를 위해 구현체에서만 사용
# base.py에서는 직접 상속하지 않음


@dataclass
class VectorSearchResult:
    """벡터 검색 결과"""

    document: Any  # type: ignore  # Document 타입
    score: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseVectorStore(ABC):
    """
    Base class for all vector stores

    모든 vector store 구현의 기본 클래스

    Note: AdvancedSearchMixin은 각 구현체에서 상속받아 사용합니다.
    (순환 참조 방지를 위해 base.py에서는 직접 상속하지 않음)
    """

    def __init__(self, embedding_function=None, **kwargs):
        """
        Args:
            embedding_function: 임베딩 함수 (texts -> vectors)
        """
        self.embedding_function = embedding_function

    @abstractmethod
    def add_documents(self, documents: List[Any], **kwargs) -> List[str]:
        """
        문서 추가

        Args:
            documents: 추가할 문서 리스트

        Returns:
            추가된 문서 ID 리스트
        """
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[VectorSearchResult]:
        """
        유사도 검색

        Args:
            query: 검색 쿼리
            k: 반환할 결과 수

        Returns:
            검색 결과 리스트
        """
        pass

    @abstractmethod
    def delete(self, ids: List[str], **kwargs) -> bool:
        """
        문서 삭제

        Args:
            ids: 삭제할 문서 ID 리스트

        Returns:
            성공 여부
        """
        pass

    def add_texts(
        self, texts: List[str], metadatas: Optional[List[Dict]] = None, **kwargs
    ) -> List[str]:
        """
        텍스트 직접 추가

        Args:
            texts: 텍스트 리스트
            metadatas: 메타데이터 리스트 (옵션)

        Returns:
            추가된 문서 ID 리스트
        """
        # 런타임에 Document import
        from ...domain.loaders import Document

        documents = [
            Document(content=text, metadata=metadatas[i] if metadatas else {})
            for i, text in enumerate(texts)
        ]
        return self.add_documents(documents, **kwargs)

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs
    ) -> List[VectorSearchResult]:
        """
        비동기 유사도 검색

        Args:
            query: 검색 쿼리
            k: 반환할 결과 수

        Returns:
            검색 결과 리스트
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.similarity_search(query, k, **kwargs))

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        코사인 유사도 계산

        Args:
            vec1: 벡터 1
            vec2: 벡터 2

        Returns:
            유사도 (0.0 ~ 1.0)
        """
        try:
            import numpy as np

            a = np.array(vec1)
            b = np.array(vec2)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        except ImportError:
            # numpy 없으면 수동 계산
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm_a = sum(a * a for a in vec1) ** 0.5
            norm_b = sum(b * b for b in vec2) ** 0.5
            return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0

    async def batch_similarity_search(
        self, queries: List[str], k: int = 4, use_gpu: bool = False, **kwargs
    ) -> List[List[VectorSearchResult]]:
        """
        배치 벡터 검색

        Args:
            queries: 검색 쿼리 리스트
            k: 반환할 결과 수
            use_gpu: GPU 사용 여부 (선택적, 기본값: False)
            **kwargs: 추가 파라미터

        Returns:
            각 쿼리별 검색 결과 리스트
        """
        if not self.embedding_function:
            raise ValueError("Embedding function required for batch search")

        # 1. 배치 임베딩
        query_vecs = await self._batch_embed(queries)

        # 2. 배치 검색 (GPU 또는 CPU)
        if use_gpu:
            return await self._gpu_batch_search(query_vecs, k, **kwargs)
        else:
            return await self._cpu_batch_search(query_vecs, k, **kwargs)

    async def _batch_embed(self, queries: List[str]) -> List[List[float]]:
        """배치 임베딩"""
        if hasattr(self.embedding_function, "embed_sync"):
            return self.embedding_function.embed_sync(queries)
        elif hasattr(self.embedding_function, "__call__"):
            # 동기 함수
            result = self.embedding_function(queries)
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
                return result
            else:
                # 단일 벡터 반환 시 리스트로 변환
                return [result] if not isinstance(result, list) else result
        else:
            # 비동기 함수
            return await self.embedding_function(queries)

    async def _cpu_batch_search(
        self, query_vecs: List[List[float]], k: int, **kwargs
    ) -> List[List[VectorSearchResult]]:
        """CPU 배치 검색 (NumPy 행렬 연산)"""
        try:
            import numpy as np
        except ImportError:
            # NumPy 없으면 순차 처리
            results = []
            for vec in query_vecs:
                # 단일 벡터 검색 (구현체별로 다름)
                result = await self.asimilarity_search_by_vector(vec, k, **kwargs)
                results.append(result)
            return results

        # 모든 벡터 가져오기 (구현체별로 override 필요)
        all_vectors, all_documents = self._get_all_vectors_and_docs()

        if not all_vectors:
            return [[] for _ in query_vecs]

        # 행렬 연산
        query_matrix = np.array(query_vecs, dtype=np.float32)
        candidate_matrix = np.array(all_vectors, dtype=np.float32)

        # 코사인 유사도 (정규화된 벡터 가정)
        # 정규화
        query_norms = np.linalg.norm(query_matrix, axis=1, keepdims=True)
        candidate_norms = np.linalg.norm(candidate_matrix, axis=1, keepdims=True)
        query_matrix_norm = query_matrix / (query_norms + 1e-8)
        candidate_matrix_norm = candidate_matrix / (candidate_norms + 1e-8)

        similarities = np.dot(query_matrix_norm, candidate_matrix_norm.T)

        # Top-k 선택
        top_k_indices = np.argsort(similarities, axis=1)[:, -k:][:, ::-1]

        # 결과 구성
        results = []
        for i, indices in enumerate(top_k_indices):
            query_results = [
                VectorSearchResult(
                    document=all_documents[idx], score=float(similarities[i, idx]), metadata={}
                )
                for idx in indices
            ]
            results.append(query_results)

        return results

    async def _gpu_batch_search(
        self, query_vecs: List[List[float]], k: int, **kwargs
    ) -> List[List[VectorSearchResult]]:
        """GPU 배치 검색 (선택적, GPU 없으면 CPU로 폴백)"""
        try:
            import cupy as cp  # noqa: F401

            HAS_CUDA = True
        except ImportError:
            HAS_CUDA = False

        if not HAS_CUDA:
            # GPU 없으면 CPU로 폴백
            return await self._cpu_batch_search(query_vecs, k, **kwargs)

        # GPU 연산은 복잡하므로 일단 CPU로 폴백
        # 필요시 나중에 구현
        return await self._cpu_batch_search(query_vecs, k, **kwargs)

    async def asimilarity_search_by_vector(
        self, query_vec: List[float], k: int = 4, **kwargs
    ) -> List[VectorSearchResult]:
        """
        벡터로 직접 검색 (배치 검색을 위한 헬퍼)

        기본 구현: similarity_search를 사용
        구현체에서 override 가능
        """
        # 기본 구현: 임시 쿼리 문자열로 검색 (비효율적)
        # 구현체에서 override 권장
        raise NotImplementedError("asimilarity_search_by_vector must be implemented by subclasses")

    def _get_all_vectors_and_docs(self) -> tuple[List[List[float]], List[Any]]:
        """
        모든 벡터와 문서 가져오기 (구현체별로 override 필요)

        기본 구현: 빈 리스트 반환
        """
        return [], []
