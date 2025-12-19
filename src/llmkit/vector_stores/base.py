"""
Base classes for vector stores
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..document_loaders import Document


@dataclass
class VectorSearchResult:
    """벡터 검색 결과"""

    document: Document
    score: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseVectorStore(ABC):
    """
    Base class for all vector stores

    모든 vector store 구현의 기본 클래스
    """

    def __init__(self, embedding_function=None, **kwargs):
        """
        Args:
            embedding_function: 임베딩 함수 (texts -> vectors)
        """
        self.embedding_function = embedding_function

    @abstractmethod
    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
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
