"""
SearchStrategy - 검색 전략 패턴
책임: 검색 방법 결정 로직 추상화 (Strategy Pattern)
SOLID 원칙:
- OCP: 새 검색 방법 추가 시 수정 불필요
- SRP: 각 전략은 단일 검색 방법만 담당
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from ...service.types import VectorStoreProtocol


class SearchStrategy(ABC):
    """
    검색 전략 인터페이스

    책임:
    - 검색 방법 정의만

    SOLID:
    - SRP: 단일 검색 방법만 담당
    - OCP: 새 전략 추가 시 기존 코드 수정 불필요
    """

    @abstractmethod
    def search(
        self, vector_store: "VectorStoreProtocol", query: str, k: int, **kwargs: Any
    ) -> List[Any]:
        """
        검색 수행

        Args:
            vector_store: 벡터 스토어
            query: 검색 쿼리
            k: 반환할 결과 수
            **kwargs: 추가 파라미터

        Returns:
            검색 결과 리스트
        """
        pass


class SimilaritySearchStrategy(SearchStrategy):
    """유사도 검색 전략"""

    def search(
        self, vector_store: "VectorStoreProtocol", query: str, k: int, **kwargs: Any
    ) -> List[Any]:
        """유사도 검색"""
        return vector_store.similarity_search(query, k=k, **kwargs)


class HybridSearchStrategy(SearchStrategy):
    """하이브리드 검색 전략"""

    def search(
        self, vector_store: "VectorStoreProtocol", query: str, k: int, **kwargs: Any
    ) -> List[Any]:
        """하이브리드 검색"""
        return vector_store.hybrid_search(query, k=k, **kwargs)


class MMRSearchStrategy(SearchStrategy):
    """MMR 검색 전략"""

    def search(
        self, vector_store: "VectorStoreProtocol", query: str, k: int, **kwargs: Any
    ) -> List[Any]:
        """MMR 검색"""
        return vector_store.mmr_search(query, k=k, **kwargs)


class SearchStrategyFactory:
    """
    검색 전략 팩토리

    책임:
    - 검색 전략 생성만

    SOLID:
    - SRP: 전략 생성만 담당
    - OCP: 새 전략 추가 시 수정 불필요
    """

    _strategies = {
        "similarity": SimilaritySearchStrategy,
        "hybrid": HybridSearchStrategy,
        "mmr": MMRSearchStrategy,
    }

    @classmethod
    def create(cls, search_type: str) -> SearchStrategy:
        """
        검색 전략 생성

        Args:
            search_type: 검색 타입 ("similarity", "hybrid", "mmr")

        Returns:
            SearchStrategy: 검색 전략 인스턴스
        """
        strategy_class = cls._strategies.get(search_type, SimilaritySearchStrategy)
        return strategy_class()
