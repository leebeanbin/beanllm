"""
Base Reranker - 재순위화 모델 추상 클래스
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union

from .types import RerankResult


class BaseReranker(ABC):
    """
    재순위화 모델 베이스 클래스

    검색 결과를 재정렬하여 관련성 높은 문서를 상위에 배치합니다.

    Example:
        ```python
        class MyReranker(BaseReranker):
            def rerank(self, query: str, documents: List[str], top_k: int = 5):
                # 재순위화 로직
                scores = self.model.score(query, documents)
                results = [
                    RerankResult(text=doc, score=score, index=idx)
                    for idx, (doc, score) in enumerate(zip(documents, scores))
                ]
                return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]
        ```
    """

    @abstractmethod
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
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
