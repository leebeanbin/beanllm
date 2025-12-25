"""
Embeddings Base - 임베딩 베이스 클래스
"""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbedding(ABC):
    """Embedding 베이스 클래스"""

    def __init__(self, model: str, **kwargs):
        """
        Args:
            model: 모델 이름
            **kwargs: provider별 추가 파라미터
        """
        self.model = model
        self.kwargs = kwargs

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        텍스트들을 임베딩

        Args:
            texts: 임베딩할 텍스트 리스트

        Returns:
            임베딩 벡터 리스트
        """
        pass

    @abstractmethod
    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """
        텍스트들을 임베딩 (동기)

        Args:
            texts: 임베딩할 텍스트 리스트

        Returns:
            임베딩 벡터 리스트
        """
        pass
