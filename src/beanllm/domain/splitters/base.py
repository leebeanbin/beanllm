"""
Splitters Base - 텍스트 분할 베이스 클래스
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, List, Optional

if TYPE_CHECKING:
    from ..loaders.types import Document


class BaseTextSplitter(ABC):
    """Text Splitter 베이스 클래스"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        keep_separator: bool = True,
    ):
        """
        Args:
            chunk_size: 청크 크기
            chunk_overlap: 청크 간 겹침
            length_function: 길이 계산 함수
            keep_separator: 구분자 유지 여부
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.keep_separator = keep_separator

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """텍스트 분할"""
        pass

    def split_documents(self, documents: List["Document"]) -> List["Document"]:
        """
        문서 분할

        Args:
            documents: 분할할 문서 리스트

        Returns:
            분할된 문서 리스트
        """

        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.content)
            metadatas.append(doc.metadata)

        return self.create_documents(texts, metadatas)

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List["Document"]:
        """
        텍스트에서 문서 생성

        Args:
            texts: 텍스트 리스트
            metadatas: 메타데이터 리스트

        Returns:
            문서 리스트
        """
        from ..loaders.types import Document

        _metadatas = metadatas or [{}] * len(texts)
        documents = []

        for i, text in enumerate(texts):
            index = 0
            for chunk in self.split_text(text):
                metadata = _metadatas[i].copy()
                metadata["chunk"] = index
                documents.append(Document(content=chunk, metadata=metadata))
                index += 1

        return documents

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """
        작은 청크들을 병합

        Args:
            splits: 분할된 텍스트 조각들
            separator: 구분자

        Returns:
            병합된 청크들
        """
        separator_len = self.length_function(separator)
        docs = []
        current_doc = []
        total = 0

        for split in splits:
            _len = self.length_function(split)

            if total + _len + (separator_len if current_doc else 0) > self.chunk_size:
                if current_doc:
                    doc = separator.join(current_doc)
                    if doc:
                        docs.append(doc)

                    # Overlap 처리
                    while total > self.chunk_overlap or (
                        total + _len + (separator_len if current_doc else 0) > self.chunk_size
                        and total > 0
                    ):
                        total -= self.length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]

            current_doc.append(split)
            total += _len + (separator_len if len(current_doc) > 1 else 0)

        # 마지막 청크
        if current_doc:
            doc = separator.join(current_doc)
            if doc:
                docs.append(doc)

        return docs
