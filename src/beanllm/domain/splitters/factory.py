"""
Splitters Factory - 텍스트 분할 팩토리
"""

from typing import TYPE_CHECKING, List, Optional

from .base import BaseTextSplitter
from .splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

if TYPE_CHECKING:
    from ..loaders.types import Document
else:
    # 런타임에만 import
    try:
        from ..loaders.types import Document
    except ImportError:
        from typing import Any

        Document = Any  # type: ignore

try:
    from ...utils.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


class TextSplitter:
    """
    Text Splitter 팩토리

    **beanllm 방식: 스마트 기본값 + 쉬운 전략 선택!**

    Example:
        ```python
        from beanllm.domain.splitters import TextSplitter

        # 방법 1: 가장 간단 (자동 최적화)
        chunks = TextSplitter.split(documents)

        # 방법 2: 전략을 쉽게 선택 (추천!)
        chunks = TextSplitter.recursive(chunk_size=1000).split_documents(docs)
        chunks = TextSplitter.character(separator="\\n\\n").split_documents(docs)
        chunks = TextSplitter.token(chunk_size=500).split_documents(docs)

        # 방법 3: 구분자만 지정 (자동 전략 선택)
        chunks = TextSplitter.split(docs, separator="\\n\\n")
        chunks = TextSplitter.split(docs, separators=["\\n\\n", "\\n"])

        # 방법 4: 전략 문자열 지정
        chunks = TextSplitter.split(docs, strategy="recursive")
        ```
    """

    # 전략별 Splitter 매핑
    SPLITTERS = {
        "character": CharacterTextSplitter,
        "recursive": RecursiveCharacterTextSplitter,
        "token": TokenTextSplitter,
        "markdown": MarkdownHeaderTextSplitter,
    }

    @classmethod
    def split(
        cls,
        documents: List["Document"],
        strategy: str = "recursive",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: Optional[str] = None,
        separators: Optional[List[str]] = None,
        **kwargs,
    ) -> List["Document"]:
        """
        문서 분할 (스마트 기본값 + 편리한 커스터마이징)

        Args:
            documents: 분할할 문서
            strategy: 분할 전략 ("recursive", "character", "token", "markdown")
            chunk_size: 청크 크기
            chunk_overlap: 청크 간 겹침
            separator: 단일 구분자 (character 전략용, 편의 기능)
            separators: 구분자 리스트 (recursive 전략용, 편의 기능)
            **kwargs: 전략별 추가 파라미터

        Returns:
            분할된 문서 리스트

        Example:
            ```python
            # 기본 (스마트 기본값)
            chunks = TextSplitter.split(docs)

            # 단일 구분자 지정 (간단!)
            chunks = TextSplitter.split(docs, separator="\\n\\n")

            # 여러 구분자 지정 (간단!)
            chunks = TextSplitter.split(docs, separators=["\\n\\n", "\\n", ". "])

            # 전략 + 구분자
            chunks = TextSplitter.split(
                docs,
                strategy="character",
                separator="\\n\\n"
            )
            ```
        """
        # separator/separators 편의 파라미터 처리
        if separator is not None:
            # 단일 구분자 → character 전략으로 자동 전환
            if strategy == "recursive":
                strategy = "character"
            kwargs["separator"] = separator

        if separators is not None:
            # 여러 구분자 → recursive 전략 (또는 유지)
            if strategy == "character":
                strategy = "recursive"
            kwargs["separators"] = separators

        splitter = cls.create(
            strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs
        )

        return splitter.split_documents(documents)

    @classmethod
    def create(
        cls, strategy: str = "recursive", chunk_size: int = 1000, chunk_overlap: int = 200, **kwargs
    ) -> BaseTextSplitter:
        """
        Splitter 생성

        Args:
            strategy: 분할 전략
            chunk_size: 청크 크기
            chunk_overlap: 청크 간 겹침
            **kwargs: 전략별 추가 파라미터

        Returns:
            TextSplitter 인스턴스
        """
        if strategy not in cls.SPLITTERS:
            logger.warning(f"Unknown strategy: {strategy}, using 'recursive'")
            strategy = "recursive"

        splitter_class = cls.SPLITTERS[strategy]

        # 마크다운은 다른 인터페이스
        if strategy == "markdown":
            return splitter_class(**kwargs)

        return splitter_class(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)

    # 전략별 팩토리 메서드 (쉬운 사용!)

    @classmethod
    def recursive(
        cls,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        **kwargs,
    ) -> RecursiveCharacterTextSplitter:
        """
        Recursive 전략 (권장, 가장 똑똑함)

        계층적 구분자로 자연스럽게 분할

        Args:
            chunk_size: 청크 크기
            chunk_overlap: 청크 간 겹침
            separators: 구분자 우선순위 (None이면 기본값)
            **kwargs: 추가 파라미터

        Returns:
            RecursiveCharacterTextSplitter 인스턴스

        Example:
            ```python
            # 기본값 사용
            splitter = TextSplitter.recursive()
            chunks = splitter.split_documents(docs)

            # 크기 조정
            splitter = TextSplitter.recursive(chunk_size=500, chunk_overlap=50)

            # 커스텀 구분자
            splitter = TextSplitter.recursive(
                separators=["\\n\\n", "\\n", ". "]
            )
            ```
        """
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators, **kwargs
        )

    @classmethod
    def character(
        cls, separator: str = "\n\n", chunk_size: int = 1000, chunk_overlap: int = 200, **kwargs
    ) -> CharacterTextSplitter:
        """
        Character 전략 (단순, 빠름)

        단일 구분자로 분할

        Args:
            separator: 구분자
            chunk_size: 청크 크기
            chunk_overlap: 청크 간 겹침
            **kwargs: 추가 파라미터

        Returns:
            CharacterTextSplitter 인스턴스

        Example:
            ```python
            # 단락으로 분할
            splitter = TextSplitter.character(separator="\\n\\n")

            # 줄로 분할
            splitter = TextSplitter.character(separator="\\n", chunk_size=500)

            # 커스텀 구분자
            splitter = TextSplitter.character(separator="---")
            ```
        """
        return CharacterTextSplitter(
            separator=separator, chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs
        )

    @classmethod
    def token(
        cls,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        encoding_name: str = "cl100k_base",
        model_name: Optional[str] = None,
        **kwargs,
    ) -> TokenTextSplitter:
        """
        Token 전략 (정확한 토큰 수 제어)

        LLM 컨텍스트 제한에 맞춰 토큰 기반 분할

        Args:
            chunk_size: 토큰 단위 청크 크기
            chunk_overlap: 토큰 단위 겹침
            encoding_name: tiktoken 인코딩 이름
            model_name: 모델 이름 (encoding_name 대신)
            **kwargs: 추가 파라미터

        Returns:
            TokenTextSplitter 인스턴스

        Example:
            ```python
            # GPT-4용 (기본)
            splitter = TextSplitter.token(chunk_size=1000)

            # 특정 모델용
            splitter = TextSplitter.token(
                model_name="gpt-3.5-turbo",
                chunk_size=2000
            )

            # 커스텀 인코딩
            splitter = TextSplitter.token(
                encoding_name="p50k_base",
                chunk_size=500
            )
            ```
        """
        return TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=encoding_name,
            model_name=model_name,
            **kwargs,
        )

    @classmethod
    def markdown(
        cls,
        headers_to_split_on: Optional[List[tuple[str, str]]] = None,
        return_each_line: bool = False,
        **kwargs,
    ) -> MarkdownHeaderTextSplitter:
        """
        Markdown 전략 (헤더 기준 분할)

        마크다운 헤더를 기준으로 분할

        Args:
            headers_to_split_on: (헤더, 메타데이터키) 튜플 리스트
            return_each_line: 각 줄을 별도 Document로 반환
            **kwargs: 추가 파라미터

        Returns:
            MarkdownHeaderTextSplitter 인스턴스

        Example:
            ```python
            # 기본 헤더 (H1, H2, H3)
            splitter = TextSplitter.markdown()

            # 커스텀 헤더
            splitter = TextSplitter.markdown(
                headers_to_split_on=[
                    ("#", "Title"),
                    ("##", "Section"),
                    ("###", "Subsection"),
                ]
            )
            ```
        """
        # 기본 헤더
        if headers_to_split_on is None:
            headers_to_split_on = [
                ("#", "Header1"),
                ("##", "Header2"),
                ("###", "Header3"),
            ]

        return MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, return_each_line=return_each_line, **kwargs
        )


# 편의 함수
def split_documents(
    documents: List["Document"],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    strategy: str = "recursive",
    separator: Optional[str] = None,
    separators: Optional[List[str]] = None,
    **kwargs,
) -> List["Document"]:
    """
    문서 분할 편의 함수

    Args:
        documents: 분할할 문서
        chunk_size: 청크 크기
        chunk_overlap: 청크 간 겹침
        strategy: 분할 전략
        separator: 단일 구분자 (간편 사용)
        separators: 구분자 리스트 (간편 사용)
        **kwargs: 추가 파라미터

    Example:
        ```python
        from beanllm.domain.splitters import split_documents

        # 가장 간단
        chunks = split_documents(docs)

        # 구분자 지정 (편리!)
        chunks = split_documents(docs, separator="\\n\\n")
        chunks = split_documents(docs, separators=["\\n\\n", "\\n"])

        # 전략 + 커스터마이징
        chunks = split_documents(docs, chunk_size=500, strategy="token")
        ```
    """
    return TextSplitter.split(
        documents=documents,
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=separator,
        separators=separators,
        **kwargs,
    )
