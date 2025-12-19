"""
Text Splitters - Smart Defaults, Pythonic
llmkit 방식: 자동 최적화 + 간단한 API
"""
from typing import List, Optional, Callable, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .document_loaders import Document
from .utils.logger import get_logger

logger = get_logger(__name__)


class BaseTextSplitter(ABC):
    """Text Splitter 베이스 클래스"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        keep_separator: bool = True
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

    def split_documents(self, documents: List[Document]) -> List[Document]:
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
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """
        텍스트에서 문서 생성

        Args:
            texts: 텍스트 리스트
            metadatas: 메타데이터 리스트

        Returns:
            문서 리스트
        """
        _metadatas = metadatas or [{}] * len(texts)
        documents = []

        for i, text in enumerate(texts):
            index = 0
            for chunk in self.split_text(text):
                metadata = _metadatas[i].copy()
                metadata["chunk"] = index
                documents.append(Document(
                    content=chunk,
                    metadata=metadata
                ))
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


class CharacterTextSplitter(BaseTextSplitter):
    """
    단순 문자 기반 분할

    Example:
        ```python
        from llmkit.text_splitters import CharacterTextSplitter

        splitter = CharacterTextSplitter(
            separator="\\n\\n",
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_text(text)
        ```
    """

    def __init__(
        self,
        separator: str = "\n\n",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        keep_separator: bool = False
    ):
        """
        Args:
            separator: 구분자
            chunk_size: 청크 크기
            chunk_overlap: 청크 간 겹침
            length_function: 길이 계산 함수
            keep_separator: 구분자 유지 여부
        """
        super().__init__(chunk_size, chunk_overlap, length_function, keep_separator)
        self.separator = separator

    def split_text(self, text: str) -> List[str]:
        """텍스트 분할"""
        if self.separator:
            splits = text.split(self.separator)
        else:
            splits = list(text)

        return self._merge_splits(splits, self.separator)


class RecursiveCharacterTextSplitter(BaseTextSplitter):
    """
    재귀적 문자 분할 (가장 권장)

    계층적 구분자를 사용해 자연스럽게 분할

    Example:
        ```python
        from llmkit.text_splitters import RecursiveCharacterTextSplitter

        # 기본 구분자 (스마트!)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)

        # 커스텀 구분자
        splitter = RecursiveCharacterTextSplitter(
            separators=["\\n\\n", "\\n", ". ", " ", ""],
            chunk_size=500
        )
        ```
    """

    def __init__(
        self,
        separators: Optional[List[str]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        keep_separator: bool = True
    ):
        """
        Args:
            separators: 구분자 우선순위 (None이면 기본값)
            chunk_size: 청크 크기
            chunk_overlap: 청크 간 겹침
            length_function: 길이 계산 함수
            keep_separator: 구분자 유지 여부
        """
        super().__init__(chunk_size, chunk_overlap, length_function, keep_separator)

        # 스마트 기본값
        self.separators = separators or [
            "\n\n",  # 단락
            "\n",    # 줄
            ". ",    # 문장
            " ",     # 단어
            ""       # 문자
        ]

    def split_text(self, text: str) -> List[str]:
        """재귀적 분할"""
        final_chunks = []

        # 적절한 구분자 찾기
        separator = self.separators[-1]
        new_separators = []

        for i, _separator in enumerate(self.separators):
            if _separator == "":
                separator = _separator
                break

            if _separator in text:
                separator = _separator
                new_separators = self.separators[i + 1:]
                break

        # 분할
        splits = text.split(separator) if separator else list(text)

        # 구분자 유지
        if self.keep_separator and separator:
            splits = [
                (split + separator if i < len(splits) - 1 else split)
                for i, split in enumerate(splits)
            ]

        # 병합
        good_splits = []
        for split in splits:
            if self.length_function(split) < self.chunk_size:
                good_splits.append(split)
            else:
                # 너무 크면 재귀적으로 분할
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []

                # 재귀
                if new_separators:
                    other_splitter = RecursiveCharacterTextSplitter(
                        separators=new_separators,
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                        length_function=self.length_function,
                        keep_separator=self.keep_separator
                    )
                    final_chunks.extend(other_splitter.split_text(split))
                else:
                    # 더 이상 구분자 없으면 강제 분할
                    final_chunks.extend(self._split_by_size(split))

        # 남은 것 병합
        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return final_chunks

    def _split_by_size(self, text: str) -> List[str]:
        """크기로 강제 분할"""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end - self.chunk_overlap

        return chunks


class TokenTextSplitter(BaseTextSplitter):
    """
    토큰 기반 분할

    Example:
        ```python
        from llmkit.text_splitters import TokenTextSplitter

        # OpenAI 토큰 기준
        splitter = TokenTextSplitter(
            encoding_name="cl100k_base",  # GPT-4
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_text(text)
        ```
    """

    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        model_name: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Args:
            encoding_name: tiktoken 인코딩 이름
            model_name: 모델 이름 (encoding_name 대신)
            chunk_size: 토큰 단위 청크 크기
            chunk_overlap: 토큰 단위 겹침
        """
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken is required for TokenTextSplitter. "
                "Install it with: pip install tiktoken"
            )

        if model_name:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        else:
            self.tokenizer = tiktoken.get_encoding(encoding_name)

        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._token_length
        )

    def _token_length(self, text: str) -> int:
        """토큰 길이 계산"""
        return len(self.tokenizer.encode(text))

    def split_text(self, text: str) -> List[str]:
        """토큰 기준 분할"""
        tokens = self.tokenizer.encode(text)
        chunks = []
        start = 0

        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

            start = end - self.chunk_overlap

        return chunks


class MarkdownHeaderTextSplitter:
    """
    마크다운 헤더 기준 분할

    Example:
        ```python
        from llmkit.text_splitters import MarkdownHeaderTextSplitter

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
        )
        chunks = splitter.split_text(markdown_text)
        ```
    """

    def __init__(
        self,
        headers_to_split_on: List[tuple[str, str]],
        return_each_line: bool = False
    ):
        """
        Args:
            headers_to_split_on: (마크다운 헤더, 메타데이터 키) 튜플 리스트
            return_each_line: 각 줄을 별도 Document로 반환
        """
        self.headers_to_split_on = headers_to_split_on
        self.return_each_line = return_each_line

    def split_text(self, text: str) -> List[Document]:
        """마크다운 분할"""
        lines = text.split("\n")
        chunks = []
        current_chunk = []
        current_metadata = {}

        for line in lines:
            # 헤더 체크
            header_found = False
            for header, name in self.headers_to_split_on:
                if line.startswith(header + " "):
                    # 이전 청크 저장
                    if current_chunk:
                        chunks.append(Document(
                            content="\n".join(current_chunk),
                            metadata=current_metadata.copy()
                        ))
                        current_chunk = []

                    # 메타데이터 업데이트
                    current_metadata[name] = line.replace(header + " ", "").strip()
                    header_found = True
                    break

            if not header_found:
                current_chunk.append(line)

                if self.return_each_line and line.strip():
                    chunks.append(Document(
                        content=line,
                        metadata=current_metadata.copy()
                    ))

        # 마지막 청크
        if current_chunk and not self.return_each_line:
            chunks.append(Document(
                content="\n".join(current_chunk),
                metadata=current_metadata.copy()
            ))

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """문서 분할"""
        all_chunks = []
        for doc in documents:
            chunks = self.split_text(doc.content)
            # 원본 메타데이터 병합
            for chunk in chunks:
                chunk.metadata.update(doc.metadata)
            all_chunks.extend(chunks)

        return all_chunks


class TextSplitter:
    """
    Text Splitter 팩토리

    **llmkit 방식: 스마트 기본값 + 쉬운 전략 선택!**

    Example:
        ```python
        from llmkit.text_splitters import TextSplitter

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
        "markdown": MarkdownHeaderTextSplitter
    }

    @classmethod
    def split(
        cls,
        documents: List[Document],
        strategy: str = "recursive",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: Optional[str] = None,
        separators: Optional[List[str]] = None,
        **kwargs
    ) -> List[Document]:
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
            kwargs['separator'] = separator

        if separators is not None:
            # 여러 구분자 → recursive 전략 (또는 유지)
            if strategy == "character":
                strategy = "recursive"
            kwargs['separators'] = separators

        splitter = cls.create(
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )

        return splitter.split_documents(documents)

    @classmethod
    def create(
        cls,
        strategy: str = "recursive",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs
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

        return splitter_class(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )

    # 전략별 팩토리 메서드 (쉬운 사용!)

    @classmethod
    def recursive(
        cls,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        **kwargs
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
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            **kwargs
        )

    @classmethod
    def character(
        cls,
        separator: str = "\n\n",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs
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
            separator=separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )

    @classmethod
    def token(
        cls,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        encoding_name: str = "cl100k_base",
        model_name: Optional[str] = None,
        **kwargs
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
            **kwargs
        )

    @classmethod
    def markdown(
        cls,
        headers_to_split_on: Optional[List[tuple[str, str]]] = None,
        return_each_line: bool = False,
        **kwargs
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
            headers_to_split_on=headers_to_split_on,
            return_each_line=return_each_line,
            **kwargs
        )


# 편의 함수
def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    strategy: str = "recursive",
    separator: Optional[str] = None,
    separators: Optional[List[str]] = None,
    **kwargs
) -> List[Document]:
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
        from llmkit.text_splitters import split_documents

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
        **kwargs
    )
