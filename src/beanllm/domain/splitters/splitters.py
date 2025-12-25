"""
Splitters Implementations - 텍스트 분할 구현체들
"""

from typing import TYPE_CHECKING, Callable, List, Optional

from .base import BaseTextSplitter

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


class CharacterTextSplitter(BaseTextSplitter):
    """
    단순 문자 기반 분할

    Example:
        ```python
        from beanllm.domain.splitters import CharacterTextSplitter

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
        keep_separator: bool = False,
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
        from beanllm.domain.splitters import RecursiveCharacterTextSplitter

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
        keep_separator: bool = True,
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
            "\n",  # 줄
            ". ",  # 문장
            " ",  # 단어
            "",  # 문자
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
                new_separators = self.separators[i + 1 :]
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
                        keep_separator=self.keep_separator,
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
        from beanllm.domain.splitters import TokenTextSplitter

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
        chunk_overlap: int = 200,
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
                "tiktoken is required for TokenTextSplitter. Install it with: pip install tiktoken"
            )

        if model_name:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        else:
            self.tokenizer = tiktoken.get_encoding(encoding_name)

        super().__init__(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=self._token_length
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
        from beanllm.domain.splitters import MarkdownHeaderTextSplitter

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

    def __init__(self, headers_to_split_on: List[tuple[str, str]], return_each_line: bool = False):
        """
        Args:
            headers_to_split_on: (마크다운 헤더, 메타데이터 키) 튜플 리스트
            return_each_line: 각 줄을 별도 Document로 반환
        """
        self.headers_to_split_on = headers_to_split_on
        self.return_each_line = return_each_line

    def split_text(self, text: str) -> List["Document"]:
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
                        chunks.append(
                            Document(
                                content="\n".join(current_chunk), metadata=current_metadata.copy()
                            )
                        )
                        current_chunk = []

                    # 메타데이터 업데이트
                    current_metadata[name] = line.replace(header + " ", "").strip()
                    header_found = True
                    break

            if not header_found:
                current_chunk.append(line)

                if self.return_each_line and line.strip():
                    chunks.append(Document(content=line, metadata=current_metadata.copy()))

        # 마지막 청크
        if current_chunk and not self.return_each_line:
            chunks.append(
                Document(content="\n".join(current_chunk), metadata=current_metadata.copy())
            )

        return chunks

    def split_documents(self, documents: List["Document"]) -> List["Document"]:
        """문서 분할"""
        all_chunks = []
        for doc in documents:
            chunks = self.split_text(doc.content)
            # 원본 메타데이터 병합
            for chunk in chunks:
                chunk.metadata.update(doc.metadata)
            all_chunks.extend(chunks)

        return all_chunks
