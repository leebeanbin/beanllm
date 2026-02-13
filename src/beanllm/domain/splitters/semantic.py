"""
Semantic Text Splitter - 의미 기반 텍스트 분할

임베딩을 사용하여 의미적으로 유사한 문장들을 그룹화하고,
의미가 급격히 변하는 지점에서 분할합니다.

Requirements:
    pip install sentence-transformers  # 또는 semchunk

Example:
    ```python
    from beanllm.domain.splitters import SemanticTextSplitter

    splitter = SemanticTextSplitter(
        model="all-MiniLM-L6-v2",
        threshold=0.5,
        min_chunk_size=100,
        max_chunk_size=1000
    )
    chunks = splitter.split_text(text)
    ```
"""

from typing import TYPE_CHECKING, Any, Callable, List, Optional, cast

try:
    from beanllm.utils.constants import DEFAULT_CHUNK_SIZE
except ImportError:
    DEFAULT_CHUNK_SIZE = 1000

from beanllm.domain.splitters.base import BaseTextSplitter
from beanllm.domain.splitters.coherence import CoherenceTextSplitter
from beanllm.domain.splitters.semantic_chunking import force_split_by_size, merge_small_chunks
from beanllm.domain.splitters.semantic_preprocessing import split_into_sentences
from beanllm.domain.splitters.semantic_similarity import find_breakpoints

if TYPE_CHECKING:
    from beanllm.domain.loaders.types import Document
else:
    try:
        from beanllm.domain.loaders.types import Document
    except ImportError:
        Document = Any  # type: ignore

try:
    from beanllm.utils.logging import get_logger
except ImportError:
    import logging

    def get_logger(name: str) -> "logging.Logger":  # type: ignore[misc]
        return logging.getLogger(name)


logger = get_logger(__name__)


class SemanticTextSplitter(BaseTextSplitter):
    """
    의미 기반 텍스트 분할기

    임베딩을 사용하여 문장 간 유사도를 계산하고,
    유사도가 급격히 떨어지는 지점에서 분할합니다.

    Approach:
    1. 텍스트를 문장 단위로 분할
    2. 각 문장의 임베딩 계산
    3. 인접 문장 간 코사인 유사도 계산
    4. 유사도가 임계값 이하인 지점에서 분할
    5. 최소/최대 청크 크기 제약 적용
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = DEFAULT_CHUNK_SIZE,
        buffer_size: int = 1,
        embedding_function: Optional[Callable[[str], List[float]]] = None,
        use_semchunk: bool = False,
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        **kwargs: Any,
    ):
        super().__init__(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        self.model_name = model
        self.threshold = threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.buffer_size = buffer_size
        self.use_semchunk = use_semchunk
        self.semchunk_size = chunk_size
        self.kwargs = kwargs

        self._embedding_function = embedding_function
        self._model = None

        if use_semchunk:
            self._init_semchunk()
        elif embedding_function is None:
            self._init_sentence_transformer()

    def _init_sentence_transformer(self) -> None:
        """sentence-transformers 모델 초기화"""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading sentence-transformers model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            model = self._model
            assert model is not None  # SentenceTransformer() 이후 항상 non-None
            self._embedding_function = lambda text: model.encode(
                text, convert_to_numpy=True
            ).tolist()
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for SemanticTextSplitter. "
                "Install it with: pip install sentence-transformers\n"
                "Or use semchunk: pip install semchunk"
            )

    def _init_semchunk(self) -> None:
        """semchunk 라이브러리 초기화"""
        try:
            import semchunk

            logger.info("Using semchunk for semantic chunking")
            self._chunker = semchunk.chunkerify(self.model_name, chunk_size=self.semchunk_size)
        except ImportError:
            raise ImportError(
                "semchunk is required when use_semchunk=True. "
                "Install it with: pip install semchunk"
            )

    def _get_sentence_embeddings(self, sentences: List[str]) -> List[List[float]]:
        """문장 임베딩 계산"""
        if self._model is not None:
            embeddings = self._model.encode(sentences, convert_to_numpy=True)
            return embeddings.tolist()
        assert self._embedding_function is not None, "embedding_function is required"
        return [self._embedding_function(s) for s in sentences]

    def split_text(self, text: str) -> List[str]:
        """의미 기반 텍스트 분할"""
        if self.use_semchunk:
            return cast(List[str], self._chunker(text))

        sentences = split_into_sentences(text)

        if len(sentences) <= 1:
            return [text] if text.strip() else []

        logger.info(f"Split into {len(sentences)} sentences")

        embeddings = self._get_sentence_embeddings(sentences)
        breakpoints = find_breakpoints(embeddings, self.threshold, self.buffer_size)

        logger.info(f"Found {len(breakpoints)} semantic breakpoints")

        chunks: List[str] = []
        start = 0

        for bp in breakpoints:
            chunk_sentences = sentences[start:bp]
            chunk = " ".join(chunk_sentences)

            if len(chunk) > self.max_chunk_size:
                chunks.extend(force_split_by_size(chunk, self.max_chunk_size))
            else:
                chunks.append(chunk)
            start = bp

        if start < len(sentences):
            chunk = " ".join(sentences[start:])
            if len(chunk) > self.max_chunk_size:
                chunks.extend(force_split_by_size(chunk, self.max_chunk_size))
            else:
                chunks.append(chunk)

        chunks = merge_small_chunks(chunks, self.min_chunk_size, self.max_chunk_size)

        logger.info(f"Final chunks: {len(chunks)}")
        return chunks

    def split_documents(self, documents: List["Document"]) -> List["Document"]:
        """문서 분할"""
        from beanllm.domain.loaders.types import Document

        result = []
        for doc in documents:
            chunks = self.split_text(doc.content)
            for i, chunk in enumerate(chunks):
                metadata = doc.metadata.copy()
                metadata["chunk"] = i
                metadata["total_chunks"] = len(chunks)
                metadata["splitter"] = "semantic"
                result.append(Document(content=chunk, metadata=metadata))
        return result

    def __repr__(self) -> str:
        if self.use_semchunk:
            return f"SemanticTextSplitter(semchunk, chunk_size={self.semchunk_size})"
        return (
            f"SemanticTextSplitter("
            f"model={self.model_name}, "
            f"threshold={self.threshold}, "
            f"min={self.min_chunk_size}, "
            f"max={self.max_chunk_size})"
        )


__all__ = ["SemanticTextSplitter", "CoherenceTextSplitter"]
