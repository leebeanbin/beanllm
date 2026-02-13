"""
Semantic Text Splitter - 의미 기반 텍스트 분할

임베딩을 사용하여 의미적으로 유사한 문장들을 그룹화하고,
의미가 급격히 변하는 지점에서 분할합니다.

Requirements:
    pip install sentence-transformers  # 또는 semchunk

References:
    - Semantic Chunking: https://github.com/isaacus-dev/semchunk
    - Greg Kamradt's approach: https://github.com/FullStackRetrieval-com/RetrievalTutorials

Example:
    ```python
    from beanllm.domain.splitters import SemanticTextSplitter

    # 기본 사용 (sentence-transformers)
    from beanllm.utils.constants import DEFAULT_CHUNK_SIZE
    splitter = SemanticTextSplitter(
        model="all-MiniLM-L6-v2",
        threshold=0.5,
        min_chunk_size=100,
        max_chunk_size=DEFAULT_CHUNK_SIZE
    )
    chunks = splitter.split_text(text)

    # semchunk 라이브러리 사용
    splitter = SemanticTextSplitter(
        use_semchunk=True,
        chunk_size=512
    )
    chunks = splitter.split_text(text)
    ```
"""

import logging
import re
from typing import TYPE_CHECKING, Any, Callable, List, Optional, cast

try:
    from beanllm.utils.constants import DEFAULT_CHUNK_SIZE
except ImportError:
    DEFAULT_CHUNK_SIZE = 1000

from beanllm.domain.splitters.base import BaseTextSplitter

if TYPE_CHECKING:
    from beanllm.domain.loaders.types import Document
else:
    try:
        from beanllm.domain.loaders.types import Document
    except ImportError:
        from typing import Any

        Document = Any  # type: ignore

try:
    from beanllm.utils.logging import get_logger
except ImportError:

    def get_logger(name: str) -> logging.Logger:  # type: ignore[misc]
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

    Attributes:
        model: 임베딩 모델 이름 또는 인스턴스
        threshold: 분할 임계값 (0-1, 낮을수록 더 자주 분할)
        min_chunk_size: 최소 청크 크기 (문자 수)
        max_chunk_size: 최대 청크 크기 (문자 수)
        use_semchunk: semchunk 라이브러리 사용 여부

    Example:
        ```python
        # 낮은 임계값 = 더 많은 분할 (세밀한 청크)
        splitter = SemanticTextSplitter(threshold=0.3)

        # 높은 임계값 = 적은 분할 (큰 청크)
        splitter = SemanticTextSplitter(threshold=0.7)

        # 커스텀 임베딩 함수
        from beanllm.domain.embeddings import OpenAIEmbedding

        embed_fn = OpenAIEmbedding().embed_query
        splitter = SemanticTextSplitter(embedding_function=embed_fn)
        ```
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
        """
        Semantic Splitter 초기화

        Args:
            model: sentence-transformers 모델 이름
            threshold: 분할 임계값 (0-1)
                - 0.3: 세밀한 분할 (많은 청크)
                - 0.5: 기본값 (균형)
                - 0.7: 느슨한 분할 (적은 청크)
            min_chunk_size: 최소 청크 크기 (문자 수)
            max_chunk_size: 최대 청크 크기 (문자 수)
            buffer_size: 유사도 계산 시 인접 문장 수
            embedding_function: 커스텀 임베딩 함수 (str -> List[float])
            use_semchunk: semchunk 라이브러리 사용 여부
            chunk_size: semchunk용 청크 크기 (토큰 수)
            chunk_overlap: 청크 간 겹침 (미사용, 호환성)
            **kwargs: 추가 옵션
        """
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

        # 임베딩 함수 설정
        self._embedding_function = embedding_function
        self._model = None  # Lazy loading

        if use_semchunk:
            self._init_semchunk()
        elif embedding_function is None:
            self._init_sentence_transformer()

    def _init_sentence_transformer(self):
        """sentence-transformers 모델 초기화"""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading sentence-transformers model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self._embedding_function = lambda text: self._model.encode(
                text, convert_to_numpy=True
            ).tolist()
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for SemanticTextSplitter. "
                "Install it with: pip install sentence-transformers\n"
                "Or use semchunk: pip install semchunk"
            )

    def _init_semchunk(self):
        """semchunk 라이브러리 초기화"""
        try:
            import semchunk

            logger.info("Using semchunk for semantic chunking")
            self._chunker = semchunk.chunkerify(self.model_name, chunk_size=self.semchunk_size)
        except ImportError:
            raise ImportError(
                "semchunk is required when use_semchunk=True. Install it with: pip install semchunk"
            )

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        텍스트를 문장 단위로 분할

        Args:
            text: 분할할 텍스트

        Returns:
            문장 리스트
        """
        # 간단한 문장 분할 (마침표, 느낌표, 물음표)
        # 줄바꿈도 문장 경계로 처리
        sentence_endings = r"(?<=[.!?])\s+|(?<=\n)\s*"
        sentences = re.split(sentence_endings, text)

        # 빈 문장 제거
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _compute_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """코사인 유사도 계산"""
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def _get_sentence_embeddings(self, sentences: List[str]) -> List[List[float]]:
        """문장 임베딩 계산"""
        if self._model is not None:
            # sentence-transformers 배치 처리
            embeddings = self._model.encode(sentences, convert_to_numpy=True)
            return embeddings.tolist()
        else:
            # 커스텀 함수 사용 (하나씩)
            assert self._embedding_function is not None, "embedding_function is required"
            return [self._embedding_function(s) for s in sentences]

    def _find_breakpoints(self, sentences: List[str], embeddings: List[List[float]]) -> List[int]:
        """
        분할 지점 찾기

        인접 문장 간 유사도를 계산하고,
        임계값 이하인 지점을 분할 지점으로 선택합니다.

        Args:
            sentences: 문장 리스트
            embeddings: 임베딩 리스트

        Returns:
            분할 지점 인덱스 리스트
        """
        if len(sentences) <= 1:
            return []

        breakpoints = []
        similarities = []

        # 인접 문장 간 유사도 계산
        for i in range(len(sentences) - 1):
            # 버퍼를 사용한 평균 유사도
            left_start = max(0, i - self.buffer_size + 1)
            right_end = min(len(sentences), i + self.buffer_size + 1)

            # 왼쪽 그룹 평균 임베딩
            left_embeddings = embeddings[left_start : i + 1]
            left_avg = [
                sum(e[j] for e in left_embeddings) / len(left_embeddings)
                for j in range(len(left_embeddings[0]))
            ]

            # 오른쪽 그룹 평균 임베딩
            right_embeddings = embeddings[i + 1 : right_end]
            right_avg = [
                sum(e[j] for e in right_embeddings) / len(right_embeddings)
                for j in range(len(right_embeddings[0]))
            ]

            similarity = self._compute_cosine_similarity(left_avg, right_avg)
            similarities.append(similarity)

        # 임계값 이하인 지점 찾기
        for i, sim in enumerate(similarities):
            if sim < self.threshold:
                breakpoints.append(i + 1)  # 분할 지점은 다음 문장 시작

        return breakpoints

    def _merge_small_chunks(self, chunks: List[str], min_size: int, max_size: int) -> List[str]:
        """
        작은 청크들을 병합

        Args:
            chunks: 청크 리스트
            min_size: 최소 크기
            max_size: 최대 크기

        Returns:
            병합된 청크 리스트
        """
        merged = []
        current = ""

        for chunk in chunks:
            if len(current) + len(chunk) <= max_size:
                current += (" " if current else "") + chunk
            else:
                if current and len(current) >= min_size:
                    merged.append(current)
                current = chunk

        # 마지막 청크
        if current:
            if len(current) >= min_size:
                merged.append(current)
            elif merged:
                # 너무 작으면 이전 청크에 병합
                if len(merged[-1]) + len(current) <= max_size:
                    merged[-1] += " " + current
                else:
                    merged.append(current)  # 어쩔 수 없이 작은 청크
            else:
                merged.append(current)

        return merged

    def split_text(self, text: str) -> List[str]:
        """
        의미 기반 텍스트 분할

        Args:
            text: 분할할 텍스트

        Returns:
            분할된 청크 리스트
        """
        # semchunk 사용
        if self.use_semchunk:
            return cast(List[str], self._chunker(text))

        # 1. 문장 분할
        sentences = self._split_into_sentences(text)

        if len(sentences) <= 1:
            return [text] if text.strip() else []

        logger.info(f"Split into {len(sentences)} sentences")

        # 2. 임베딩 계산
        embeddings = self._get_sentence_embeddings(sentences)

        # 3. 분할 지점 찾기
        breakpoints = self._find_breakpoints(sentences, embeddings)

        logger.info(f"Found {len(breakpoints)} semantic breakpoints")

        # 4. 청크 생성
        chunks = []
        start = 0

        for bp in breakpoints:
            chunk_sentences = sentences[start:bp]
            chunk = " ".join(chunk_sentences)

            # 최대 크기 체크
            if len(chunk) > self.max_chunk_size:
                # 너무 크면 강제 분할
                chunks.extend(self._force_split(chunk))
            else:
                chunks.append(chunk)

            start = bp

        # 마지막 청크
        if start < len(sentences):
            chunk = " ".join(sentences[start:])
            if len(chunk) > self.max_chunk_size:
                chunks.extend(self._force_split(chunk))
            else:
                chunks.append(chunk)

        # 5. 작은 청크 병합
        chunks = self._merge_small_chunks(chunks, self.min_chunk_size, self.max_chunk_size)

        logger.info(f"Final chunks: {len(chunks)}")

        return chunks

    def _force_split(self, text: str) -> List[str]:
        """최대 크기 초과 시 강제 분할"""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.max_chunk_size

            # 단어 경계에서 분할
            if end < len(text):
                space_idx = text.rfind(" ", start, end)
                if space_idx > start:
                    end = space_idx

            chunks.append(text[start:end].strip())
            start = end

        return chunks

    def split_documents(self, documents: List["Document"]) -> List["Document"]:
        """
        문서 분할

        Args:
            documents: 분할할 문서 리스트

        Returns:
            분할된 문서 리스트
        """
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


class CoherenceTextSplitter(BaseTextSplitter):
    """
    일관성 기반 텍스트 분할기

    문장 간 주제 일관성을 분석하여 분할합니다.
    클러스터링을 사용하여 유사한 문장들을 그룹화합니다.

    Example:
        ```python
        splitter = CoherenceTextSplitter(
            model="all-MiniLM-L6-v2",
            num_clusters="auto",  # 자동 결정
            min_cluster_size=3
        )
        chunks = splitter.split_text(text)
        ```
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        num_clusters: int | str = "auto",
        min_cluster_size: int = 2,
        max_chunk_size: int = DEFAULT_CHUNK_SIZE,
        **kwargs: Any,
    ):
        """
        Coherence Splitter 초기화

        Args:
            model: 임베딩 모델 이름
            num_clusters: 클러스터 수 ("auto" 또는 정수)
            min_cluster_size: 최소 클러스터 크기
            max_chunk_size: 최대 청크 크기
            **kwargs: 추가 옵션
        """
        super().__init__(chunk_size=max_chunk_size, chunk_overlap=0)

        self.model_name = model
        self.num_clusters = num_clusters
        self.min_cluster_size = min_cluster_size
        self.max_chunk_size = max_chunk_size
        self.kwargs = kwargs

        self._model = None

    def _init_model(self):
        """모델 초기화"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )

    def split_text(self, text: str) -> List[str]:
        """
        일관성 기반 텍스트 분할

        Args:
            text: 분할할 텍스트

        Returns:
            분할된 청크 리스트
        """
        try:
            import numpy as np
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError(
                "scikit-learn is required for CoherenceTextSplitter. "
                "Install with: pip install scikit-learn"
            )

        self._init_model()
        assert self._model is not None

        # 1. 문장 분할
        sentences = re.split(r"(?<=[.!?])\s+|(?<=\n)\s*", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 1:
            return [text] if text.strip() else []

        # 2. 임베딩 계산
        embeddings = self._model.encode(sentences, convert_to_numpy=True)

        # 3. 클러스터 수 결정
        if self.num_clusters == "auto":
            # sqrt(n) 휴리스틱
            n_clusters = max(2, int(np.sqrt(len(sentences))))
        else:
            n_clusters = min(self.num_clusters, len(sentences))

        # 4. 클러스터링
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # 5. 연속된 같은 레이블을 그룹화
        chunks = []
        current_sentences = [sentences[0]]
        current_label = labels[0]

        for i in range(1, len(sentences)):
            if labels[i] == current_label:
                current_sentences.append(sentences[i])
            else:
                chunk = " ".join(current_sentences)
                chunks.append(chunk)
                current_sentences = [sentences[i]]
                current_label = labels[i]

        # 마지막 청크
        if current_sentences:
            chunks.append(" ".join(current_sentences))

        # 6. 최대 크기 체크
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.max_chunk_size:
                # 강제 분할
                for i in range(0, len(chunk), self.max_chunk_size):
                    final_chunks.append(chunk[i : i + self.max_chunk_size])
            else:
                final_chunks.append(chunk)

        logger.info(
            f"CoherenceTextSplitter: {len(sentences)} sentences -> "
            f"{len(final_chunks)} chunks (clusters={n_clusters})"
        )

        return final_chunks

    def __repr__(self) -> str:
        return f"CoherenceTextSplitter(model={self.model_name}, num_clusters={self.num_clusters})"
