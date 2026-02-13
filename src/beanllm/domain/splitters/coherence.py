"""
Coherence Text Splitter - 일관성 기반 텍스트 분할

문장 간 주제 일관성을 K-Means 클러스터링으로 분석하여 분할합니다.
"""

import re
from typing import Any, List

try:
    from beanllm.utils.constants import DEFAULT_CHUNK_SIZE
except ImportError:
    DEFAULT_CHUNK_SIZE = 1000

from beanllm.domain.splitters.base import BaseTextSplitter

try:
    from beanllm.utils.logging import get_logger
except ImportError:
    import logging

    def get_logger(name: str) -> "logging.Logger":  # type: ignore[misc]
        return logging.getLogger(name)


logger = get_logger(__name__)


class CoherenceTextSplitter(BaseTextSplitter):
    """
    일관성 기반 텍스트 분할기

    문장 간 주제 일관성을 분석하여 분할합니다.
    클러스터링을 사용하여 유사한 문장들을 그룹화합니다.

    Example:
        ```python
        splitter = CoherenceTextSplitter(
            model="all-MiniLM-L6-v2",
            num_clusters="auto",
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

    def _init_model(self) -> None:
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

        sentences = re.split(r"(?<=[.!?])\s+|(?<=\n)\s*", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 1:
            return [text] if text.strip() else []

        embeddings = self._model.encode(sentences, convert_to_numpy=True)

        if self.num_clusters == "auto":
            n_clusters = max(2, int(np.sqrt(len(sentences))))
        else:
            n_clusters = min(self.num_clusters, len(sentences))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        chunks: List[str] = []
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

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        final_chunks: List[str] = []
        for chunk in chunks:
            if len(chunk) > self.max_chunk_size:
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
