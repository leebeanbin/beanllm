"""
Qwen3 embedding implementation.
"""

from __future__ import annotations

from typing import List

from beanllm.domain.embeddings.base import BaseLocalEmbedding

try:
    from beanllm.utils.logging import get_logger
except ImportError:
    import logging

    def get_logger(name: str) -> logging.Logger:  # type: ignore[misc]
        return logging.getLogger(name)


logger = get_logger(__name__)


class Qwen3Embedding(BaseLocalEmbedding):
    """
    Qwen3-Embedding - Alibaba의 최신 임베딩 모델 (2025년)

    Qwen3-Embedding 특징:
    - Alibaba Cloud의 최신 임베딩 모델 (2025년 1월 출시)
    - 8B 파라미터 (대규모 성능)
    - 다국어 지원 (영어, 중국어, 일본어, 한국어 등)
    - MTEB 벤치마크 상위권
    - 긴 컨텍스트 지원 (8192 토큰)

    지원 모델:
    - Qwen/Qwen3-Embedding-8B: 메인 모델 (8B 파라미터)
    - Qwen/Qwen3-Embedding-1.5B: 경량 모델

    Example:
        ```python
        from beanllm.domain.embeddings import Qwen3Embedding

        # Qwen3-Embedding-8B 사용
        emb = Qwen3Embedding(model="Qwen/Qwen3-Embedding-8B", use_gpu=True)
        vectors = emb.embed_sync(["텍스트 1", "텍스트 2"])

        # 경량 모델 사용
        emb = Qwen3Embedding(model="Qwen/Qwen3-Embedding-1.5B")
        vectors = emb.embed_sync(["text"])
        ```

    References:
        - https://huggingface.co/Qwen/Qwen3-Embedding-8B
        - https://qwenlm.github.io/
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-Embedding-8B",
        use_gpu: bool = True,
        normalize: bool = True,
        batch_size: int = 16,
        **kwargs,
    ):
        """
        Args:
            model: Qwen3 모델 이름 (Qwen/Qwen3-Embedding-8B 또는 1.5B)
            use_gpu: GPU 사용 여부 (기본: True)
            normalize: 임베딩 정규화 여부 (기본: True)
            batch_size: 배치 크기 (기본: 16, 8B 모델용)
            **kwargs: 추가 파라미터
        """
        super().__init__(model, use_gpu, **kwargs)

        self.normalize = normalize
        self.batch_size = batch_size

    def _load_model(self):
        """모델 로딩 (lazy loading, 분산 락 적용)"""

        def _load_impl():
            # Import 검증
            self._validate_import("sentence_transformers", "sentence-transformers")

            from sentence_transformers import SentenceTransformer

            # Device 설정
            self._device = self._get_device()

            logger.info(f"Loading Qwen3 model: {self.model} on {self._device}")

            # 모델 로드
            self._model = SentenceTransformer(self.model, device=self._device)

            logger.info(
                f"Qwen3 model loaded: {self.model} (max_seq_length: {self._model.max_seq_length})"
            )

        # 분산 락을 사용한 모델 로딩
        self._load_model_with_lock(self.model, _load_impl)

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (동기)"""
        self._load_model()

        try:
            # Sentence Transformers로 임베딩
            embeddings = self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            self._log_embed_success(len(texts), f"shape: {embeddings.shape}")

            return embeddings.tolist()

        except Exception as e:
            self._handle_embed_error("Qwen3", e)
