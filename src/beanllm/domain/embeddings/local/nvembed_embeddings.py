"""
NVIDIA NV-Embed-v2 embedding implementation.
"""

from __future__ import annotations

from typing import List, Optional

from beanllm.domain.embeddings.base import BaseLocalEmbedding

try:
    from beanllm.utils.logging import get_logger
except ImportError:
    import logging

    def get_logger(name: str) -> logging.Logger:  # type: ignore[misc]
        return logging.getLogger(name)


logger = get_logger(__name__)


class NVEmbedEmbedding(BaseLocalEmbedding):
    """
    NVIDIA NV-Embed-v2 임베딩 (MTEB 1위, 2024-2025)

    NVIDIA의 최신 임베딩 모델로 MTEB 벤치마크 1위 (69.32)를 달성했습니다.

    성능:
    - MTEB Score: 69.32 (1위)
    - Retrieval: 60.92
    - Classification: 80.19
    - Clustering: 54.23
    - Pair Classification: 89.68
    - Reranking: 62.58
    - STS: 87.86

    Features:
    - Instruction-aware embedding
    - Passage 및 Query prefix 지원
    - Latent attention layer
    - 최대 32K 토큰 지원

    Example:
        ```python
        from beanllm.domain.embeddings import NVEmbedEmbedding

        # 기본 사용 (passage)
        emb = NVEmbedEmbedding(use_gpu=True)
        vectors = emb.embed_sync(["This is a passage."])

        # Query 임베딩
        emb = NVEmbedEmbedding(prefix="query")
        vectors = emb.embed_sync(["What is AI?"])

        # Instruction 사용
        emb = NVEmbedEmbedding(
            prefix="query",
            instruction="Retrieve relevant passages for the query"
        )
        vectors = emb.embed_sync(["machine learning"])
        ```
    """

    def __init__(
        self,
        model: str = "nvidia/NV-Embed-v2",
        use_gpu: bool = True,
        prefix: str = "passage",
        instruction: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = 32,
        **kwargs,
    ):
        """
        Args:
            model: NVIDIA NV-Embed 모델 이름
            use_gpu: GPU 사용 여부 (기본: True, 권장)
            prefix: "passage" 또는 "query" (기본: "passage")
            instruction: 추가 instruction (선택)
            normalize: 임베딩 정규화 여부 (기본: True)
            batch_size: 배치 크기 (기본: 32)
            **kwargs: 추가 파라미터
        """
        super().__init__(model, use_gpu, **kwargs)

        self.prefix = prefix
        self.instruction = instruction
        self.normalize = normalize
        self.batch_size = batch_size

    def _load_model(self):
        """모델 로딩 (lazy loading)"""
        if self._model is not None:
            return

        # Import 검증
        self._validate_import("sentence_transformers", "sentence-transformers")

        from sentence_transformers import SentenceTransformer

        # Device 설정
        self._device = self._get_device()

        if self._device == "cpu":
            logger.warning("NV-Embed works best on GPU. CPU mode may be slow.")

        logger.info(f"Loading NVIDIA NV-Embed-v2 on {self._device}")

        # 모델 로드
        self._model = SentenceTransformer(self.model, device=self._device, trust_remote_code=True)

        logger.info(f"NVIDIA NV-Embed-v2 loaded (max_seq_length: {self._model.max_seq_length})")

    def _prepare_texts(self, texts: List[str]) -> List[str]:
        """
        NV-Embed 포맷으로 텍스트 준비

        Format:
        - Passage: "passage: {text}"
        - Query: "query: {text}"
        - Instruction: "Instruct: {instruction}\nQuery: {text}"
        """
        prepared = []

        for text in texts:
            if self.instruction:
                # Instruction mode
                prepared_text = f"Instruct: {self.instruction}\nQuery: {text}"
            else:
                # Prefix mode
                prepared_text = f"{self.prefix}: {text}"

            prepared.append(prepared_text)

        return prepared

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (동기)"""
        # 모델 로드
        self._load_model()

        try:
            # NV-Embed 포맷으로 준비
            prepared_texts = self._prepare_texts(texts)

            # Encode
            embeddings = self._model.encode(
                prepared_texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            self._log_embed_success(len(texts), f"prefix: {self.prefix}, shape: {embeddings.shape}")

            return embeddings.tolist()

        except Exception as e:
            self._handle_embed_error("NVIDIA NV-Embed", e)
