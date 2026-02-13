"""
Code embedding implementation.
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


class CodeEmbedding(BaseLocalEmbedding):
    """
    Code Embedding - 코드 전용 임베딩 모델 (2024-2025)

    코드 검색, 코드 이해, 코드 생성을 위한 전용 임베딩입니다.

    지원 모델:
    - microsoft/codebert-base: CodeBERT (기본)
    - microsoft/graphcodebert-base: GraphCodeBERT (그래프 구조 이해)
    - microsoft/unixcoder-base: UniXcoder (다국어 코드)
    - Salesforce/codet5-base: CodeT5 (코드-텍스트)

    Features:
    - 프로그래밍 언어 자동 감지
    - 코드 구조 이해 (AST, 데이터 플로우)
    - 자연어-코드 간 의미 매칭
    - 코드 검색 및 유사도 비교

    Example:
        ```python
        from beanllm.domain.embeddings import CodeEmbedding

        # CodeBERT 사용
        emb = CodeEmbedding(model="microsoft/codebert-base")

        # 코드 임베딩
        code_vectors = emb.embed_sync([
            "def hello(): print('Hello')",
            "function hello() { console.log('Hello'); }"
        ])

        # 자연어 쿼리로 코드 검색
        query_vec = emb.embed_sync(["print hello to console"])[0]
        # query_vec와 code_vectors 비교하여 관련 코드 찾기
        ```

    Use Cases:
    - 코드 검색 (Semantic Code Search)
    - 코드 복제 감지 (Clone Detection)
    - 코드 문서화 자동 생성
    - 코드 추천 시스템

    References:
        - CodeBERT: https://arxiv.org/abs/2002.08155
        - GraphCodeBERT: https://arxiv.org/abs/2009.08366
        - UniXcoder: https://arxiv.org/abs/2203.03850
    """

    def __init__(
        self,
        model: str = "microsoft/codebert-base",
        use_gpu: bool = True,
        normalize: bool = True,
        batch_size: int = 16,
        **kwargs,
    ):
        """
        Args:
            model: 코드 임베딩 모델
                - microsoft/codebert-base: CodeBERT (기본)
                - microsoft/graphcodebert-base: GraphCodeBERT
                - microsoft/unixcoder-base: UniXcoder
                - Salesforce/codet5-base: CodeT5
            use_gpu: GPU 사용 여부 (기본: True)
            normalize: 임베딩 정규화 여부 (기본: True)
            batch_size: 배치 크기 (기본: 16)
            **kwargs: 추가 파라미터
        """
        super().__init__(model, use_gpu, **kwargs)

        self.normalize = normalize
        self.batch_size = batch_size

        # Lazy loading
        self._tokenizer = None

    def _load_model(self):
        """모델 로딩 (lazy loading, 분산 락 적용)"""

        def _load_impl():
            # Import 검증
            self._validate_import("transformers", "transformers")

            from transformers import AutoModel, AutoTokenizer

            # Device 설정
            self._device = self._get_device()

            logger.info(f"Loading Code model: {self.model} on {self._device}")

            # 모델 및 토크나이저 로드
            self._tokenizer = AutoTokenizer.from_pretrained(self.model)
            self._model = AutoModel.from_pretrained(self.model)
            self._model.to(self._device)
            self._model.eval()

            logger.info(f"Code model loaded: {self.model}")

        # 분산 락을 사용한 모델 로딩
        self._load_model_with_lock(self.model, _load_impl)

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling with attention mask"""
        import torch

        token_embeddings = model_output[0]  # First element = token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """코드들을 임베딩 (동기)"""
        self._load_model()
        assert self._tokenizer is not None
        assert self._model is not None

        try:
            import torch

            all_embeddings = []

            # 배치 처리
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]

                # 토크나이징
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self._device) for k, v in encoded.items()}

                # 추론
                with torch.no_grad():
                    model_output = self._model(**encoded)

                # Mean pooling
                embeddings = self._mean_pooling(model_output, encoded["attention_mask"])

                # 정규화
                if self.normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                # CPU로 이동 및 리스트 변환
                batch_embeddings = embeddings.cpu().numpy().tolist()
                all_embeddings.extend(batch_embeddings)

            self._log_embed_success(len(texts), f"batch_size: {self.batch_size}")

            return all_embeddings

        except Exception as e:
            self._handle_embed_error("Code", e)
