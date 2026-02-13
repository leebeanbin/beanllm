"""
Local-Based Embeddings - 로컬 모델 기반 임베딩 Provider 구현체들

Re-export hub. Implementations live in:
- huggingface_embeddings: HuggingFaceEmbedding
- nvembed_embeddings: NVEmbedEmbedding
- qwen3_embeddings: Qwen3Embedding
- code_embeddings: CodeEmbedding

이 모듈은 로컬에서 실행되는 4개의 임베딩 Provider를 포함합니다:
- HuggingFaceEmbedding: HuggingFace Sentence Transformers 범용 임베딩
- NVEmbedEmbedding: NVIDIA NV-Embed-v2 (MTEB 1위)
- Qwen3Embedding: Alibaba Qwen3 임베딩 (2025년)
- CodeEmbedding: 코드 전용 임베딩 (CodeBERT 등)

모든 Provider는 GPU/CPU 자동 선택, Lazy Loading, 배치 처리 최적화를 지원합니다.
"""

from __future__ import annotations

from beanllm.domain.embeddings.local.code_embeddings import CodeEmbedding
from beanllm.domain.embeddings.local.huggingface_embeddings import HuggingFaceEmbedding
from beanllm.domain.embeddings.local.nvembed_embeddings import NVEmbedEmbedding
from beanllm.domain.embeddings.local.qwen3_embeddings import Qwen3Embedding

__all__ = [
    "HuggingFaceEmbedding",
    "NVEmbedEmbedding",
    "Qwen3Embedding",
    "CodeEmbedding",
]
