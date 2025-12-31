"""
Embeddings Domain - 임베딩 도메인
"""

from .advanced import (
    MatryoshkaEmbedding,
    batch_truncate_embeddings,
    find_hard_negatives,
    mmr_search,
    query_expansion,
    truncate_embedding,
)
from .base import BaseEmbedding
from .cache import EmbeddingCache
from .factory import Embedding, embed, embed_sync
from .providers import (
    CodeEmbedding,
    CohereEmbedding,
    GeminiEmbedding,
    HuggingFaceEmbedding,
    JinaEmbedding,
    MistralEmbedding,
    NVEmbedEmbedding,
    OllamaEmbedding,
    OpenAIEmbedding,
    Qwen3Embedding,
    VoyageEmbedding,
)
from .types import EmbeddingResult
from .utils import (
    batch_cosine_similarity,
    cosine_similarity,
    euclidean_distance,
    normalize_vector,
)

__all__ = [
    "EmbeddingResult",
    "BaseEmbedding",
    "OpenAIEmbedding",
    "GeminiEmbedding",
    "OllamaEmbedding",
    "VoyageEmbedding",
    "JinaEmbedding",
    "MistralEmbedding",
    "CohereEmbedding",
    "HuggingFaceEmbedding",
    "NVEmbedEmbedding",
    "Qwen3Embedding",
    "CodeEmbedding",
    "Embedding",
    "EmbeddingCache",
    "embed",
    "embed_sync",
    "cosine_similarity",
    "euclidean_distance",
    "normalize_vector",
    "batch_cosine_similarity",
    "find_hard_negatives",
    "mmr_search",
    "query_expansion",
    "truncate_embedding",
    "batch_truncate_embeddings",
    "MatryoshkaEmbedding",
]
