"""
Embeddings Domain - 임베딩 도메인
"""

from .advanced import find_hard_negatives, mmr_search, query_expansion
from .base import BaseEmbedding
from .cache import EmbeddingCache
from .factory import Embedding, embed, embed_sync
from .providers import (
    CohereEmbedding,
    GeminiEmbedding,
    JinaEmbedding,
    MistralEmbedding,
    OllamaEmbedding,
    OpenAIEmbedding,
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
]
