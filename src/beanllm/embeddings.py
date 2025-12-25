"""
Embeddings - Unified Interface (하위 호환성을 위한 Re-export)
새로운 위치: domain/embeddings/
"""

# 하위 호환성을 위한 re-export
from .domain.embeddings import (
    BaseEmbedding,
    CohereEmbedding,
    Embedding,
    EmbeddingCache,
    EmbeddingResult,
    GeminiEmbedding,
    JinaEmbedding,
    MistralEmbedding,
    OllamaEmbedding,
    OpenAIEmbedding,
    VoyageEmbedding,
    batch_cosine_similarity,
    cosine_similarity,
    embed,
    embed_sync,
    euclidean_distance,
    find_hard_negatives,
    mmr_search,
    normalize_vector,
    query_expansion,
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
