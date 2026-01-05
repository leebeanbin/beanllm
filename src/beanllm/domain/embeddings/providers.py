"""
Embeddings Providers - 임베딩 Provider 구현체들 (Re-export Module)

이 모듈은 모든 임베딩 Provider 클래스를 re-export하여 backward compatibility를 보장합니다.

실제 구현은 다음 모듈로 분리되어 있습니다:
- api_embeddings.py: API 기반 임베딩 (OpenAI, Gemini, Ollama, Voyage, Jina, Mistral, Cohere)
- local_embeddings.py: 로컬 모델 기반 임베딩 (HuggingFace, NVEmbed, Qwen3, Code)

사용법:
    ```python
    # 기존 코드와 동일하게 사용 가능 (backward compatible)
    from beanllm.domain.embeddings.providers import OpenAIEmbedding, HuggingFaceEmbedding

    # 또는 세부 모듈에서 직접 import
    from beanllm.domain.embeddings.api_embeddings import OpenAIEmbedding
    from beanllm.domain.embeddings.local_embeddings import HuggingFaceEmbedding
    ```
"""

# Re-export all providers for backward compatibility

# API-based embeddings (7개)
from .api_embeddings import (
    CohereEmbedding,
    GeminiEmbedding,
    JinaEmbedding,
    MistralEmbedding,
    OllamaEmbedding,
    OpenAIEmbedding,
    VoyageEmbedding,
)

# Local-based embeddings (4개)
from .local_embeddings import (
    CodeEmbedding,
    HuggingFaceEmbedding,
    NVEmbedEmbedding,
    Qwen3Embedding,
)

# Explicit __all__ for better IDE support
__all__ = [
    # API-based embeddings
    "OpenAIEmbedding",
    "GeminiEmbedding",
    "OllamaEmbedding",
    "VoyageEmbedding",
    "JinaEmbedding",
    "MistralEmbedding",
    "CohereEmbedding",
    # Local-based embeddings
    "HuggingFaceEmbedding",
    "NVEmbedEmbedding",
    "Qwen3Embedding",
    "CodeEmbedding",
]
