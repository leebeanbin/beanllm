"""
Embeddings Factory - 임베딩 팩토리
"""

import os
from typing import List, Optional, Union

from .base import BaseEmbedding
from .providers import (
    CohereEmbedding,
    GeminiEmbedding,
    JinaEmbedding,
    MistralEmbedding,
    OllamaEmbedding,
    OpenAIEmbedding,
    VoyageEmbedding,
)

try:
    from ...utils.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


class Embedding:
    """
    Embedding 팩토리 - 자동 provider 감지

    **beanllm 방식: Client와 같은 패턴!**

    Example:
        ```python
        from beanllm.domain.embeddings import Embedding

        # 자동 감지 (모델 이름으로)
        emb = Embedding(model="text-embedding-3-small")  # OpenAI 자동
        emb = Embedding(model="embed-english-v3.0")      # Cohere 자동

        # 임베딩
        vectors = await emb.embed(["text1", "text2"])

        # 동기 버전
        vectors = emb.embed_sync(["text1", "text2"])
        ```
    """

    # 모델 이름 패턴으로 provider 감지
    PROVIDER_PATTERNS = {
        "openai": [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ],
        "gemini": [
            "models/embedding-001",
            "models/text-embedding-004",
            "embedding-001",
            "text-embedding-004",
        ],
        "ollama": [
            "nomic-embed-text",
            "mxbai-embed-large",
            "all-minilm",
        ],
        "voyage": [
            "voyage-2",
            "voyage-large-2",
            "voyage-code-2",
            "voyage-lite-02-instruct",
        ],
        "jina": [
            "jina-embeddings-v2-base-en",
            "jina-embeddings-v2-small-en",
            "jina-embeddings-v2-base-zh",
            "jina-clip-v1",
        ],
        "mistral": [
            "mistral-embed",
        ],
        "cohere": [
            "embed-english-v3.0",
            "embed-english-light-v3.0",
            "embed-multilingual-v3.0",
            "embed-english-v2.0",
        ],
    }

    # Provider별 클래스 매핑
    PROVIDERS = {
        "openai": OpenAIEmbedding,
        "gemini": GeminiEmbedding,
        "ollama": OllamaEmbedding,
        "voyage": VoyageEmbedding,
        "jina": JinaEmbedding,
        "mistral": MistralEmbedding,
        "cohere": CohereEmbedding,
    }

    # Provider별 필요한 환경변수
    PROVIDER_ENV_VARS = {
        "openai": "OPENAI_API_KEY",
        "gemini": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
        "ollama": None,  # 로컬, API 키 불필요
        "voyage": "VOYAGE_API_KEY",
        "jina": "JINA_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "cohere": "COHERE_API_KEY",
    }

    def __new__(cls, model: str, provider: Optional[str] = None, **kwargs) -> BaseEmbedding:
        """
        Embedding 인스턴스 생성 (자동 provider 감지)

        Args:
            model: 모델 이름
            provider: Provider 명시 (None이면 자동 감지)
            **kwargs: Provider별 추가 파라미터

        Returns:
            적절한 Embedding 인스턴스
        """
        # Provider 감지
        if provider is None:
            provider = cls._detect_provider(model)
            if provider:
                logger.info(f"Auto-detected provider: {provider} for model: {model}")
            else:
                # 기본: OpenAI
                logger.warning(
                    f"Could not detect provider for model: {model}, defaulting to OpenAI"
                )
                provider = "openai"

        # Provider 클래스 선택
        if provider not in cls.PROVIDERS:
            raise ValueError(
                f"Unknown provider: {provider}. Supported: {list(cls.PROVIDERS.keys())}"
            )

        embedding_class = cls.PROVIDERS[provider]
        return embedding_class(model=model, **kwargs)

    @classmethod
    def _detect_provider(cls, model: str) -> Optional[str]:
        """모델 이름으로 provider 감지"""
        model_lower = model.lower()

        for provider, patterns in cls.PROVIDER_PATTERNS.items():
            for pattern in patterns:
                if pattern.lower() in model_lower:
                    return provider

        return None

    @classmethod
    def openai(cls, model: str = "text-embedding-3-small", **kwargs) -> OpenAIEmbedding:
        """
        OpenAI Embedding 생성 (명시적)

        Example:
            ```python
            emb = Embedding.openai()
            emb = Embedding.openai(model="text-embedding-3-large")
            ```
        """
        return OpenAIEmbedding(model=model, **kwargs)

    @classmethod
    def gemini(cls, model: str = "models/embedding-001", **kwargs) -> GeminiEmbedding:
        """
        Gemini Embedding 생성 (명시적)

        Example:
            ```python
            emb = Embedding.gemini()
            emb = Embedding.gemini(model="models/text-embedding-004")
            ```
        """
        return GeminiEmbedding(model=model, **kwargs)

    @classmethod
    def ollama(cls, model: str = "nomic-embed-text", **kwargs) -> OllamaEmbedding:
        """
        Ollama Embedding 생성 (명시적)

        Example:
            ```python
            emb = Embedding.ollama()
            emb = Embedding.ollama(model="mxbai-embed-large")
            ```
        """
        return OllamaEmbedding(model=model, **kwargs)

    @classmethod
    def voyage(cls, model: str = "voyage-2", **kwargs) -> VoyageEmbedding:
        """
        Voyage AI Embedding 생성 (명시적)

        Example:
            ```python
            emb = Embedding.voyage()
            emb = Embedding.voyage(model="voyage-large-2")
            ```
        """
        return VoyageEmbedding(model=model, **kwargs)

    @classmethod
    def jina(cls, model: str = "jina-embeddings-v2-base-en", **kwargs) -> JinaEmbedding:
        """
        Jina AI Embedding 생성 (명시적)

        Example:
            ```python
            emb = Embedding.jina()
            emb = Embedding.jina(model="jina-embeddings-v2-small-en")
            ```
        """
        return JinaEmbedding(model=model, **kwargs)

    @classmethod
    def mistral(cls, model: str = "mistral-embed", **kwargs) -> MistralEmbedding:
        """
        Mistral AI Embedding 생성 (명시적)

        Example:
            ```python
            emb = Embedding.mistral()
            ```
        """
        return MistralEmbedding(model=model, **kwargs)

    @classmethod
    def cohere(cls, model: str = "embed-english-v3.0", **kwargs) -> CohereEmbedding:
        """
        Cohere Embedding 생성 (명시적)

        Example:
            ```python
            emb = Embedding.cohere()
            emb = Embedding.cohere(model="embed-multilingual-v3.0")
            ```
        """
        return CohereEmbedding(model=model, **kwargs)

    @classmethod
    def list_available_providers(cls) -> List[str]:
        """
        사용 가능한 provider 목록

        API 키가 설정된 provider만 반환

        Returns:
            사용 가능한 provider 이름 리스트

        Example:
            ```python
            providers = Embedding.list_available_providers()
            print(f"Available: {providers}")
            # ['openai', 'ollama']
            ```
        """
        available = []

        for provider, env_var in cls.PROVIDER_ENV_VARS.items():
            if env_var is None:  # Ollama (로컬)
                available.append(provider)
            elif isinstance(env_var, list):  # 여러 가능한 환경변수
                if any(os.getenv(var) for var in env_var):
                    available.append(provider)
            else:  # 단일 환경변수
                if os.getenv(env_var):
                    available.append(provider)

        return available

    @classmethod
    def get_default_provider(cls) -> Optional[str]:
        """
        기본 provider 반환

        사용 가능한 provider 중 우선순위가 가장 높은 것

        우선순위: OpenAI > Gemini > Voyage > Cohere > Ollama

        Returns:
            기본 provider 이름

        Example:
            ```python
            provider = Embedding.get_default_provider()
            emb = Embedding(model="...", provider=provider)
            ```
        """
        priority = ["openai", "gemini", "voyage", "cohere", "ollama"]
        available = cls.list_available_providers()

        for provider in priority:
            if provider in available:
                return provider

        return None


# 편의 함수
async def embed(
    texts: Union[str, List[str]], model: str = "text-embedding-3-small", **kwargs
) -> List[List[float]]:
    """
    텍스트를 임베딩하는 편의 함수

    Args:
        texts: 단일 텍스트 또는 리스트
        model: 모델 이름
        **kwargs: 추가 파라미터

    Returns:
        임베딩 벡터 리스트

    Example:
        ```python
        from beanllm.domain.embeddings import embed

        # 단일 텍스트
        vector = await embed("Hello world")

        # 여러 텍스트
        vectors = await embed(["text1", "text2", "text3"])
        ```
    """
    # 단일 텍스트를 리스트로 변환
    if isinstance(texts, str):
        texts = [texts]

    embedding = Embedding(model=model, **kwargs)
    return await embedding.embed(texts)


def embed_sync(
    texts: Union[str, List[str]], model: str = "text-embedding-3-small", **kwargs
) -> List[List[float]]:
    """
    텍스트를 임베딩하는 편의 함수 (동기)

    Args:
        texts: 단일 텍스트 또는 리스트
        model: 모델 이름
        **kwargs: 추가 파라미터

    Returns:
        임베딩 벡터 리스트

    Example:
        ```python
        from beanllm.domain.embeddings import embed_sync

        # 단일 텍스트
        vector = embed_sync("Hello world")

        # 여러 텍스트
        vectors = embed_sync(["text1", "text2", "text3"])
        ```
    """
    # 단일 텍스트를 리스트로 변환
    if isinstance(texts, str):
        texts = [texts]

    embedding = Embedding(model=model, **kwargs)
    return embedding.embed_sync(texts)
