"""
Vector Store Factory - 벡터 스토어 팩토리
"""

import os
from typing import List, Optional

from .base import BaseVectorStore
from .implementations import (
    ChromaVectorStore,
    FAISSVectorStore,
    PineconeVectorStore,
    QdrantVectorStore,
    WeaviateVectorStore,
)


class VectorStore:
    """
    Unified vector store interface with auto-detection
    Client 패턴과 동일한 방식
    """

    PROVIDERS = {
        "chroma": ChromaVectorStore,
        "pinecone": PineconeVectorStore,
        "faiss": FAISSVectorStore,
        "qdrant": QdrantVectorStore,
        "weaviate": WeaviateVectorStore,
    }

    PROVIDER_ENV_VARS = {
        "chroma": None,  # 로컬, API 키 불필요
        "pinecone": "PINECONE_API_KEY",
        "faiss": None,  # 로컬, API 키 불필요
        "qdrant": None,  # 로컬/클라우드, 선택적
        "weaviate": None,  # 로컬/클라우드, 선택적
    }

    def __new__(cls, provider: Optional[str] = None, **kwargs):
        """
        Factory method to create vector store instance

        Args:
            provider: Provider 이름 (선택적). None이면 자동으로 가장 좋은 provider 선택.

        Examples:
            # 방법 1: 자동 선택 (추천)
            store = VectorStore(embedding_function=embed_func)

            # 방법 2: 명시적 선택
            store = VectorStore(provider="chroma", embedding_function=embed_func)

            # 방법 3: 팩토리 메서드
            store = VectorStore.chroma(embedding_function=embed_func)
        """
        # provider 자동 선택
        if provider is None:
            provider = cls.get_default_provider()

        if provider not in cls.PROVIDERS:
            raise ValueError(
                f"Unknown provider: {provider}. Available: {list(cls.PROVIDERS.keys())}"
            )

        vector_store_class = cls.PROVIDERS[provider]
        return vector_store_class(**kwargs)

    @classmethod
    def chroma(cls, **kwargs) -> ChromaVectorStore:
        """Create Chroma vector store"""
        return ChromaVectorStore(**kwargs)

    @classmethod
    def pinecone(cls, **kwargs) -> PineconeVectorStore:
        """Create Pinecone vector store"""
        return PineconeVectorStore(**kwargs)

    @classmethod
    def faiss(cls, **kwargs) -> FAISSVectorStore:
        """Create FAISS vector store"""
        return FAISSVectorStore(**kwargs)

    @classmethod
    def qdrant(cls, **kwargs) -> QdrantVectorStore:
        """Create Qdrant vector store"""
        return QdrantVectorStore(**kwargs)

    @classmethod
    def weaviate(cls, **kwargs) -> WeaviateVectorStore:
        """Create Weaviate vector store"""
        return WeaviateVectorStore(**kwargs)

    @classmethod
    def list_available_providers(cls) -> List[str]:
        """사용 가능한 provider 목록 반환"""
        available = []

        for provider, env_var in cls.PROVIDER_ENV_VARS.items():
            if env_var is None:
                # 로컬 provider (항상 사용 가능)
                available.append(provider)
            else:
                # API 키 확인
                if os.getenv(env_var):
                    available.append(provider)

        return available

    @classmethod
    def get_default_provider(cls) -> str:
        """기본 provider 반환 (우선순위 기반)"""
        # 우선순위: chroma > faiss > qdrant > pinecone > weaviate
        priority = ["chroma", "faiss", "qdrant", "pinecone", "weaviate"]
        available = cls.list_available_providers()

        for provider in priority:
            if provider in available:
                return provider

        return "chroma"  # 기본값


# Fluent API helper
class VectorStoreBuilder:
    """
    Fluent API for easy vector store creation and usage

    Example:
        store = (VectorStoreBuilder()
            .use_chroma()
            .with_embedding(embed_func)
            .build())
    """

    def __init__(self):
        self.provider = "chroma"
        self.embedding_function = None
        self.kwargs = {}

    def use_chroma(self, **kwargs) -> "VectorStoreBuilder":
        """Use Chroma"""
        self.provider = "chroma"
        self.kwargs.update(kwargs)
        return self

    def use_pinecone(self, **kwargs) -> "VectorStoreBuilder":
        """Use Pinecone"""
        self.provider = "pinecone"
        self.kwargs.update(kwargs)
        return self

    def use_faiss(self, **kwargs) -> "VectorStoreBuilder":
        """Use FAISS"""
        self.provider = "faiss"
        self.kwargs.update(kwargs)
        return self

    def use_qdrant(self, **kwargs) -> "VectorStoreBuilder":
        """Use Qdrant"""
        self.provider = "qdrant"
        self.kwargs.update(kwargs)
        return self

    def use_weaviate(self, **kwargs) -> "VectorStoreBuilder":
        """Use Weaviate"""
        self.provider = "weaviate"
        self.kwargs.update(kwargs)
        return self

    def with_embedding(self, embedding_function) -> "VectorStoreBuilder":
        """Set embedding function"""
        self.embedding_function = embedding_function
        return self

    def with_collection(self, name: str) -> "VectorStoreBuilder":
        """Set collection/index name"""
        self.kwargs["collection_name"] = name
        return self

    def build(self) -> BaseVectorStore:
        """Build vector store"""
        return VectorStore(
            provider=self.provider, embedding_function=self.embedding_function, **self.kwargs
        )


# Convenience functions
def create_vector_store(
    provider: Optional[str] = None, embedding_function=None, **kwargs
) -> BaseVectorStore:
    """
    편리한 vector store 생성 함수

    Args:
        provider: Provider 이름 (선택적). None이면 자동 선택.
        embedding_function: 임베딩 함수
        **kwargs: 추가 파라미터

    Examples:
        # 자동 선택
        store = create_vector_store(embedding_function=embed_func)

        # 명시적 선택
        store = create_vector_store("chroma", embedding_function=embed_func)
    """
    return VectorStore(provider=provider, embedding_function=embedding_function, **kwargs)


def from_documents(
    documents, embedding_function, provider: Optional[str] = None, **kwargs
) -> BaseVectorStore:
    """
    문서에서 직접 vector store 생성

    Args:
        documents: 문서 리스트
        embedding_function: 임베딩 함수
        provider: Provider 이름 (선택적). None이면 자동 선택.
        **kwargs: 추가 파라미터

    Examples:
        # 자동 선택 (가장 간단!)
        store = from_documents(docs, embed_func)

        # 명시적 선택
        store = from_documents(docs, embed_func, provider="chroma")
    """
    store = create_vector_store(provider=provider, embedding_function=embedding_function, **kwargs)
    store.add_documents(documents)
    return store
