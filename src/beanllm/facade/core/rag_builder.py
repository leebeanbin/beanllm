"""
RAGBuilder - Fluent API for RAG construction (Builder pattern helper).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from beanllm.facade.core.client_facade import Client

if TYPE_CHECKING:
    from beanllm.facade.core.rag_facade import RAGChain


class RAGBuilder:
    """
    Fluent API for RAG construction (기존 rag_chain.py의 RAGBuilder 정확히 마이그레이션)

    Example:
        rag = (RAGBuilder()
            .load_documents("doc.pdf")
            .split_text(chunk_size=500)
            .embed_with(Embedding.openai())
            .store_in(VectorStore.chroma())
            .use_llm(Client(model="gpt-4o"))
            .build())
    """

    def __init__(self) -> None:
        """기존 rag_chain.py의 __init__ 정확히 마이그레이션"""

        self.documents: Optional[List[Any]] = None
        self.chunks: Optional[List[Any]] = None
        self.embedding: Optional[Any] = None
        self.vector_store: Optional[Any] = None
        self.llm_client: Optional[Client] = None
        self.prompt_template: Optional[str] = None
        self.retriever_config: Dict[str, Any] = {}

        # 설정
        self.chunk_size = 500
        self.chunk_overlap = 50

    def load_documents(self, source: Union[str, Path, List[Any]]) -> RAGBuilder:
        """문서 로딩 (기존 rag_chain.py와 정확히 동일)"""
        from beanllm.domain.loaders import DocumentLoader

        if isinstance(source, (str, Path)):
            self.documents = DocumentLoader.load(source)
        else:
            self.documents = source
        return self

    def split_text(
        self, chunk_size: int = 500, chunk_overlap: int = 50, **kwargs: Any
    ) -> RAGBuilder:
        """텍스트 분할 (기존 rag_chain.py와 정확히 동일)"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        return self

    def embed_with(self, embedding: Any) -> RAGBuilder:
        """임베딩 설정 (기존 rag_chain.py와 정확히 동일)"""
        self.embedding = embedding
        return self

    def store_in(self, vector_store: Any) -> RAGBuilder:
        """Vector Store 설정 (기존 rag_chain.py와 정확히 동일)"""
        self.vector_store = vector_store
        return self

    def use_llm(self, llm_client: Client) -> RAGBuilder:
        """LLM 설정 (기존 rag_chain.py와 정확히 동일)"""
        self.llm_client = llm_client
        return self

    def with_prompt(self, template: str) -> RAGBuilder:
        """프롬프트 템플릿 설정 (기존 rag_chain.py와 정확히 동일)"""
        self.prompt_template = template
        return self

    def with_retriever_config(self, **config: Any) -> RAGBuilder:
        """검색 설정 (기존 rag_chain.py와 정확히 동일)"""
        self.retriever_config.update(config)
        return self

    def build(self) -> "RAGChain":
        """RAGChain 생성 (기존 rag_chain.py의 build 정확히 마이그레이션)"""
        from beanllm.domain.embeddings import Embedding
        from beanllm.domain.splitters import TextSplitter
        from beanllm.domain.vector_stores import from_documents
        from beanllm.facade.core.rag_facade import RAGChain

        # 문서 체크 (기존과 동일)
        if self.documents is None:
            raise ValueError("Documents not loaded. Call load_documents() first.")

        # 청크 생성 (기존과 동일)
        if self.chunks is None:
            self.chunks = TextSplitter.split(
                self.documents, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )

        # 임베딩 기본값 - Ollama 사용 시 Ollama embedding 사용
        if self.embedding is None:
            # LLM이 Ollama면 Ollama embedding 사용 (API key 불필요)
            if (
                self.llm_client
                and hasattr(self.llm_client, "provider")
                and self.llm_client.provider == "ollama"
            ):
                self.embedding = Embedding(model="nomic-embed-text")
            elif self.llm_client and hasattr(self.llm_client, "model"):
                # 모델 이름으로 Ollama 감지
                model_lower = self.llm_client.model.lower()
                if any(x in model_lower for x in ["llama", "mistral", "qwen", "phi", "gemma"]):
                    self.embedding = Embedding(model="nomic-embed-text")
                else:
                    self.embedding = Embedding(model="text-embedding-3-small")
            else:
                self.embedding = Embedding(model="text-embedding-3-small")

        # Vector Store 생성 (기존과 동일)
        if self.vector_store is None:
            embed_func = getattr(self.embedding, "embed_sync", None)
            if embed_func is None or not callable(embed_func):
                raise ValueError("embedding must provide embed_sync callable")
            self.vector_store = from_documents(self.chunks, embed_func)
        else:
            # Vector Store가 제공되었으면 문서 추가 (기존과 동일)
            self.vector_store.add_documents(self.chunks)

        # LLM 기본값 (기존과 동일)
        if self.llm_client is None:
            self.llm_client = Client(model="gpt-4o-mini")

        # RAGChain 생성 (기존과 동일)
        return RAGChain(
            vector_store=self.vector_store,
            llm=self.llm_client,
            prompt_template=self.prompt_template,
            retriever_config=self.retriever_config,
        )


def create_rag(
    source: Union[str, Path, List[Any]],
    chunk_size: int = 500,
    embedding_model: str = "text-embedding-3-small",
    llm_model: str = "gpt-4o-mini",
    **kwargs: Any,
) -> "RAGChain":
    """
    간단한 RAG 생성 (기존 rag_chain.py의 create_rag 정확히 마이그레이션)

    Args:
        source: 문서 경로 또는 Document 리스트
        chunk_size: 청크 크기
        embedding_model: 임베딩 모델
        llm_model: LLM 모델
        **kwargs: 추가 파라미터

    Returns:
        RAGChain

    Example:
        rag = create_rag("document.pdf")
        answer = rag.query("What is this about?")
    """
    from beanllm.facade.core.rag_facade import RAGChain

    return RAGChain.from_documents(
        source,
        chunk_size=chunk_size,
        embedding_model=embedding_model,
        llm_model=llm_model,
        **kwargs,
    )
