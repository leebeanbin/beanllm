"""
LlamaIndex Query Engine - beanLLM 스타일 Query Engine

LlamaIndex의 Query Engine을 beanLLM 인터페이스로 제공합니다.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from .bridge import LlamaIndexBridge

try:
    from beanllm.utils.logger import get_logger
except ImportError:

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


class LlamaIndexQueryEngine:
    """
    LlamaIndex Query Engine (beanLLM 인터페이스)

    LlamaIndex의 고급 RAG 기능을 beanLLM 스타일로 제공합니다.

    Features:
    - Vector Store Index
    - Query Transformation
    - Response Synthesis
    - Retrieval + Generation

    Example:
        ```python
        from beanllm.integrations.llamaindex import LlamaIndexQueryEngine
        from beanllm.domain.loaders import TextLoader
        from beanllm.domain.embeddings import OpenAIEmbedding
        from beanllm.facade import create_client

        # beanLLM 컴포넌트
        loader = TextLoader("document.txt")
        docs = loader.load()
        embedding_model = OpenAIEmbedding()
        llm_client = create_client(model="gpt-4o-mini")

        # Query Engine 생성
        query_engine = LlamaIndexQueryEngine.from_documents(
            documents=docs,
            embedding_function=embedding_model.embed,
            llm_client=llm_client
        )

        # 쿼리
        response = query_engine.query("What is this document about?")
        print(response.answer)
        print(response.source_nodes)
        ```
    """

    def __init__(
        self,
        llamaindex_query_engine: Any,
        **kwargs,
    ):
        """
        Args:
            llamaindex_query_engine: LlamaIndex QueryEngine 객체
            **kwargs: 추가 파라미터
        """
        self.query_engine = llamaindex_query_engine
        self.kwargs = kwargs

    @classmethod
    def from_documents(
        cls,
        documents: List[Any],
        embedding_function: Callable[[str], List[float]],
        llm_client: Optional[Any] = None,
        similarity_top_k: int = 5,
        response_mode: str = "compact",
        **kwargs,
    ) -> "LlamaIndexQueryEngine":
        """
        문서로부터 Query Engine 생성

        Args:
            documents: beanLLM Document 리스트
            embedding_function: beanLLM 임베딩 함수
            llm_client: beanLLM LLM 클라이언트 (선택)
            similarity_top_k: 검색할 상위 k개 문서 (기본: 5)
            response_mode: 응답 생성 모드
                - "compact": 컴팩트 (기본)
                - "tree_summarize": 트리 요약
                - "refine": 정제
            **kwargs: 추가 파라미터

        Returns:
            LlamaIndexQueryEngine 인스턴스
        """
        try:
            from llama_index.core import Settings, VectorStoreIndex
        except ImportError:
            raise ImportError(
                "llama-index is required. " "Install it with: pip install llama-index"
            )

        # beanLLM Document → LlamaIndex Document
        bridge = LlamaIndexBridge()
        llama_docs = bridge.convert_documents(documents)

        # beanLLM Embeddings → LlamaIndex Embeddings
        llama_embeddings = bridge.wrap_embeddings(embedding_function)

        # Settings 설정
        Settings.embed_model = llama_embeddings

        # LLM 설정 (있으면)
        if llm_client is not None:
            llama_llm = bridge.wrap_llm(llm_client)
            Settings.llm = llama_llm

        # VectorStoreIndex 생성
        index = VectorStoreIndex.from_documents(llama_docs)

        # Query Engine 생성
        query_engine = index.as_query_engine(
            similarity_top_k=similarity_top_k,
            response_mode=response_mode,
            **kwargs,
        )

        logger.info(
            f"LlamaIndex Query Engine created: "
            f"docs={len(documents)}, top_k={similarity_top_k}, "
            f"mode={response_mode}"
        )

        return cls(query_engine, **kwargs)

    def query(self, query_text: str, **kwargs) -> "QueryResponse":
        """
        쿼리 실행

        Args:
            query_text: 쿼리 텍스트
            **kwargs: 추가 파라미터

        Returns:
            QueryResponse 객체
        """
        # LlamaIndex 쿼리
        response = self.query_engine.query(query_text, **kwargs)

        # QueryResponse로 래핑
        return QueryResponse(
            answer=str(response),
            source_nodes=response.source_nodes if hasattr(response, "source_nodes") else [],
            metadata=response.metadata if hasattr(response, "metadata") else {},
        )

    def __repr__(self) -> str:
        return f"LlamaIndexQueryEngine(engine={type(self.query_engine).__name__})"


class QueryResponse:
    """
    Query Response

    쿼리 응답을 담는 데이터 클래스

    Attributes:
        answer: 생성된 답변
        source_nodes: 소스 노드 (검색된 문서)
        metadata: 메타데이터
    """

    def __init__(
        self,
        answer: str,
        source_nodes: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.answer = answer
        self.source_nodes = source_nodes
        self.metadata = metadata or {}

    def __str__(self) -> str:
        return self.answer

    def __repr__(self) -> str:
        return f"QueryResponse(answer={self.answer[:50]}..., sources={len(self.source_nodes)})"


def create_llamaindex_query_engine(
    documents: List[Any],
    embedding_function: Callable[[str], List[float]],
    llm_client: Optional[Any] = None,
    **kwargs,
) -> LlamaIndexQueryEngine:
    """
    LlamaIndex Query Engine 생성 (편의 함수)

    Args:
        documents: beanLLM Document 리스트
        embedding_function: beanLLM 임베딩 함수
        llm_client: beanLLM LLM 클라이언트 (선택)
        **kwargs: 추가 파라미터

    Returns:
        LlamaIndexQueryEngine 인스턴스

    Example:
        ```python
        from beanllm.integrations.llamaindex import create_llamaindex_query_engine
        from beanllm.domain.loaders import TextLoader
        from beanllm.domain.embeddings import OpenAIEmbedding

        loader = TextLoader("document.txt")
        docs = loader.load()
        embedding_model = OpenAIEmbedding()

        # Query Engine 생성
        query_engine = create_llamaindex_query_engine(
            documents=docs,
            embedding_function=embedding_model.embed,
            similarity_top_k=5
        )

        # 쿼리
        response = query_engine.query("What is this about?")
        print(response.answer)
        ```
    """
    return LlamaIndexQueryEngine.from_documents(
        documents=documents,
        embedding_function=embedding_function,
        llm_client=llm_client,
        **kwargs,
    )
