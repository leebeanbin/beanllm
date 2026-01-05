"""
LlamaIndex Bridge - beanLLM ↔ LlamaIndex 브릿지

beanLLM의 Document, Embeddings를 LlamaIndex 형식으로 변환합니다.
"""

import logging
from typing import Any, Callable, List, Optional

try:
    from beanllm.utils.logger import get_logger
except ImportError:

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


class LlamaIndexBridge:
    """
    beanLLM ↔ LlamaIndex 브릿지

    beanLLM의 타입을 LlamaIndex 형식으로 변환합니다.

    Features:
    - beanLLM Document → LlamaIndex Document
    - beanLLM Embedding Function → LlamaIndex Embeddings
    - 메타데이터 보존

    Example:
        ```python
        from beanllm.integrations.llamaindex import LlamaIndexBridge
        from beanllm.domain.loaders import TextLoader
        from beanllm.domain.embeddings import OpenAIEmbedding

        # beanLLM 문서 로드
        loader = TextLoader("document.txt")
        bean_docs = loader.load()

        # beanLLM 임베딩
        embedding_model = OpenAIEmbedding()

        # LlamaIndex로 변환
        bridge = LlamaIndexBridge()
        llama_docs = bridge.convert_documents(bean_docs)
        llama_embeddings = bridge.wrap_embeddings(embedding_model.embed)

        # LlamaIndex에서 사용
        from llama_index.core import VectorStoreIndex

        index = VectorStoreIndex.from_documents(
            llama_docs,
            embed_model=llama_embeddings
        )
        ```
    """

    @staticmethod
    def convert_documents(bean_documents: List[Any]) -> List[Any]:
        """
        beanLLM Document → LlamaIndex Document 변환

        Args:
            bean_documents: beanLLM Document 리스트

        Returns:
            LlamaIndex Document 리스트
        """
        try:
            from llama_index.core import Document as LlamaDocument
        except ImportError:
            raise ImportError(
                "llama-index is required for LlamaIndexBridge. "
                "Install it with: pip install llama-index"
            )

        llama_docs = []

        for bean_doc in bean_documents:
            # LlamaIndex Document 생성
            llama_doc = LlamaDocument(
                text=bean_doc.content,
                metadata=bean_doc.metadata or {},
                doc_id=bean_doc.metadata.get("id") if bean_doc.metadata else None,
            )

            llama_docs.append(llama_doc)

        logger.info(f"Converted {len(bean_docs)} beanLLM documents to LlamaIndex format")

        return llama_docs

    @staticmethod
    def convert_to_bean_documents(llama_documents: List[Any]) -> List[Any]:
        """
        LlamaIndex Document → beanLLM Document 변환

        Args:
            llama_documents: LlamaIndex Document 리스트

        Returns:
            beanLLM Document 리스트
        """
        try:
            from beanllm.domain.loaders import Document as BeanDocument
        except ImportError:
            raise ImportError("beanLLM Document not available")

        bean_docs = []

        for llama_doc in llama_documents:
            # beanLLM Document 생성
            bean_doc = BeanDocument(
                content=llama_doc.text,
                metadata=llama_doc.metadata or {},
                source=llama_doc.metadata.get("source", "llamaindex"),
            )

            bean_docs.append(bean_doc)

        logger.info(f"Converted {len(llama_docs)} LlamaIndex documents to beanLLM format")

        return bean_docs

    @staticmethod
    def wrap_embeddings(
        embedding_function: Callable[[str], List[float]],
        model_name: str = "beanllm-custom",
    ) -> Any:
        """
        beanLLM Embedding Function → LlamaIndex BaseEmbedding 래핑

        Args:
            embedding_function: beanLLM 임베딩 함수 (str -> List[float])
            model_name: 모델 이름 (메타데이터용)

        Returns:
            LlamaIndex BaseEmbedding 객체

        Example:
            ```python
            from beanllm.domain.embeddings import OpenAIEmbedding

            embedding_model = OpenAIEmbedding()
            llama_embeddings = LlamaIndexBridge.wrap_embeddings(
                embedding_function=embedding_model.embed,
                model_name="text-embedding-3-small"
            )
            ```
        """
        try:
            from llama_index.core.embeddings import BaseEmbedding
        except ImportError:
            raise ImportError(
                "llama-index is required. " "Install it with: pip install llama-index"
            )

        class BeanLLMEmbeddingWrapper(BaseEmbedding):
            """beanLLM Embedding Wrapper for LlamaIndex"""

            def __init__(self, embedding_fn: Callable[[str], List[float]], model: str):
                super().__init__()
                self.embedding_fn = embedding_fn
                self.model_name = model

            def _get_query_embedding(self, query: str) -> List[float]:
                """쿼리 임베딩"""
                return self.embedding_fn(query)

            def _get_text_embedding(self, text: str) -> List[float]:
                """텍스트 임베딩"""
                return self.embedding_fn(text)

            async def _aget_query_embedding(self, query: str) -> List[float]:
                """비동기 쿼리 임베딩"""
                # 동기 함수를 비동기로 래핑
                return self._get_query_embedding(query)

            async def _aget_text_embedding(self, text: str) -> List[float]:
                """비동기 텍스트 임베딩"""
                return self._get_text_embedding(text)

        wrapper = BeanLLMEmbeddingWrapper(
            embedding_fn=embedding_function, model=model_name
        )

        logger.info(f"Wrapped beanLLM embedding function for LlamaIndex: {model_name}")

        return wrapper

    @staticmethod
    def wrap_llm(llm_client: Any, model_name: str = "beanllm-custom") -> Any:
        """
        beanLLM LLM Client → LlamaIndex LLM 래핑

        Args:
            llm_client: beanLLM LLM 클라이언트
            model_name: 모델 이름

        Returns:
            LlamaIndex LLM 객체

        Example:
            ```python
            from beanllm.facade import create_client

            client = create_client(model="gpt-4o-mini")
            llama_llm = LlamaIndexBridge.wrap_llm(client, model_name="gpt-4o-mini")
            ```
        """
        try:
            from llama_index.core.llms import CompletionResponse, CustomLLM
            from llama_index.core.llms.callbacks import llm_completion_callback
        except ImportError:
            raise ImportError(
                "llama-index is required. " "Install it with: pip install llama-index"
            )

        class BeanLLMWrapper(CustomLLM):
            """beanLLM Client Wrapper for LlamaIndex"""

            def __init__(self, client: Any, model: str):
                super().__init__()
                self.client = client
                self.model_name = model

            @property
            def metadata(self):
                return {
                    "model_name": self.model_name,
                    "is_chat_model": True,
                }

            @llm_completion_callback()
            def complete(self, prompt: str, **kwargs) -> CompletionResponse:
                """Completion"""
                response = self.client.chat(
                    messages=[{"role": "user", "content": prompt}], **kwargs
                )
                return CompletionResponse(text=response.content)

            @llm_completion_callback()
            def stream_complete(self, prompt: str, **kwargs):
                """Stream Completion (not implemented)"""
                # beanLLM 클라이언트가 스트리밍을 지원하면 구현 가능
                raise NotImplementedError("Streaming not supported yet")

        wrapper = BeanLLMWrapper(client=llm_client, model=model_name)

        logger.info(f"Wrapped beanLLM LLM client for LlamaIndex: {model_name}")

        return wrapper
