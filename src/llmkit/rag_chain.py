"""
RAG Chain - 완전한 RAG를 한 줄로
간단하지만 강력한 질문-답변 시스템
"""
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple, Iterator
from dataclasses import dataclass
from pathlib import Path

from .document_loaders import DocumentLoader, Document
from .text_splitters import TextSplitter
from .embeddings import Embedding
from .vector_stores import VectorStore, from_documents, VectorSearchResult
from .client import Client


@dataclass
class RAGResponse:
    """RAG 응답"""
    answer: str
    sources: List[VectorSearchResult]
    metadata: Dict[str, Any]


class RAGChain:
    """
    완전한 RAG 파이프라인

    Example:
        # 간단한 사용
        rag = RAGChain.from_documents("doc.pdf")
        answer = rag.query("What is this about?")

        # 세밀한 제어
        rag = RAGChain(
            vector_store=store,
            llm=client,
            prompt_template=custom_template
        )
        answer = rag.query("question", k=5, rerank=True)
    """

    DEFAULT_PROMPT_TEMPLATE = """Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""

    def __init__(
        self,
        vector_store,
        llm: Optional[Client] = None,
        prompt_template: Optional[str] = None,
        retriever_config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            vector_store: VectorStore 인스턴스
            llm: LLM Client (기본: gpt-4o-mini)
            prompt_template: 프롬프트 템플릿
            retriever_config: 검색 설정
        """
        self.vector_store = vector_store
        self.llm = llm or Client(model="gpt-4o-mini")
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self.retriever_config = retriever_config or {}

    @classmethod
    def from_documents(
        cls,
        source: Union[str, Path, List[Document]],
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_model: str = "text-embedding-3-small",
        vector_store_provider: Optional[str] = None,
        llm_model: str = "gpt-4o-mini",
        **kwargs
    ) -> 'RAGChain':
        """
        문서에서 직접 RAG 생성 (가장 간단!)

        Args:
            source: 문서 경로 또는 Document 리스트
            chunk_size: 청크 크기
            chunk_overlap: 청크 겹침
            embedding_model: 임베딩 모델
            vector_store_provider: Vector store provider
            llm_model: LLM 모델
            **kwargs: 추가 파라미터

        Example:
            rag = RAGChain.from_documents("doc.pdf")
            answer = rag.query("What is this about?")
        """
        # 1. 문서 로딩
        if isinstance(source, (str, Path)):
            documents = DocumentLoader.load(source)
        else:
            documents = source

        # 2. 텍스트 분할
        chunks = TextSplitter.split(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # 3. 임베딩 및 Vector Store
        embed = Embedding(model=embedding_model)
        embed_func = embed.embed_sync

        vector_store = from_documents(
            chunks,
            embed_func,
            provider=vector_store_provider
        )

        # 4. LLM
        llm = Client(model=llm_model)

        return cls(vector_store=vector_store, llm=llm, **kwargs)

    def retrieve(
        self,
        query: str,
        k: int = 4,
        rerank: bool = False,
        mmr: bool = False,
        hybrid: bool = False,
        **kwargs
    ) -> List[VectorSearchResult]:
        """
        문서 검색

        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            rerank: Cross-encoder로 재순위화
            mmr: MMR로 다양성 고려
            hybrid: Hybrid search (벡터 + 키워드)
            **kwargs: 추가 파라미터

        Returns:
            검색 결과 리스트
        """
        # 검색 방법 선택
        if hybrid:
            results = self.vector_store.hybrid_search(
                query,
                k=k * 2 if rerank else k,
                **kwargs
            )
        elif mmr:
            results = self.vector_store.mmr_search(
                query,
                k=k * 2 if rerank else k,
                **kwargs
            )
        else:
            results = self.vector_store.similarity_search(
                query,
                k=k * 2 if rerank else k,
                **kwargs
            )

        # 재순위화
        if rerank:
            try:
                results = self.vector_store.rerank(query, results, top_k=k)
            except ImportError:
                # sentence-transformers 없으면 skip
                results = results[:k]

        return results

    def _build_context(self, results: List[VectorSearchResult]) -> str:
        """검색 결과에서 컨텍스트 생성"""
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[{i}] {result.document.content}"
            )
        return "\n\n".join(context_parts)

    def _build_prompt(self, query: str, context: str) -> str:
        """프롬프트 생성"""
        return self.prompt_template.format(
            context=context,
            question=query
        )

    def query(
        self,
        question: str,
        k: int = 4,
        include_sources: bool = False,
        rerank: bool = False,
        mmr: bool = False,
        hybrid: bool = False,
        model: Optional[str] = None,
        **kwargs
    ) -> Union[str, Tuple[str, List[VectorSearchResult]]]:
        """
        질문에 답변

        Args:
            question: 질문
            k: 검색할 문서 수
            include_sources: 출처 포함 여부
            rerank: 재순위화 여부
            mmr: MMR 사용 여부
            hybrid: Hybrid search 사용 여부
            model: LLM 모델 (None이면 기본 모델 사용)
            **kwargs: 추가 파라미터

        Returns:
            답변 (include_sources=True면 (답변, 출처) 튜플)

        Example:
            # 간단한 사용
            answer = rag.query("What is AI?")

            # 다른 모델 사용
            answer = rag.query("What is AI?", model="gpt-4o")

            # 출처 포함
            answer, sources = rag.query("What is AI?", include_sources=True)

            # 고급 검색
            answer = rag.query("AI", k=10, rerank=True, mmr=True)
        """
        # 1. 검색
        results = self.retrieve(
            question,
            k=k,
            rerank=rerank,
            mmr=mmr,
            hybrid=hybrid,
            **kwargs
        )

        # 2. 컨텍스트 생성
        context = self._build_context(results)

        # 3. 프롬프트 생성
        prompt = self._build_prompt(question, context)

        # 4. LLM으로 답변 생성
        # 모델 지정되면 새 Client 사용, 아니면 기본 Client 사용
        if model:
            llm = Client(model=model)
            response = llm.chat(prompt)
        else:
            response = self.llm.chat(prompt)

        answer = response.content

        # 5. 반환
        if include_sources:
            return answer, results
        return answer

    def stream_query(
        self,
        question: str,
        k: int = 4,
        rerank: bool = False,
        mmr: bool = False,
        hybrid: bool = False,
        model: Optional[str] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        스트리밍 답변

        Args:
            question: 질문
            k: 검색할 문서 수
            rerank: 재순위화 여부
            mmr: MMR 사용 여부
            hybrid: Hybrid search 사용 여부
            model: LLM 모델 (None이면 기본 모델 사용)
            **kwargs: 추가 파라미터

        Yields:
            답변 청크

        Example:
            for chunk in rag.stream_query("What is AI?"):
                print(chunk, end="", flush=True)

            # 다른 모델 사용
            for chunk in rag.stream_query("What is AI?", model="gpt-4o"):
                print(chunk, end="", flush=True)
        """
        # 1. 검색
        results = self.retrieve(
            question,
            k=k,
            rerank=rerank,
            mmr=mmr,
            hybrid=hybrid,
            **kwargs
        )

        # 2. 컨텍스트 생성
        context = self._build_context(results)

        # 3. 프롬프트 생성
        prompt = self._build_prompt(question, context)

        # 4. 스트리밍 답변
        # 모델 지정되면 새 Client 사용, 아니면 기본 Client 사용
        if model:
            llm = Client(model=model)
            for chunk in llm.stream(prompt):
                yield chunk.content
        else:
            for chunk in self.llm.stream(prompt):
                yield chunk.content

    def batch_query(
        self,
        questions: List[str],
        k: int = 4,
        model: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """
        여러 질문에 대해 배치 답변

        Args:
            questions: 질문 리스트
            k: 검색할 문서 수
            model: LLM 모델 (None이면 기본 모델 사용)
            **kwargs: 추가 파라미터

        Returns:
            답변 리스트

        Example:
            questions = ["What is AI?", "What is ML?", "What is DL?"]
            answers = rag.batch_query(questions)

            # 다른 모델 사용
            answers = rag.batch_query(questions, model="gpt-4o")
        """
        answers = []
        for question in questions:
            answer = self.query(question, k=k, model=model, **kwargs)
            answers.append(answer)
        return answers

    async def aquery(
        self,
        question: str,
        k: int = 4,
        include_sources: bool = False,
        model: Optional[str] = None,
        **kwargs
    ) -> Union[str, Tuple[str, List[VectorSearchResult]]]:
        """
        비동기 질의 (간단한 구현)

        Args:
            question: 질문
            k: 검색할 문서 수
            include_sources: 출처 포함 여부
            model: LLM 모델 (None이면 기본 모델 사용)
            **kwargs: 추가 파라미터

        Returns:
            답변 (include_sources=True면 (답변, 출처) 튜플)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.query(question, k, include_sources, model=model, **kwargs)
        )


class RAGBuilder:
    """
    Fluent API for RAG construction

    Example:
        rag = (RAGBuilder()
            .load_documents("doc.pdf")
            .split_text(chunk_size=500)
            .embed_with(Embedding.openai())
            .store_in(VectorStore.chroma())
            .use_llm(Client(model="gpt-4o"))
            .build())
    """

    def __init__(self):
        self.documents = None
        self.chunks = None
        self.embedding = None
        self.vector_store = None
        self.llm_client = None
        self.prompt_template = None
        self.retriever_config = {}

        # 설정
        self.chunk_size = 500
        self.chunk_overlap = 50

    def load_documents(
        self,
        source: Union[str, Path, List[Document]]
    ) -> 'RAGBuilder':
        """문서 로딩"""
        if isinstance(source, (str, Path)):
            self.documents = DocumentLoader.load(source)
        else:
            self.documents = source
        return self

    def split_text(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        **kwargs
    ) -> 'RAGBuilder':
        """텍스트 분할"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        return self

    def embed_with(self, embedding) -> 'RAGBuilder':
        """임베딩 설정"""
        self.embedding = embedding
        return self

    def store_in(self, vector_store) -> 'RAGBuilder':
        """Vector Store 설정"""
        self.vector_store = vector_store
        return self

    def use_llm(self, llm_client: Client) -> 'RAGBuilder':
        """LLM 설정"""
        self.llm_client = llm_client
        return self

    def with_prompt(self, template: str) -> 'RAGBuilder':
        """프롬프트 템플릿 설정"""
        self.prompt_template = template
        return self

    def with_retriever_config(self, **config) -> 'RAGBuilder':
        """검색 설정"""
        self.retriever_config.update(config)
        return self

    def build(self) -> RAGChain:
        """RAGChain 생성"""
        # 문서 체크
        if self.documents is None:
            raise ValueError("Documents not loaded. Call load_documents() first.")

        # 청크 생성
        if self.chunks is None:
            self.chunks = TextSplitter.split(
                self.documents,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )

        # 임베딩 기본값
        if self.embedding is None:
            self.embedding = Embedding(model="text-embedding-3-small")

        # Vector Store 생성
        if self.vector_store is None:
            embed_func = self.embedding.embed_sync
            self.vector_store = from_documents(self.chunks, embed_func)
        else:
            # Vector Store가 제공되었으면 문서 추가
            self.vector_store.add_documents(self.chunks)

        # LLM 기본값
        if self.llm_client is None:
            self.llm_client = Client(model="gpt-4o-mini")

        # RAGChain 생성
        return RAGChain(
            vector_store=self.vector_store,
            llm=self.llm_client,
            prompt_template=self.prompt_template,
            retriever_config=self.retriever_config
        )


# 편의 함수
def create_rag(
    source: Union[str, Path, List[Document]],
    chunk_size: int = 500,
    embedding_model: str = "text-embedding-3-small",
    llm_model: str = "gpt-4o-mini",
    **kwargs
) -> RAGChain:
    """
    간단한 RAG 생성

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
    return RAGChain.from_documents(
        source,
        chunk_size=chunk_size,
        embedding_model=embedding_model,
        llm_model=llm_model,
        **kwargs
    )


# 별칭 (더 짧은 이름)
RAG = RAGChain
