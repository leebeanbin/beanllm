"""
RAGChain Facade - 기존 RAGChain API를 위한 Facade
책임: 하위 호환성 유지, 내부적으로는 Handler/Service 사용
SOLID 원칙:
- Facade 패턴: 복잡한 내부 구조를 단순한 인터페이스로
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple, Union

from .client_facade import Client

if TYPE_CHECKING:
    from ..service.types import VectorStoreProtocol


class RAGChain:
    """
    완전한 RAG 파이프라인 (Facade 패턴)

    기존 API를 유지하면서 내부적으로는 Handler/Service 사용

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
        vector_store: "VectorStoreProtocol",
        llm: Optional[Client] = None,
        prompt_template: Optional[str] = None,
        retriever_config: Optional[Dict[str, Any]] = None,
    ) -> None:
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

        # Handler/Service 초기화 (의존성 주입)
        self._init_services()

    def _init_services(self) -> None:
        """Service 및 Handler 초기화 (의존성 주입) - DI Container 사용"""
        from ..utils.di_container import get_container

        container = get_container()
        service_factory = container.get_service_factory(vector_store=self.vector_store)
        handler_factory = container.get_handler_factory(service_factory)

        # RAGHandler 생성
        self._rag_handler = handler_factory.create_rag_handler()

    @classmethod
    def from_documents(
        cls,
        source: Union[str, Path, List[Any]],
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_model: str = "text-embedding-3-small",
        vector_store_provider: Optional[str] = None,
        llm_model: str = "gpt-4o-mini",
        **kwargs: Any,
    ) -> "RAGChain":
        """
        문서에서 직접 RAG 생성 (가장 간단!)

        내부적으로는 기존 로직 사용 (점진적 마이그레이션)

        Args:
            source: 문서 경로 또는 Document 리스트
            chunk_size: 청크 크기
            chunk_overlap: 청크 겹침
            embedding_model: 임베딩 모델
            vector_store_provider: Vector store provider
            llm_model: LLM 모델
            **kwargs: 추가 파라미터
        """
        # 기존 로직 사용 (점진적 마이그레이션)
        from ..document_loaders import DocumentLoader
        from ..embeddings import Embedding
        from ..text_splitters import TextSplitter
        from ..vector_stores import from_documents

        # 1. 문서 로딩
        if isinstance(source, (str, Path)):
            documents = DocumentLoader.load(source)
        else:
            documents = source

        # 2. 텍스트 분할
        chunks = TextSplitter.split(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # 3. 임베딩 및 Vector Store
        embed = Embedding(model=embedding_model)
        embed_func = embed.embed_sync

        vector_store = from_documents(chunks, embed_func, provider=vector_store_provider)

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
        **kwargs: Any,
    ) -> List[Any]:
        """
        문서 검색

        내부적으로 RAGService 사용

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
        # Handler를 통한 처리
        import asyncio

        from ..dto.request.rag_request import RAGRequest

        RAGRequest(
            query=query,
            vector_store=self.vector_store,
            k=k,
            rerank=rerank,
            mmr=mmr,
            hybrid=hybrid,
            **kwargs,
        )

        # Handler를 통한 처리
        return asyncio.run(
            self._rag_handler.handle_retrieve(
                query=query,
                vector_store=self.vector_store,
                k=k,
                rerank=rerank,
                mmr=mmr,
                hybrid=hybrid,
                **kwargs,
            )
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
        **kwargs: Any,
    ) -> Union[str, Tuple[str, List[Any]]]:
        """
        질문에 답변

        내부적으로 Handler를 사용하여 처리

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
        """
        # Handler를 통한 처리
        import asyncio

        llm_model = model or (self.llm.model if self.llm else "gpt-4o-mini")

        response = asyncio.run(
            self._rag_handler.handle_query(
                query=question,
                vector_store=self.vector_store,
                k=k,
                rerank=rerank,
                mmr=mmr,
                hybrid=hybrid,
                llm_model=llm_model,
                prompt_template=self.prompt_template,
                **kwargs,
            )
        )

        if include_sources:
            return response.answer, response.sources
        return response.answer

    def stream_query(
        self,
        question: str,
        k: int = 4,
        rerank: bool = False,
        mmr: bool = False,
        hybrid: bool = False,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        스트리밍 답변 (기존 rag_chain.py의 stream_query 정확히 마이그레이션)

        Args:
            question: 질문
            k: 검색할 문서 수
            rerank: 재순위화 여부
            mmr: MMR 사용 여부
            hybrid: Hybrid search 사용 여부
            model: LLM 모델
            **kwargs: 추가 파라미터

        Yields:
            답변 청크
        """
        # 기존 rag_chain.py의 stream_query 정확히 마이그레이션
        # 기존: for chunk in llm.stream(prompt): yield chunk.content
        # 기존 코드: llm.stream(prompt)는 llm.stream_chat([{"role": "user", "content": prompt}])와 동일
        import asyncio

        llm_model = model or (self.llm.model if self.llm else "gpt-4o-mini")

        # 비동기 제너레이터를 동기 Iterator로 변환
        async def async_stream():
            async for chunk in self._rag_handler.handle_stream_query(
                query=question,
                vector_store=self.vector_store,
                k=k,
                rerank=rerank,
                mmr=mmr,
                hybrid=hybrid,
                llm_model=llm_model,
                prompt_template=self.prompt_template,
                **kwargs,
            ):
                yield chunk

        # 비동기 제너레이터를 동기 Iterator로 변환 (기존 동작 보장)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async_gen = async_stream()

        # 동기 Iterator로 변환
        if loop.is_running():
            # 이미 실행 중인 루프가 있는 경우
            import queue
            import threading

            q: queue.Queue[Optional[str]] = queue.Queue()
            stop_flag = threading.Event()

            async def collect():
                try:
                    async for chunk in async_gen:
                        q.put(chunk)
                finally:
                    q.put(None)
                    stop_flag.set()

            asyncio.create_task(collect())
            while not stop_flag.is_set() or not q.empty():
                try:
                    chunk = q.get(timeout=0.1)
                    if chunk is None:
                        break
                    yield chunk
                except queue.Empty:
                    if stop_flag.is_set():
                        break
        else:
            # 새 루프에서 실행
            while True:
                try:
                    chunk = loop.run_until_complete(async_gen.__anext__())
                    yield chunk
                except StopAsyncIteration:
                    break

    def batch_query(
        self, questions: List[str], k: int = 4, model: Optional[str] = None, **kwargs: Any
    ) -> List[str]:
        """
        여러 질문에 대해 배치 답변 (내부적으로 자동 병렬 처리)

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
        # 내부적으로 병렬 처리 사용 (사용자는 신경 쓸 필요 없음)
        import asyncio

        from ...utils.error_handling import AsyncTokenBucket

        # 자동 최적화 설정
        rate_limiter = AsyncTokenBucket(rate=1.0, capacity=20.0)
        max_concurrent = 10

        async def _batch_query_async():
            semaphore = asyncio.Semaphore(max_concurrent)

            async def query_one(question: str):
                """단일 질의 (Rate Limiting + Semaphore)"""
                await rate_limiter.wait(cost=1.0)
                async with semaphore:
                    return await self.aquery(question, k=k, model=model, **kwargs)

            tasks = [query_one(q) for q in questions]
            answers = await asyncio.gather(*tasks, return_exceptions=True)

            # 결과 정리
            results = []
            for ans in answers:
                if isinstance(ans, Exception):
                    results.append(f"Error: {str(ans)}")
                elif isinstance(ans, tuple):
                    results.append(ans[0])  # (answer, sources) 튜플인 경우
                else:
                    results.append(str(ans))

            return results

        # 비동기 실행
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 이미 실행 중인 루프가 있으면 순차 처리로 폴백
                # (중첩 이벤트 루프는 복잡하므로)
                answers = []
                for question in questions:
                    answer = self.query(question, k=k, model=model, **kwargs)
                    answers.append(answer)
                return answers
            else:
                return loop.run_until_complete(_batch_query_async())
        except RuntimeError:
            # 루프가 없으면 새로 생성
            return asyncio.run(_batch_query_async())

    async def aquery(
        self,
        question: str,
        k: int = 4,
        include_sources: bool = False,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[str, Tuple[str, List[Any]]]:
        """
        비동기 질의 (기존 rag_chain.py의 aquery 정확히 마이그레이션)

        Args:
            question: 질문
            k: 검색할 문서 수
            include_sources: 출처 포함 여부
            model: LLM 모델 (None이면 기본 모델 사용)
            **kwargs: 추가 파라미터

        Returns:
            답변 (include_sources=True면 (답변, 출처) 튜플)
        """
        # 기존 rag_chain.py의 aquery 정확히 마이그레이션
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.query(question, k, include_sources, model=model, **kwargs)
        )


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

    def load_documents(self, source: Union[str, Path, List[Any]]) -> "RAGBuilder":
        """문서 로딩 (기존 rag_chain.py와 정확히 동일)"""
        from ..document_loaders import DocumentLoader

        if isinstance(source, (str, Path)):
            self.documents = DocumentLoader.load(source)
        else:
            self.documents = source
        return self

    def split_text(
        self, chunk_size: int = 500, chunk_overlap: int = 50, **kwargs: Any
    ) -> "RAGBuilder":
        """텍스트 분할 (기존 rag_chain.py와 정확히 동일)"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        return self

    def embed_with(self, embedding: Any) -> "RAGBuilder":
        """임베딩 설정 (기존 rag_chain.py와 정확히 동일)"""
        self.embedding = embedding
        return self

    def store_in(self, vector_store: Any) -> "RAGBuilder":
        """Vector Store 설정 (기존 rag_chain.py와 정확히 동일)"""
        self.vector_store = vector_store
        return self

    def use_llm(self, llm_client: Client) -> "RAGBuilder":
        """LLM 설정 (기존 rag_chain.py와 정확히 동일)"""
        self.llm_client = llm_client
        return self

    def with_prompt(self, template: str) -> "RAGBuilder":
        """프롬프트 템플릿 설정 (기존 rag_chain.py와 정확히 동일)"""
        self.prompt_template = template
        return self

    def with_retriever_config(self, **config: Any) -> "RAGBuilder":
        """검색 설정 (기존 rag_chain.py와 정확히 동일)"""
        self.retriever_config.update(config)
        return self

    def build(self) -> RAGChain:
        """RAGChain 생성 (기존 rag_chain.py의 build 정확히 마이그레이션)"""
        from ..embeddings import Embedding
        from ..text_splitters import TextSplitter
        from ..vector_stores import from_documents

        # 문서 체크 (기존과 동일)
        if self.documents is None:
            raise ValueError("Documents not loaded. Call load_documents() first.")

        # 청크 생성 (기존과 동일)
        if self.chunks is None:
            self.chunks = TextSplitter.split(
                self.documents, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )

        # 임베딩 기본값 (기존과 동일)
        if self.embedding is None:
            self.embedding = Embedding(model="text-embedding-3-small")

        # Vector Store 생성 (기존과 동일)
        if self.vector_store is None:
            embed_func = self.embedding.embed_sync
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


# 편의 함수 (기존 rag_chain.py의 create_rag 정확히 마이그레이션)
def create_rag(
    source: Union[str, Path, List[Any]],
    chunk_size: int = 500,
    embedding_model: str = "text-embedding-3-small",
    llm_model: str = "gpt-4o-mini",
    **kwargs: Any,
) -> RAGChain:
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
    return RAGChain.from_documents(
        source,
        chunk_size=chunk_size,
        embedding_model=embedding_model,
        llm_model=llm_model,
        **kwargs,
    )


# 별칭 (더 짧은 이름) - 기존 rag_chain.py와 동일
RAG = RAGChain
