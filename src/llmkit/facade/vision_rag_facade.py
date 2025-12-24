"""
Vision RAG Facade - 기존 Vision RAG API를 위한 Facade
책임: 하위 호환성 유지, 내부적으로는 Handler/Service 사용
SOLID 원칙:
- Facade 패턴: 복잡한 내부 구조를 단순한 인터페이스로
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

from ..domain.vision.embeddings import CLIPEmbedding, MultimodalEmbedding
from ..domain.vision.loaders import load_images
from ..handler.vision_rag_handler import VisionRAGHandler
from ..utils.logger import get_logger
from ..vector_stores import VectorSearchResult
from .client_facade import Client

if TYPE_CHECKING:
    from ..service.types import VectorStoreProtocol

logger = get_logger(__name__)


class VisionRAG:
    """
    Vision RAG - 이미지 포함 RAG (Facade 패턴)

    기존 API를 유지하면서 내부적으로는 Handler/Service 사용

    Example:
        # 간단한 사용
        rag = VisionRAG.from_images("images/")
        answer = rag.query("Show me images of cats")

        # 세밀한 제어
        rag = VisionRAG(
            vector_store=store,
            vision_embedding=CLIPEmbedding(),
            llm=Client(model="gpt-4o")  # Vision 지원 모델
        )
    """

    DEFAULT_PROMPT_TEMPLATE = """Based on the following context (including images), answer the question.

Context:
{context}

Question: {question}

Answer:"""

    def __init__(
        self,
        vector_store: "VectorStoreProtocol",
        vision_embedding: Optional[Union[CLIPEmbedding, MultimodalEmbedding]] = None,
        llm: Optional[Client] = None,
        prompt_template: Optional[str] = None,
    ):
        """
        Args:
            vector_store: Vector store 인스턴스
            vision_embedding: Vision 임베딩 (기본: CLIP)
            llm: Vision-enabled LLM (기본: gpt-4o)
            prompt_template: 프롬프트 템플릿
        """
        self.vector_store = vector_store
        self.vision_embedding = vision_embedding or CLIPEmbedding()
        self.llm = llm or Client(model="gpt-4o")  # GPT-4o는 vision 지원
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE

        # Handler/Service 초기화 (의존성 주입)
        self._init_services()

    def _init_services(self) -> None:
        """Service 및 Handler 초기화 (의존성 주입) - DI Container 사용"""
        from ..service.impl.vision_rag_service_impl import VisionRAGServiceImpl
        from ..utils.di_container import get_container

        container = get_container()
        service_factory = container.get_service_factory(vector_store=self.vector_store)

        # ChatService 생성
        chat_service = service_factory.create_chat_service()

        # VisionRAGService 생성 (커스텀 의존성)
        vision_rag_service = VisionRAGServiceImpl(
            vector_store=self.vector_store,
            vision_embedding=self.vision_embedding,
            chat_service=chat_service,
            llm=self.llm,
            prompt_template=self.prompt_template,
        )

        # VisionRAGHandler 생성
        self._vision_rag_handler = VisionRAGHandler(vision_rag_service)

    @classmethod
    def from_images(
        cls,
        source: Union[str, Path],
        generate_captions: bool = True,
        llm_model: str = "gpt-4o",
        **kwargs,
    ) -> "VisionRAG":
        """
        이미지에서 직접 Vision RAG 생성 (기존 vision_rag.py의 VisionRAG.from_images() 정확히 마이그레이션)

        Args:
            source: 이미지 디렉토리 또는 파일
            generate_captions: 이미지 캡션 자동 생성
            llm_model: LLM 모델 (vision 지원 필요)
            **kwargs: 추가 파라미터

        Returns:
            VisionRAG 인스턴스

        Example:
            rag = VisionRAG.from_images("images/", generate_captions=True)
            answer = rag.query("What animals are in the images?")
        """
        # 1. 이미지 로딩 (기존과 동일)
        images = load_images(source, generate_captions=generate_captions)

        # 2. 임베딩 (기존과 동일)
        vision_embed = CLIPEmbedding()

        # 이미지를 임베딩하는 함수 (기존과 동일)
        def embed_func(texts):
            # ImageDocument의 경우 이미지 경로 사용
            # 일반 텍스트의 경우 텍스트 임베딩
            results = []
            for text in texts:
                # 간단히 텍스트 임베딩 사용 (실제로는 이미지 구분 필요)
                vec = vision_embed.embed_sync([text])[0]
                results.append(vec)
            return results

        # 3. Vector Store (기존과 동일)
        from ..vector_stores import from_documents

        vector_store = from_documents(images, embed_func)

        # 4. LLM (기존과 동일)
        llm = Client(model=llm_model)

        return cls(vector_store=vector_store, vision_embedding=vision_embed, llm=llm, **kwargs)

    def retrieve(self, query: str, k: int = 4, **kwargs) -> List[VectorSearchResult]:
        """
        이미지 검색 (기존 vision_rag.py의 VisionRAG.retrieve() 정확히 마이그레이션)

        내부적으로 Handler를 사용하여 처리

        Args:
            query: 검색 쿼리 (텍스트)
            k: 반환할 결과 수

        Returns:
            검색 결과 리스트 (ImageDocument 포함)
        """
        # 동기 메서드이지만 내부적으로는 비동기 사용
        response = asyncio.run(self._vision_rag_handler.handle_retrieve(query=query, k=k, **kwargs))
        # DTO에서 값 추출 (기존 API 호환성 유지)
        return response.results or []

    def query(
        self,
        question: str,
        k: int = 4,
        include_sources: bool = False,
        include_images: bool = True,
        **kwargs,
    ) -> Union[str, tuple]:
        """
        질문에 답변 (이미지 포함) (기존 vision_rag.py의 VisionRAG.query() 정확히 마이그레이션)

        내부적으로 Handler를 사용하여 처리

        Args:
            question: 질문
            k: 검색할 문서 수
            include_sources: 출처 포함 여부
            include_images: 이미지 포함 여부
            **kwargs: 추가 파라미터

        Returns:
            답변 (include_sources=True면 (답변, 출처) 튜플)

        Example:
            # 간단한 사용
            answer = rag.query("What is in this image?")

            # 출처 포함
            answer, sources = rag.query("Describe the images", include_sources=True)
        """
        # 동기 메서드이지만 내부적으로는 비동기 사용
        response = asyncio.run(
            self._vision_rag_handler.handle_query(
                question=question,
                k=k,
                include_sources=include_sources,
                include_images=include_images,
                llm_model=self.llm.model if self.llm else "gpt-4o",
                **kwargs,
            )
        )
        # DTO에서 값 추출 (기존 API 호환성 유지)
        if include_sources:
            return response.answer or "", response.sources or []
        return response.answer or ""

    def batch_query(self, questions: List[str], k: int = 4, **kwargs) -> List[str]:
        """
        여러 질문에 대해 배치 답변 (기존 vision_rag.py의 VisionRAG.batch_query() 정확히 마이그레이션)

        내부적으로 Handler를 사용하여 처리

        Args:
            questions: 질문 리스트
            k: 검색할 문서 수
            **kwargs: 추가 파라미터

        Returns:
            답변 리스트
        """
        # 동기 메서드이지만 내부적으로는 비동기 사용
        response = asyncio.run(
            self._vision_rag_handler.handle_batch_query(
                questions=questions,
                k=k,
                include_images=True,
                llm_model=self.llm.model if self.llm else "gpt-4o",
                **kwargs,
            )
        )
        # DTO에서 값 추출 (기존 API 호환성 유지)
        return response.answers or []


class MultimodalRAG(VisionRAG):
    """
    멀티모달 RAG (Facade 패턴)

    텍스트, 이미지, PDF 등을 모두 처리

    Example:
        rag = MultimodalRAG.from_sources([
            "documents/",  # 텍스트 문서
            "images/",     # 이미지
            "pdfs/"        # PDF
        ])

        answer = rag.query("Summarize the documents and images")
    """

    @classmethod
    def from_sources(
        cls,
        sources: List[Union[str, Path]],
        generate_captions: bool = True,
        llm_model: str = "gpt-4o",
        **kwargs,
    ) -> "MultimodalRAG":
        """
        여러 소스에서 멀티모달 RAG 생성 (기존 vision_rag.py의 MultimodalRAG.from_sources() 정확히 마이그레이션)

        Args:
            sources: 소스 경로 리스트
            generate_captions: 이미지 캡션 자동 생성
            llm_model: LLM 모델
            **kwargs: 추가 파라미터

        Returns:
            MultimodalRAG 인스턴스
        """
        from ..document_loaders import DocumentLoader
        from ..text_splitters import TextSplitter
        from ..vision_loaders import ImageLoader, PDFWithImagesLoader

        all_documents = []

        for source in sources:
            source_path = Path(source)

            # 이미지 디렉토리 (기존과 동일)
            if source_path.is_dir():
                # 이미지 찾기
                image_loader = ImageLoader(generate_captions=generate_captions)
                try:
                    images = image_loader.load(source_path)
                    all_documents.extend(images)
                except Exception:
                    pass

                # 텍스트 문서 찾기
                try:
                    docs = DocumentLoader.load(source_path)
                    chunks = TextSplitter.split(docs)
                    all_documents.extend(chunks)
                except Exception:
                    pass

            # 개별 파일 (기존과 동일)
            else:
                if source_path.suffix.lower() == ".pdf":
                    # PDF with images
                    pdf_loader = PDFWithImagesLoader()
                    docs = pdf_loader.load(source_path)
                    all_documents.extend(docs)
                else:
                    # 일반 문서
                    try:
                        docs = DocumentLoader.load(source_path)
                        chunks = TextSplitter.split(docs)
                        all_documents.extend(chunks)
                    except Exception:
                        pass

        # 임베딩 (기존과 동일)
        multimodal_embed = MultimodalEmbedding()

        def embed_func(texts):
            return multimodal_embed.embed_sync(texts)

        # Vector Store (기존과 동일)
        from ..vector_stores import from_documents

        vector_store = from_documents(all_documents, embed_func)

        # LLM (기존과 동일)
        llm = Client(model=llm_model)

        return cls(vector_store=vector_store, vision_embedding=multimodal_embed, llm=llm, **kwargs)


# 편의 함수
def create_vision_rag(
    source: Union[str, Path, List[Union[str, Path]]],
    generate_captions: bool = True,
    llm_model: str = "gpt-4o",
    **kwargs,
) -> Union[VisionRAG, MultimodalRAG]:
    """
    Vision RAG 생성 (간편 함수) (기존 vision_rag.py의 create_vision_rag() 정확히 마이그레이션)

    Args:
        source: 소스 경로 (단일 또는 리스트)
        generate_captions: 이미지 캡션 자동 생성
        llm_model: LLM 모델
        **kwargs: 추가 파라미터

    Returns:
        VisionRAG 또는 MultimodalRAG 인스턴스

    Example:
        # 단일 소스
        rag = create_vision_rag("images/")

        # 여러 소스
        rag = create_vision_rag(["docs/", "images/", "pdfs/"])
    """
    if isinstance(source, list):
        return MultimodalRAG.from_sources(source, generate_captions, llm_model, **kwargs)
    else:
        return VisionRAG.from_images(source, generate_captions, llm_model, **kwargs)
