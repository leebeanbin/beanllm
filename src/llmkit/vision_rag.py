"""
Vision RAG
이미지를 포함한 멀티모달 RAG 시스템
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .client import Client
from .vector_stores import VectorSearchResult, from_documents
from .vision_embeddings import CLIPEmbedding, MultimodalEmbedding
from .vision_loaders import ImageDocument, load_images


class VisionRAG:
    """
    Vision RAG - 이미지 포함 RAG

    텍스트와 이미지를 함께 검색하고 답변 생성

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
        vector_store,
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

    @classmethod
    def from_images(
        cls,
        source: Union[str, Path],
        generate_captions: bool = True,
        llm_model: str = "gpt-4o",
        **kwargs,
    ) -> "VisionRAG":
        """
        이미지에서 직접 Vision RAG 생성

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
        # 1. 이미지 로딩
        images = load_images(source, generate_captions=generate_captions)

        # 2. 임베딩
        vision_embed = CLIPEmbedding()

        # 이미지를 임베딩하는 함수
        def embed_func(texts):
            # ImageDocument의 경우 이미지 경로 사용
            # 일반 텍스트의 경우 텍스트 임베딩
            results = []
            for text in texts:
                # 간단히 텍스트 임베딩 사용 (실제로는 이미지 구분 필요)
                vec = vision_embed.embed_sync([text])[0]
                results.append(vec)
            return results

        # 3. Vector Store
        vector_store = from_documents(images, embed_func)

        # 4. LLM
        llm = Client(model=llm_model)

        return cls(vector_store=vector_store, vision_embedding=vision_embed, llm=llm, **kwargs)

    def retrieve(self, query: str, k: int = 4, **kwargs) -> List[VectorSearchResult]:
        """
        이미지 검색

        Args:
            query: 검색 쿼리 (텍스트)
            k: 반환할 결과 수

        Returns:
            검색 결과 리스트 (ImageDocument 포함)
        """
        return self.vector_store.similarity_search(query, k=k, **kwargs)

    def _build_context(
        self, results: List[VectorSearchResult], include_images: bool = True
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        검색 결과에서 컨텍스트 생성

        Args:
            results: 검색 결과
            include_images: 이미지 포함 여부

        Returns:
            컨텍스트 (텍스트 또는 멀티모달 메시지)
        """
        if not include_images:
            # 텍스트만
            context_parts = []
            for i, result in enumerate(results, 1):
                context_parts.append(f"[{i}] {result.document.content}")
            return "\n\n".join(context_parts)

        # 멀티모달 컨텍스트 (GPT-4V 스타일)
        context_messages = []

        for i, result in enumerate(results, 1):
            doc = result.document

            # ImageDocument인 경우
            if isinstance(doc, ImageDocument) and doc.image_path:
                # 이미지 + 캡션
                message = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{doc.get_image_base64()}"},
                }
                context_messages.append(message)

                if doc.caption:
                    context_messages.append({"type": "text", "text": f"[Image {i}] {doc.caption}"})
            else:
                # 텍스트만
                context_messages.append({"type": "text", "text": f"[{i}] {doc.content}"})

        return context_messages

    def query(
        self,
        question: str,
        k: int = 4,
        include_sources: bool = False,
        include_images: bool = True,
        **kwargs,
    ) -> Union[str, tuple]:
        """
        질문에 답변 (이미지 포함)

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
        # 1. 검색
        results = self.retrieve(question, k=k, **kwargs)

        # 2. 컨텍스트 생성
        context = self._build_context(results, include_images=include_images)

        # 3. LLM으로 답변 생성
        if include_images and isinstance(context, list):
            # 멀티모달 메시지
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"Question: {question}\n\nContext:"}]
                    + context
                    + [{"type": "text", "text": "\nAnswer:"}],
                }
            ]
            response = self.llm.chat(messages)
        else:
            # 텍스트만
            prompt = self.prompt_template.format(context=context, question=question)
            response = self.llm.chat(prompt)

        answer = response.content

        # 4. 반환
        if include_sources:
            return answer, results
        return answer

    def batch_query(self, questions: List[str], k: int = 4, **kwargs) -> List[str]:
        """
        여러 질문에 대해 배치 답변

        Args:
            questions: 질문 리스트
            k: 검색할 문서 수
            **kwargs: 추가 파라미터

        Returns:
            답변 리스트
        """
        answers = []
        for question in questions:
            answer = self.query(question, k=k, **kwargs)
            answers.append(answer)
        return answers


class MultimodalRAG(VisionRAG):
    """
    멀티모달 RAG

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
        여러 소스에서 멀티모달 RAG 생성

        Args:
            sources: 소스 경로 리스트
            generate_captions: 이미지 캡션 자동 생성
            llm_model: LLM 모델
            **kwargs: 추가 파라미터

        Returns:
            MultimodalRAG 인스턴스
        """
        from .document_loaders import DocumentLoader
        from .text_splitters import TextSplitter
        from .vision_loaders import ImageLoader, PDFWithImagesLoader

        all_documents = []

        for source in sources:
            source_path = Path(source)

            # 이미지 디렉토리
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

            # 개별 파일
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

        # 임베딩
        multimodal_embed = MultimodalEmbedding()

        def embed_func(texts):
            return multimodal_embed.embed_sync(texts)

        # Vector Store
        vector_store = from_documents(all_documents, embed_func)

        # LLM
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
    Vision RAG 생성 (간편 함수)

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
