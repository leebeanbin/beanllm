"""
VisionRAGServiceImpl - Vision RAG 서비스 구현체
SOLID 원칙:
- SRP: Vision RAG 비즈니스 로직만 담당
- DIP: 인터페이스에 의존 (의존성 주입)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from ...dto.request.vision_rag_request import VisionRAGRequest
from ...dto.response.vision_rag_response import VisionRAGResponse
from ...utils.logger import get_logger
from ..vision_rag_service import IVisionRAGService

if TYPE_CHECKING:
    from ...facade.client_facade import Client
    from ...service.chat_service import IChatService
    from ...service.types import VectorStoreProtocol
    from ...vision_embeddings import CLIPEmbedding, MultimodalEmbedding

logger = get_logger(__name__)


class VisionRAGServiceImpl(IVisionRAGService):
    """
    Vision RAG 서비스 구현체

    책임:
    - Vision RAG 비즈니스 로직만
    - 검증 없음 (Handler에서 처리)
    - 에러 처리 없음 (Handler에서 처리)

    SOLID:
    - SRP: Vision RAG 비즈니스 로직만
    - DIP: 인터페이스에 의존 (의존성 주입)
    """

    DEFAULT_PROMPT_TEMPLATE = """Based on the following context (including images), answer the question.

Context:
{context}

Question: {question}

Answer:"""

    def __init__(
        self,
        vector_store: "VectorStoreProtocol",
        vision_embedding: Optional[Union["CLIPEmbedding", "MultimodalEmbedding"]] = None,
        chat_service: Optional["IChatService"] = None,
        llm: Optional["Client"] = None,
        prompt_template: Optional[str] = None,
    ) -> None:
        """
        의존성 주입을 통한 생성자

        Args:
            vector_store: 벡터 스토어
            vision_embedding: Vision 임베딩 (선택적)
            chat_service: 채팅 서비스 (선택적, llm이 없을 때 사용)
            llm: LLM Client (선택적, chat_service가 없을 때 사용)
            prompt_template: 프롬프트 템플릿 (선택적)
        """
        self._vector_store = vector_store
        self._vision_embedding = vision_embedding
        self._chat_service = chat_service
        self._llm = llm
        self._prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE

    async def retrieve(self, request: VisionRAGRequest) -> VisionRAGResponse:
        """
        이미지 검색 (기존 vision_rag.py의 VisionRAG.retrieve() 정확히 마이그레이션)

        Args:
            request: Vision RAG 요청 DTO

        Returns:
            VisionRAGResponse: Vision RAG 응답 DTO (results 필드에 검색 결과 포함)
        """
        # 기존과 동일: vector_store.similarity_search 호출
        results = self._vector_store.similarity_search(request.query or "", k=request.k)

        return VisionRAGResponse(results=results)

    def _build_context(
        self, results: List[Any], include_images: bool = True
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        검색 결과에서 컨텍스트 생성 (기존 vision_rag.py의 VisionRAG._build_context() 정확히 마이그레이션)

        Args:
            results: 검색 결과 (VectorSearchResult 리스트)
            include_images: 이미지 포함 여부

        Returns:
            컨텍스트 (텍스트 또는 멀티모달 메시지)
        """
        try:
            from ...vision_loaders import ImageDocument
        except ImportError:
            # vision_loaders가 없으면 텍스트만 사용
            ImageDocument = None

        if not include_images or ImageDocument is None:
            # 텍스트만 (기존과 동일)
            context_parts = []
            for i, result in enumerate(results, 1):
                context_parts.append(f"[{i}] {result.document.content}")
            return "\n\n".join(context_parts)

        # 멀티모달 컨텍스트 (GPT-4V 스타일) (기존과 동일)
        context_messages = []

        for i, result in enumerate(results, 1):
            doc = result.document

            # ImageDocument인 경우 (기존과 동일)
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

    async def query(self, request: VisionRAGRequest) -> VisionRAGResponse:
        """
        질문에 답변 (이미지 포함) (기존 vision_rag.py의 VisionRAG.query() 정확히 마이그레이션)

        Args:
            request: Vision RAG 요청 DTO

        Returns:
            VisionRAGResponse: Vision RAG 응답 DTO (answer, sources 필드 포함)
        """
        # 1. 검색 (기존과 동일)
        retrieve_request = VisionRAGRequest(
            query=request.question or request.query,
            k=request.k,
            extra_params=request.extra_params,
        )
        retrieve_response = await self.retrieve(retrieve_request)
        results = retrieve_response.results or []

        # 2. 컨텍스트 생성 (기존과 동일)
        context = self._build_context(results, include_images=request.include_images)

        # 3. LLM으로 답변 생성 (기존과 동일)
        if request.include_images and isinstance(context, list):
            # 멀티모달 메시지 (기존과 동일)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Question: {request.question or request.query}\n\nContext:",
                        }
                    ]
                    + context
                    + [{"type": "text", "text": "\nAnswer:"}],
                }
            ]

            # LLM 호출 (기존과 동일)
            if self._llm:
                response = await self._llm.chat(messages)
                answer = response.content
            elif self._chat_service:
                from ...dto.request.chat_request import ChatRequest

                chat_request = ChatRequest(messages=messages, model=request.llm_model)
                chat_response = await self._chat_service.chat(chat_request)
                answer = chat_response.content
            else:
                raise ValueError("Either llm or chat_service must be provided")
        else:
            # 텍스트만 (기존과 동일)
            prompt = self._prompt_template.format(
                context=context, question=request.question or request.query
            )

            if self._llm:
                response = await self._llm.chat(prompt)
                answer = response.content
            elif self._chat_service:
                from ...dto.request.chat_request import ChatRequest

                chat_request = ChatRequest(
                    messages=[{"role": "user", "content": prompt}], model=request.llm_model
                )
                chat_response = await self._chat_service.chat(chat_request)
                answer = chat_response.content
            else:
                raise ValueError("Either llm or chat_service must be provided")

        # 4. 반환 (기존과 동일)
        if request.include_sources:
            return VisionRAGResponse(answer=answer, sources=results)
        return VisionRAGResponse(answer=answer)

    async def batch_query(self, request: VisionRAGRequest) -> VisionRAGResponse:
        """
        여러 질문에 대해 배치 답변 (기존 vision_rag.py의 VisionRAG.batch_query() 정확히 마이그레이션)

        Args:
            request: Vision RAG 요청 DTO (questions 필드 사용)

        Returns:
            VisionRAGResponse: Vision RAG 응답 DTO (answers 필드에 답변 리스트 포함)
        """
        if not request.questions:
            raise ValueError("questions field is required for batch_query")

        # 기존과 동일: 각 질문에 대해 query 호출
        answers = []
        for question in request.questions:
            query_request = VisionRAGRequest(
                question=question,
                k=request.k,
                include_sources=False,  # 배치에서는 출처 제외
                include_images=request.include_images,
                llm_model=request.llm_model,
                extra_params=request.extra_params,
            )
            query_response = await self.query(query_request)
            answers.append(query_response.answer or "")

        return VisionRAGResponse(answers=answers)
