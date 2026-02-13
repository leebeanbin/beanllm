"""
VisionRAGServiceImpl - Vision RAG 서비스 구현체
SOLID 원칙:
- SRP: Vision RAG 비즈니스 로직만 담당
- DIP: 인터페이스에 의존 (의존성 주입)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from beanllm.dto.request.ml.vision_rag_request import VisionRAGRequest
from beanllm.dto.response.ml.vision_rag_response import VisionRAGResponse
from beanllm.infrastructure.distributed import get_event_logger, get_rate_limiter
from beanllm.infrastructure.distributed.pipeline_decorators import with_distributed_features
from beanllm.service.vision_rag_service import IVisionRAGService
from beanllm.utils.logging import get_logger

# 환경변수로 분산 모드 활성화 여부 확인
USE_DISTRIBUTED = os.getenv("USE_DISTRIBUTED", "false").lower() == "true"

if TYPE_CHECKING:
    from beanllm.service.chat_service import IChatService
    from beanllm.service.types import VectorStoreProtocol
    from beanllm.vision_embeddings import CLIPEmbedding, MultimodalEmbedding

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
        prompt_template: Optional[str] = None,
    ) -> None:
        """
        의존성 주입을 통한 생성자

        Args:
            vector_store: 벡터 스토어
            vision_embedding: Vision 임베딩 (선택적)
            chat_service: 채팅 서비스 인터페이스 (LLM 호출에 사용)
            prompt_template: 프롬프트 템플릿 (선택적)
        """
        self._vector_store = vector_store
        self._vision_embedding = vision_embedding
        self._chat_service = chat_service
        self._prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE

    @with_distributed_features(
        pipeline_type="vision_rag",
        enable_cache=True,
        enable_rate_limiting=True,
        enable_event_streaming=True,
        cache_key_prefix="vision_rag:retrieve",
        rate_limit_key="vision:embedding",
        event_type="vision_rag.retrieve",
    )
    async def retrieve(self, request: VisionRAGRequest) -> VisionRAGResponse:
        """
        이미지 검색 (기존 vision_rag.py의 VisionRAG.retrieve() 정확히 마이그레이션)

        Args:
            request: Vision RAG 요청 DTO

        Returns:
            VisionRAGResponse: Vision RAG 응답 DTO (results 필드에 검색 결과 포함)
        """
        query = request.query or ""
        k = request.k

        # 기존과 동일: vector_store.similarity_search 호출
        results = self._vector_store.similarity_search(query, k=k)

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
            from beanllm.vision_loaders import ImageDocument
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

    @with_distributed_features(
        pipeline_type="vision_rag",
        enable_rate_limiting=True,
        enable_event_streaming=True,
        rate_limit_key=lambda self, args, kwargs: (
            f"llm:{(args[0] if args else kwargs.get('request')).llm_model if hasattr(args[0] if args else kwargs.get('request'), 'llm_model') else 'default'}"
        ),
        event_type="vision_rag.query",
    )
    async def query(self, request: VisionRAGRequest) -> VisionRAGResponse:
        """
        질문에 답변 (이미지 포함) (기존 vision_rag.py의 VisionRAG.query() 정확히 마이그레이션)

        Args:
            request: Vision RAG 요청 DTO

        Returns:
            VisionRAGResponse: Vision RAG 응답 DTO (answer, sources 필드 포함)
        """
        question = request.question or request.query or ""

        # 1. 검색 (기존과 동일)
        retrieve_request = VisionRAGRequest(
            query=question,
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

            # LLM 호출 (IChatService 인터페이스 사용)
            if not self._chat_service:
                raise ValueError("chat_service must be provided for query operations")

            from beanllm.dto.request.core.chat_request import ChatRequest

            chat_request = ChatRequest(messages=messages, model=request.llm_model)
            chat_response = await self._chat_service.chat(chat_request)
            answer = chat_response.content
        else:
            # 텍스트만
            prompt = self._prompt_template.format(
                context=context, question=request.question or request.query
            )

            if not self._chat_service:
                raise ValueError("chat_service must be provided for query operations")

            from beanllm.dto.request.core.chat_request import ChatRequest

            chat_request = ChatRequest(
                messages=[{"role": "user", "content": prompt}], model=request.llm_model
            )
            chat_response = await self._chat_service.chat(chat_request)
            answer = chat_response.content

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

        # 분산 모드: Task Queue 사용
        if USE_DISTRIBUTED and len(request.questions) > 5:
            try:
                from beanllm.infrastructure.distributed import BatchProcessor, ConcurrencyController

                batch_processor = BatchProcessor(task_type="vision_rag.tasks", max_concurrent=10)
                concurrency_controller = ConcurrencyController()
                rate_limiter = get_rate_limiter()

                async def process_question(question: str) -> str:
                    """단일 질문 처리 (Rate Limiting + 동시성 제어)"""
                    await rate_limiter.wait(f"llm:{request.llm_model or 'default'}", cost=1.0)

                    async with concurrency_controller.with_concurrency_control(
                        "vision_rag.query",
                        max_concurrent=10,
                        rate_limit_key=f"llm:{request.llm_model or 'default'}",
                    ):
                        query_request = VisionRAGRequest(
                            question=question,
                            k=request.k,
                            include_sources=False,
                            include_images=request.include_images,
                            llm_model=request.llm_model,
                            extra_params=request.extra_params,
                        )
                        query_response = await self.query(query_request)
                        return query_response.answer or ""

                # 배치 처리
                tasks_data = [{"question": q} for q in request.questions]
                results = await batch_processor.process_items(
                    task_name="query",
                    items=request.questions,
                    item_to_task_data=lambda q: {"question": q},
                    handler=lambda task_data: process_question(task_data["question"]),
                    priority=0,
                )

                answers = [r.get("result", "") if isinstance(r, dict) else str(r) for r in results]

                # 이벤트 발행
                event_logger = get_event_logger()
                await event_logger.log_event(
                    "vision_rag.batch_query",
                    {"question_count": len(request.questions), "answer_count": len(answers)},
                    level="info",
                )

                return VisionRAGResponse(answers=answers)
            except Exception as e:
                logger.warning(
                    f"Distributed batch processing failed: {e}, falling back to sequential"
                )
                # Fallback to sequential

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
