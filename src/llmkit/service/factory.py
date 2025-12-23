"""
ServiceFactory - 서비스 의존성 주입 팩토리
SOLID 원칙:
- DIP: 인터페이스에 의존
- OCP: 확장 가능 (새 서비스 추가 시 수정 불필요)
- SRP: 의존성 관리만 담당
- DRY: 공통 생성 로직 재사용
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from ..infrastructure.adapter import ParameterAdapter
from .agent_service import IAgentService
from .audio_service import IAudioService
from .chain_service import IChainService
from .chat_service import IChatService
from .evaluation_service import IEvaluationService
from .graph_service import IGraphService
from .multi_agent_service import IMultiAgentService
from .rag_service import IRAGService
from .state_graph_service import IStateGraphService
from .vision_rag_service import IVisionRAGService
from .web_search_service import IWebSearchService

if TYPE_CHECKING:
    from .types import (
        EmbeddingServiceProtocol,
        ProviderFactoryProtocol,
        ToolRegistryProtocol,
        VectorStoreProtocol,
    )


class ServiceFactory:
    """
    서비스 팩토리

    책임:
    - 서비스 인스턴스 생성 및 의존성 주입
    - 의존성 관리만 (비즈니스 로직 없음)

    SOLID:
    - SRP: 의존성 관리만
    - DIP: 인터페이스에 의존
    - OCP: 확장 가능
    """

    def __init__(
        self,
        provider_factory: "ProviderFactoryProtocol",
        parameter_adapter: Optional[ParameterAdapter] = None,
        vector_store: Optional["VectorStoreProtocol"] = None,
        embedding_service: Optional["EmbeddingServiceProtocol"] = None,
    ) -> None:
        """
        의존성 주입을 통한 생성자

        Args:
            provider_factory: Provider 생성 팩토리
            parameter_adapter: 파라미터 어댑터 (선택적)
            vector_store: 벡터 스토어 (선택적)
            embedding_service: 임베딩 서비스 (선택적)
        """
        self._provider_factory = provider_factory
        self._parameter_adapter = parameter_adapter or ParameterAdapter()
        self._vector_store = vector_store
        self._embedding_service = embedding_service

    def create_chat_service(self) -> IChatService:
        """
        채팅 서비스 생성 (의존성 주입)

        Returns:
            IChatService: 채팅 서비스 인스턴스

        책임:
            - 의존성 주입만
            - 비즈니스 로직 없음
        """
        from .impl.chat_service_impl import ChatServiceImpl

        return ChatServiceImpl(
            provider_factory=self._provider_factory,
            parameter_adapter=self._parameter_adapter,
        )

    def create_rag_service(self, chat_service: Optional[IChatService] = None) -> IRAGService:
        """
        RAG 서비스 생성 (의존성 주입)

        Args:
            chat_service: 채팅 서비스 (없으면 자동 생성)

        Returns:
            IRAGService: RAG 서비스 인스턴스

        책임:
            - 의존성 주입만
            - 비즈니스 로직 없음
        """
        from .impl.rag_service_impl import RAGServiceImpl

        # 공통 로직: chat_service 자동 생성
        chat_service = self._get_or_create_chat_service(chat_service)

        return RAGServiceImpl(
            vector_store=self._vector_store,
            chat_service=chat_service,
            embedding_service=self._embedding_service,
        )

    def create_agent_service(
        self,
        chat_service: Optional[IChatService] = None,
        tool_registry: Optional["ToolRegistryProtocol"] = None,
    ) -> IAgentService:
        """
        에이전트 서비스 생성 (의존성 주입)

        Args:
            chat_service: 채팅 서비스 (없으면 자동 생성)
            tool_registry: 도구 레지스트리 (선택적)

        Returns:
            IAgentService: 에이전트 서비스 인스턴스

        책임:
            - 의존성 주입만
            - 비즈니스 로직 없음
        """
        from .impl.agent_service_impl import AgentServiceImpl

        # 공통 로직: chat_service 자동 생성
        chat_service = self._get_or_create_chat_service(chat_service)

        return AgentServiceImpl(
            chat_service=chat_service,
            tool_registry=tool_registry,
        )

    def _get_or_create_chat_service(self, chat_service: Optional[IChatService]) -> IChatService:
        """
        ChatService 가져오기 또는 생성 (공통 로직)

        책임:
            - 중복 코드 제거
            - DRY 원칙 적용
        """
        return chat_service if chat_service is not None else self.create_chat_service()

    def create_chain_service(self, chat_service: Optional[IChatService] = None) -> IChainService:
        """
        Chain 서비스 생성 (의존성 주입)

        Args:
            chat_service: 채팅 서비스 (없으면 자동 생성)

        Returns:
            IChainService: Chain 서비스 인스턴스

        책임:
            - 의존성 주입만
            - 비즈니스 로직 없음
        """
        from .impl.chain_service_impl import ChainServiceImpl

        # 공통 로직: chat_service 자동 생성
        chat_service = self._get_or_create_chat_service(chat_service)

        return ChainServiceImpl(chat_service=chat_service)

    def create_graph_service(self) -> IGraphService:
        """
        Graph 서비스 생성 (의존성 주입)

        Returns:
            IGraphService: Graph 서비스 인스턴스

        책임:
            - 의존성 주입만
            - 비즈니스 로직 없음
        """
        from .impl.graph_service_impl import GraphServiceImpl

        return GraphServiceImpl()

    def create_state_graph_service(self) -> IStateGraphService:
        """
        StateGraph 서비스 생성 (의존성 주입)

        Returns:
            IStateGraphService: StateGraph 서비스 인스턴스

        책임:
            - 의존성 주입만
            - 비즈니스 로직 없음
        """
        from .impl.state_graph_service_impl import StateGraphServiceImpl

        return StateGraphServiceImpl()

    def create_multi_agent_service(self) -> IMultiAgentService:
        """
        Multi-Agent 서비스 생성 (의존성 주입)

        Returns:
            IMultiAgentService: Multi-Agent 서비스 인스턴스

        책임:
            - 의존성 주입만
            - 비즈니스 로직 없음
        """
        from .impl.multi_agent_service_impl import MultiAgentServiceImpl

        return MultiAgentServiceImpl()

    def create_web_search_service(self) -> IWebSearchService:
        """
        Web Search 서비스 생성 (의존성 주입)

        Returns:
            IWebSearchService: Web Search 서비스 인스턴스

        책임:
            - 의존성 주입만
            - 비즈니스 로직 없음
        """
        from .impl.web_search_service_impl import WebSearchServiceImpl

        return WebSearchServiceImpl()

    def create_vision_rag_service(
        self,
        vector_store: "VectorStoreProtocol",
        vision_embedding: Optional[Any] = None,
        llm: Optional[Any] = None,
        chat_service: Optional[IChatService] = None,
        prompt_template: Optional[str] = None,
    ) -> IVisionRAGService:
        """
        Vision RAG 서비스 생성 (의존성 주입)

        Args:
            vector_store: 벡터 스토어 (필수)
            vision_embedding: Vision 임베딩 (선택적)
            llm: LLM Client (선택적)
            chat_service: 채팅 서비스 (선택적, llm이 없을 때 사용)
            prompt_template: 프롬프트 템플릿 (선택적)

        Returns:
            IVisionRAGService: Vision RAG 서비스 인스턴스

        책임:
            - 의존성 주입만
            - 비즈니스 로직 없음
        """
        from .impl.vision_rag_service_impl import VisionRAGServiceImpl

        # chat_service 자동 생성
        if not chat_service and not llm:
            chat_service = self.create_chat_service()

        return VisionRAGServiceImpl(
            vector_store=vector_store,
            vision_embedding=vision_embedding,
            chat_service=chat_service,
            llm=llm,
            prompt_template=prompt_template,
        )

    def create_audio_service(
        self,
        whisper_model: Optional[Union[str, Any]] = None,
        whisper_device: Optional[str] = None,
        whisper_language: Optional[str] = None,
        tts_provider: Optional[Union[str, Any]] = None,
        tts_api_key: Optional[str] = None,
        tts_model: Optional[str] = None,
        tts_voice: Optional[str] = None,
        vector_store: Optional["VectorStoreProtocol"] = None,
        embedding_model: Optional[Any] = None,
    ) -> IAudioService:
        """
        Audio 서비스 생성 (의존성 주입)

        Args:
            whisper_model: Whisper 모델 크기
            whisper_device: Whisper 디바이스
            whisper_language: Whisper 언어
            tts_provider: TTS 제공자
            tts_api_key: TTS API 키
            tts_model: TTS 모델
            tts_voice: TTS 음성
            vector_store: 벡터 스토어 (AudioRAG용)
            embedding_model: 임베딩 모델 (AudioRAG용)

        Returns:
            IAudioService: Audio 서비스 인스턴스

        책임:
            - 의존성 주입만
            - 비즈니스 로직 없음
        """
        from .impl.audio_service_impl import AudioServiceImpl

        return AudioServiceImpl(
            whisper_model=whisper_model,
            whisper_device=whisper_device,
            whisper_language=whisper_language,
            tts_provider=tts_provider,
            tts_api_key=tts_api_key,
            tts_model=tts_model,
            tts_voice=tts_voice,
            vector_store=vector_store,
            embedding_model=embedding_model,
        )

    def create_evaluation_service(
        self,
        client: Optional[Any] = None,
        embedding_model: Optional[Any] = None,
    ) -> IEvaluationService:
        """
        Evaluation 서비스 생성 (의존성 주입)

        Args:
            client: LLM 클라이언트 (선택적)
            embedding_model: 임베딩 모델 (선택적)

        Returns:
            IEvaluationService: Evaluation 서비스 인스턴스

        책임:
            - 의존성 주입만
            - 비즈니스 로직 없음
        """
        from .impl.evaluation_service_impl import EvaluationServiceImpl

        return EvaluationServiceImpl(client=client, embedding_model=embedding_model)

    def create_all_services(self) -> Dict[str, Any]:
        """
        모든 서비스 생성 (의존성 주입)

        Returns:
            dict: 서비스 인스턴스 딕셔너리

        책임:
            - 의존성 주입만
            - 비즈니스 로직 없음
        """
        chat_service = self.create_chat_service()
        rag_service = self.create_rag_service(chat_service)
        agent_service = self.create_agent_service(chat_service)
        chain_service = self.create_chain_service(chat_service)
        graph_service = self.create_graph_service()
        state_graph_service = self.create_state_graph_service()
        multi_agent_service = self.create_multi_agent_service()
        web_search_service = self.create_web_search_service()
        evaluation_service = self.create_evaluation_service()
        finetuning_service = self.create_finetuning_service()

        return {
            "chat": chat_service,
            "rag": rag_service,
            "agent": agent_service,
            "chain": chain_service,
            "graph": graph_service,
            "state_graph": state_graph_service,
            "multi_agent": multi_agent_service,
            "web_search": web_search_service,
            "evaluation": evaluation_service,
            "finetuning": finetuning_service,
        }
