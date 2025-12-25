"""
HandlerFactory - Handler 의존성 주입 팩토리
SOLID 원칙:
- DIP: 인터페이스에 의존
- OCP: 확장 가능
- SRP: 의존성 관리만 담당
- DRY: 공통 생성 로직 재사용
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Type

from ..service.factory import ServiceFactory
from .agent_handler import AgentHandler
from .audio_handler import AudioHandler
from .chain_handler import ChainHandler
from .chat_handler import ChatHandler
from .graph_handler import GraphHandler
from .multi_agent_handler import MultiAgentHandler
from .rag_handler import RAGHandler
from .state_graph_handler import StateGraphHandler
from .vision_rag_handler import VisionRAGHandler
from .web_search_handler import WebSearchHandler

if TYPE_CHECKING:
    from ..service.audio_service import IAudioService
    from ..service.vision_rag_service import IVisionRAGService


class HandlerFactory:
    """
    Handler 팩토리

    책임:
    - Handler 인스턴스 생성 및 의존성 주입
    - 의존성 관리만 (비즈니스 로직 없음)

    SOLID:
    - SRP: 의존성 관리만
    - DIP: 인터페이스에 의존
    - OCP: 확장 가능
    """

    def __init__(self, service_factory: ServiceFactory) -> None:
        """
        의존성 주입을 통한 생성자

        Args:
            service_factory: 서비스 팩토리
        """
        self._service_factory = service_factory

    def create_chat_handler(self) -> ChatHandler:
        """
        채팅 Handler 생성 (의존성 주입)

        Returns:
            ChatHandler: 채팅 Handler 인스턴스
        """
        chat_service = self._service_factory.create_chat_service()
        return ChatHandler(chat_service)

    def create_rag_handler(self) -> RAGHandler:
        """
        RAG Handler 생성 (의존성 주입)

        Returns:
            RAGHandler: RAG Handler 인스턴스
        """
        rag_service = self._service_factory.create_rag_service()
        return RAGHandler(rag_service)

    def create_agent_handler(self) -> AgentHandler:
        """
        에이전트 Handler 생성 (의존성 주입)

        Returns:
            AgentHandler: 에이전트 Handler 인스턴스
        """
        agent_service = self._service_factory.create_agent_service()
        return AgentHandler(agent_service)

    def _create_handler(self, handler_class: Type[Any], service: Any) -> Any:
        """
        Handler 생성 공통 로직 (DRY 원칙)

        Args:
            handler_class: Handler 클래스
            service: Service 인스턴스

        Returns:
            Handler 인스턴스
        """
        return handler_class(service)

    def create_chain_handler(self) -> ChainHandler:
        """
        Chain Handler 생성 (의존성 주입)

        Returns:
            ChainHandler: Chain Handler 인스턴스
        """
        chain_service = self._service_factory.create_chain_service()
        return ChainHandler(chain_service)

    def create_graph_handler(self) -> GraphHandler:
        """
        Graph Handler 생성 (의존성 주입)

        Returns:
            GraphHandler: Graph Handler 인스턴스
        """
        graph_service = self._service_factory.create_graph_service()
        return GraphHandler(graph_service)

    def create_state_graph_handler(self) -> StateGraphHandler:
        """
        StateGraph Handler 생성 (의존성 주입)

        Returns:
            StateGraphHandler: StateGraph Handler 인스턴스
        """
        state_graph_service = self._service_factory.create_state_graph_service()
        return StateGraphHandler(state_graph_service)

    def create_multi_agent_handler(self) -> MultiAgentHandler:
        """
        Multi-Agent Handler 생성 (의존성 주입)

        Returns:
            MultiAgentHandler: Multi-Agent Handler 인스턴스
        """
        multi_agent_service = self._service_factory.create_multi_agent_service()
        return MultiAgentHandler(multi_agent_service)

    def create_web_search_handler(self) -> WebSearchHandler:
        """
        Web Search Handler 생성 (의존성 주입)

        Returns:
            WebSearchHandler: Web Search Handler 인스턴스
        """
        web_search_service = self._service_factory.create_web_search_service()
        return WebSearchHandler(web_search_service)

    def create_vision_rag_handler(
        self,
        vision_rag_service: "IVisionRAGService",
    ) -> VisionRAGHandler:
        """
        Vision RAG Handler 생성 (의존성 주입)

        Args:
            vision_rag_service: Vision RAG 서비스 (필수)

        Returns:
            VisionRAGHandler: Vision RAG Handler 인스턴스
        """
        return VisionRAGHandler(vision_rag_service)

    def create_audio_handler(
        self,
        audio_service: "IAudioService",
    ) -> AudioHandler:
        """
        Audio Handler 생성 (의존성 주입)

        Args:
            audio_service: Audio 서비스 (필수)

        Returns:
            AudioHandler: Audio Handler 인스턴스
        """
        return AudioHandler(audio_service)

    def create_all_handlers(self) -> Dict[str, Any]:
        """
        모든 Handler 생성 (의존성 주입)

        Returns:
            dict: Handler 인스턴스 딕셔너리
        """
        return {
            "chat": self.create_chat_handler(),
            "rag": self.create_rag_handler(),
            "agent": self.create_agent_handler(),
            "chain": self.create_chain_handler(),
            "graph": self.create_graph_handler(),
            "state_graph": self.create_state_graph_handler(),
            "multi_agent": self.create_multi_agent_handler(),
            "web_search": self.create_web_search_handler(),
            "evaluation": self.create_evaluation_handler(),
            "finetuning": self.create_finetuning_handler(),
        }
