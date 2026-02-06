"""
IKnowledgeGraphService - Knowledge Graph 서비스 인터페이스
SOLID 원칙:
- ISP: Knowledge Graph 관련 메서드만 포함
- DIP: 인터페이스에 의존
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from beanllm.dto.request.graph.kg_request import (
    BuildGraphRequest,
    ExtractEntitiesRequest,
    ExtractRelationsRequest,
    QueryGraphRequest,
)
from beanllm.dto.response.graph.kg_response import (
    BuildGraphResponse,
    EntitiesResponse,
    GraphRAGResponse,
    QueryGraphResponse,
    RelationsResponse,
)


class IKnowledgeGraphService(ABC):
    """
    Knowledge Graph 서비스 인터페이스

    책임:
    - 엔티티/관계 추출, 그래프 구축, 그래프 기반 RAG 비즈니스 로직 정의

    SOLID:
    - ISP: Knowledge Graph 관련 메서드만
    - DIP: 구현체가 아닌 인터페이스에 의존
    """

    @abstractmethod
    async def extract_entities(self, request: ExtractEntitiesRequest) -> EntitiesResponse:
        """
        문서에서 엔티티 추출 (LLM-based NER)

        Args:
            request: 엔티티 추출 요청 DTO

        Returns:
            EntitiesResponse: 추출된 엔티티 목록
        """
        pass

    @abstractmethod
    async def extract_relations(self, request: ExtractRelationsRequest) -> RelationsResponse:
        """
        엔티티 간 관계 추출

        Args:
            request: 관계 추출 요청 DTO

        Returns:
            RelationsResponse: 추출된 관계 목록
        """
        pass

    @abstractmethod
    async def build_graph(self, request: BuildGraphRequest) -> BuildGraphResponse:
        """
        Knowledge Graph 구축 (NetworkX/Neo4j)

        Args:
            request: 그래프 구축 요청 DTO

        Returns:
            BuildGraphResponse: 그래프 정보
        """
        pass

    @abstractmethod
    async def query_graph(self, request: QueryGraphRequest) -> QueryGraphResponse:
        """
        그래프 쿼리 (Cypher-like)

        Args:
            request: 그래프 쿼리 요청 DTO

        Returns:
            QueryGraphResponse: 쿼리 결과
        """
        pass

    @abstractmethod
    async def graph_rag(self, query: str, graph_id: str) -> GraphRAGResponse:
        """
        그래프 기반 RAG (entity-centric retrieval, path reasoning)

        Args:
            query: 사용자 질의
            graph_id: 그래프 ID

        Returns:
            GraphRAGResponse: RAG 응답
        """
        pass

    @abstractmethod
    async def visualize_graph(self, graph_id: str) -> str:
        """
        그래프 시각화 (ASCII)

        Args:
            graph_id: 그래프 ID

        Returns:
            str: ASCII 그래프 다이어그램
        """
        pass

    @abstractmethod
    async def get_graph_stats(self, graph_id: str) -> Dict[str, Any]:
        """
        그래프 통계 (노드 수, 엣지 수, 밀도 등)

        Args:
            graph_id: 그래프 ID

        Returns:
            Dict: 그래프 통계
        """
        pass

    @abstractmethod
    async def list_graphs(self) -> List[Dict[str, Any]]:
        """
        모든 그래프 목록 조회

        Returns:
            List[Dict]: 그래프 목록 [{id, name, num_nodes, num_edges, ...}]
        """
        pass
