"""
KnowledgeGraph - Knowledge Graph Facade (공개 API)

책임:
- 사용자 친화적인 Knowledge Graph API 제공
- 편의 메서드 제공 (knowledge_graph_convenience)
- Handler 오케스트레이션

Core API here; convenience methods in knowledge_graph_convenience.py,
standalone functions in knowledge_graph_standalone.py (re-exported below).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

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
from beanllm.facade.advanced.knowledge_graph_convenience import (
    KnowledgeGraphConvenienceMixin,
)
from beanllm.handler.advanced.knowledge_graph_handler import KnowledgeGraphHandler

logger = logging.getLogger(__name__)


class KnowledgeGraph(KnowledgeGraphConvenienceMixin):
    """
    Knowledge Graph Facade (공개 API)

    Example:
        ```python
        from beanllm import KnowledgeGraph

        # 초기화
        kg = KnowledgeGraph(client=client)

        # 간단한 사용 (quick_build)
        graph_response = await kg.quick_build(
            documents=[
                "Apple was founded by Steve Jobs in 1976.",
                "Steve Jobs was the CEO of Apple.",
                "Apple is headquartered in Cupertino, California."
            ]
        )

        # 쿼리
        results = await kg.find_entities_by_type(
            graph_id=graph_response.graph_id,
            entity_type="PERSON"
        )

        # Graph RAG
        answer = await kg.ask(
            query="Who founded Apple?",
            graph_id=graph_response.graph_id
        )
        ```
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        handler: Optional[KnowledgeGraphHandler] = None,
    ) -> None:
        """
        초기화

        Args:
            client: LLM Client (optional)
            handler: KnowledgeGraphHandler (optional, 테스트용)
        """
        if handler:
            self._handler = handler
        else:
            from beanllm.utils.core.di_container import get_container

            container = get_container()
            service_factory = container.get_service_factory()
            service = service_factory.create_knowledge_graph_service(client=client)
            self._handler = KnowledgeGraphHandler(service=service)

        logger.info("KnowledgeGraph facade initialized")

    # --- 핵심 메서드 ---

    async def extract_entities(
        self,
        text: str,
        entity_types: Optional[List[str]] = None,
        resolve_coreferences: bool = True,
    ) -> EntitiesResponse:
        """
        문서에서 엔티티 추출 (LLM-based NER)

        Args:
            text: 입력 텍스트
            entity_types: 추출할 엔티티 타입 필터 (optional)
                - "person", "organization", "location", "event",
                  "product", "concept", "date", "number", "other"
            resolve_coreferences: 대명사/별칭 해결 여부 (default: True)

        Returns:
            EntitiesResponse: 추출된 엔티티 목록

        Example:
            ```python
            response = await kg.extract_entities(
                text="Apple was founded by Steve Jobs.",
                entity_types=["person", "organization"]
            )

            for entity in response.entities:
                print(f"{entity['name']} ({entity['type']})")
            ```
        """
        request = ExtractEntitiesRequest(
            document_id=str(uuid.uuid4()),
            text=text,
            entity_types=entity_types or [],
            resolve_coreferences=resolve_coreferences,
        )
        return await self._handler.handle_extract_entities(request)

    async def extract_relations(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        relation_types: Optional[List[str]] = None,
        infer_implicit: bool = True,
    ) -> RelationsResponse:
        """
        엔티티 간 관계 추출

        Args:
            text: 입력 텍스트
            entities: 엔티티 목록 (extract_entities 결과)
            relation_types: 추출할 관계 타입 필터 (optional)
                - "founded", "works_for", "located_in", "part_of",
                  "owns", "created", "produces", etc.
            infer_implicit: 암시적 관계 추론 여부 (default: True)

        Returns:
            RelationsResponse: 추출된 관계 목록

        Example:
            ```python
            # 엔티티 먼저 추출
            entities_response = await kg.extract_entities(
                text="Apple was founded by Steve Jobs."
            )

            # 관계 추출
            relations_response = await kg.extract_relations(
                text="Apple was founded by Steve Jobs.",
                entities=entities_response.entities
            )

            for relation in relations_response.relations:
                print(f"{relation['source_id']} --[{relation['type']}]--> {relation['target_id']}")
            ```
        """
        request = ExtractRelationsRequest(
            document_id=str(uuid.uuid4()),
            text=text,
            entities=entities,
            relation_types=relation_types or [],
            infer_implicit=infer_implicit,
        )
        return await self._handler.handle_extract_relations(request)

    async def build_graph(
        self,
        documents: List[str],
        graph_id: Optional[str] = None,
        entity_types: Optional[List[str]] = None,
        relation_types: Optional[List[str]] = None,
        persist_to_neo4j: bool = False,
        clear_existing: bool = False,
    ) -> BuildGraphResponse:
        """
        Knowledge Graph 구축 (NetworkX/Neo4j)

        Args:
            documents: 문서 목록
            graph_id: 그래프 ID (optional, 자동 생성)
            entity_types: 추출할 엔티티 타입 필터 (optional)
            relation_types: 추출할 관계 타입 필터 (optional)
            persist_to_neo4j: Neo4j에 저장 여부 (default: False)
            clear_existing: 기존 그래프 삭제 여부 (default: False)

        Returns:
            BuildGraphResponse: 그래프 정보

        Example:
            ```python
            response = await kg.build_graph(
                documents=[
                    "Apple was founded by Steve Jobs in 1976.",
                    "Steve Jobs was the CEO of Apple.",
                    "Apple is headquartered in Cupertino, California."
                ],
                entity_types=["person", "organization", "location"],
                persist_to_neo4j=True
            )

            print(f"Graph ID: {response.graph_id}")
            print(f"Nodes: {response.num_nodes}, Edges: {response.num_edges}")
            ```
        """
        request = BuildGraphRequest(
            documents=documents,
            graph_id=graph_id,
            entity_types=entity_types,
            relation_types=relation_types,
            persist_to_neo4j=persist_to_neo4j,
            clear_existing=clear_existing,
        )
        return await self._handler.handle_build_graph(request)

    async def query_graph(
        self,
        graph_id: str,
        query_type: str = "cypher",
        query: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryGraphResponse:
        """
        그래프 쿼리 (Cypher-like)

        Args:
            graph_id: 그래프 ID
            query_type: 쿼리 타입
                - "cypher": Cypher 쿼리 (Neo4j 전용)
                - "find_entities_by_type": 타입별 엔티티 검색
                - "find_entities_by_name": 이름 기반 검색
                - "find_related_entities": 관계 기반 탐색
                - "find_shortest_path": 최단 경로
                - "get_entity_details": 엔티티 상세 정보
            query: Cypher 쿼리 (query_type="cypher"일 때만 필요)
            params: 쿼리 파라미터

        Returns:
            QueryGraphResponse: 쿼리 결과

        Example:
            ```python
            # 타입별 검색
            response = await kg.query_graph(
                graph_id="tech_companies",
                query_type="find_entities_by_type",
                params={"entity_type": "PERSON"}
            )

            # 관계 기반 탐색
            response = await kg.query_graph(
                graph_id="tech_companies",
                query_type="find_related_entities",
                params={"entity_id": "steve_jobs", "max_hops": 2}
            )

            # Cypher 쿼리 (Neo4j)
            response = await kg.query_graph(
                graph_id="tech_companies",
                query_type="cypher",
                query="MATCH (p:PERSON)-[r:FOUNDED]->(o:ORGANIZATION) RETURN p, o"
            )
            ```
        """
        request = QueryGraphRequest(
            graph_id=graph_id,
            query_type=query_type,
            query=query or "",
            params=params or {},
        )
        return await self._handler.handle_query_graph(request)

    async def graph_rag(
        self,
        query: str,
        graph_id: str,
    ) -> GraphRAGResponse:
        """
        그래프 기반 RAG (entity-centric retrieval, path reasoning)

        Args:
            query: 사용자 질의
            graph_id: 그래프 ID

        Returns:
            GraphRAGResponse: RAG 응답

        Example:
            ```python
            response = await kg.graph_rag(
                query="Who founded Apple?",
                graph_id="tech_companies"
            )

            print(f"Entity Results: {len(response.entity_results)}")
            print(f"Path Results: {len(response.path_results)}")
            print(f"Hybrid Results: {len(response.hybrid_results)}")

            for result in response.hybrid_results:
                print(f"  - {result['entity']['name']}: {result['score']}")
            ```
        """
        return await self._handler.handle_graph_rag(
            query=query,
            graph_id=graph_id,
        )

    async def visualize_graph(self, graph_id: str) -> str:
        """
        그래프 시각화 (ASCII)

        Args:
            graph_id: 그래프 ID

        Returns:
            str: ASCII 그래프 다이어그램

        Example:
            ```python
            diagram = await kg.visualize_graph("tech_companies")
            print(diagram)
            ```
        """
        return await self._handler.handle_visualize_graph(graph_id)

    async def get_graph_stats(self, graph_id: str) -> Dict[str, Any]:
        """
        그래프 통계 (노드 수, 엣지 수, 밀도 등)

        Args:
            graph_id: 그래프 ID

        Returns:
            Dict: 그래프 통계

        Example:
            ```python
            stats = await kg.get_graph_stats("tech_companies")

            print(f"Nodes: {stats['num_nodes']}")
            print(f"Edges: {stats['num_edges']}")
            print(f"Density: {stats['density']:.3f}")
            print(f"Connected Components: {stats['num_connected_components']}")

            # 엔티티 타입별 개수
            for entity_type, count in stats['entity_type_counts'].items():
                print(f"  {entity_type}: {count}")
            ```
        """
        return await self._handler.handle_get_graph_stats(graph_id)


# Re-export standalone functions for backward compatibility
from beanllm.facade.advanced.knowledge_graph_standalone import (  # noqa: E402
    quick_graph_rag,
    quick_knowledge_graph,
)

__all__ = [
    "KnowledgeGraph",
    "quick_knowledge_graph",
    "quick_graph_rag",
]
