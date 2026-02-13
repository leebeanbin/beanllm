"""
Knowledge Graph convenience methods - quick_build, find_*, ask, merge_graphs.

Extracted from knowledge_graph_facade for smaller, focused modules.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

from beanllm.dto.response.graph.kg_response import BuildGraphResponse


class KnowledgeGraphConvenienceMixin:
    """
    Mixin providing convenience methods for KnowledgeGraph facade.

    Expects the mixed-in class to have: build_graph(), query_graph(), graph_rag()
    and _handler.
    """

    async def quick_build(
        self,
        documents: List[str],
        graph_id: Optional[str] = None,
    ) -> BuildGraphResponse:
        """
        간단한 그래프 구축 (모든 엔티티/관계 타입)

        Args:
            documents: 문서 목록
            graph_id: 그래프 ID (optional)

        Returns:
            BuildGraphResponse: 그래프 정보

        Example:
            ```python
            response = await kg.quick_build(
                documents=["Apple was founded by Steve Jobs in 1976."]
            )
            ```
        """
        return cast(
            BuildGraphResponse,
            await self.build_graph(  # type: ignore[attr-defined]
                documents=documents,
                graph_id=graph_id,
                entity_types=None,
                relation_types=None,
            ),
        )

    async def find_entities_by_type(
        self,
        graph_id: str,
        entity_type: str,
    ) -> List[Dict[str, Any]]:
        """
        타입별 엔티티 검색

        Args:
            graph_id: 그래프 ID
            entity_type: 엔티티 타입

        Returns:
            List[Dict]: 엔티티 목록

        Example:
            ```python
            persons = await kg.find_entities_by_type(
                graph_id="tech_companies",
                entity_type="PERSON"
            )
            ```
        """
        response = await self.query_graph(  # type: ignore[attr-defined]
            graph_id=graph_id,
            query_type="find_entities_by_type",
            params={"entity_type": entity_type},
        )
        return cast(List[Dict[str, Any]], response.results)

    async def find_entities_by_name(
        self,
        graph_id: str,
        name: str,
        fuzzy: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        이름 기반 엔티티 검색

        Args:
            graph_id: 그래프 ID
            name: 엔티티 이름
            fuzzy: Fuzzy matching 사용 여부

        Returns:
            List[Dict]: 엔티티 목록

        Example:
            ```python
            entities = await kg.find_entities_by_name(
                graph_id="tech_companies",
                name="Steve",
                fuzzy=True
            )
            ```
        """
        response = await self.query_graph(  # type: ignore[attr-defined]
            graph_id=graph_id,
            query_type="find_entities_by_name",
            params={"name": name, "fuzzy": fuzzy},
        )
        return cast(List[Dict[str, Any]], response.results)

    async def find_related_entities(
        self,
        graph_id: str,
        entity_id: str,
        relation_type: Optional[str] = None,
        max_hops: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        관계 기반 엔티티 탐색

        Args:
            graph_id: 그래프 ID
            entity_id: 시작 엔티티 ID
            relation_type: 관계 타입 필터 (optional)
            max_hops: 최대 hop 수 (default: 1)

        Returns:
            List[Dict]: 관련 엔티티 목록

        Example:
            ```python
            related = await kg.find_related_entities(
                graph_id="tech_companies",
                entity_id="steve_jobs",
                max_hops=2
            )
            ```
        """
        response = await self.query_graph(  # type: ignore[attr-defined]
            graph_id=graph_id,
            query_type="find_related_entities",
            params={
                "entity_id": entity_id,
                "relation_type": relation_type,
                "max_hops": max_hops,
            },
        )
        return cast(List[Dict[str, Any]], response.results)

    async def find_path(
        self,
        graph_id: str,
        source_id: str,
        target_id: str,
    ) -> Optional[List[str]]:
        """
        두 엔티티 간 최단 경로 탐색

        Args:
            graph_id: 그래프 ID
            source_id: 시작 엔티티 ID
            target_id: 목표 엔티티 ID

        Returns:
            Optional[List[str]]: 경로 (엔티티 ID 리스트)

        Example:
            ```python
            path = await kg.find_path(
                graph_id="tech_companies",
                source_id="steve_jobs",
                target_id="apple"
            )

            if path:
                print(" -> ".join(path))
            ```
        """
        response = await self.query_graph(  # type: ignore[attr-defined]
            graph_id=graph_id,
            query_type="find_shortest_path",
            params={"source_id": source_id, "target_id": target_id},
        )

        if response.results:
            path = response.results[0].get("path")
            return cast(Optional[List[str]], path) if isinstance(path, list) else None
        return None

    async def get_entity_details(
        self,
        graph_id: str,
        entity_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        엔티티 상세 정보 (속성, outgoing/incoming 관계)

        Args:
            graph_id: 그래프 ID
            entity_id: 엔티티 ID

        Returns:
            Optional[Dict]: 엔티티 상세 정보

        Example:
            ```python
            details = await kg.get_entity_details(
                graph_id="tech_companies",
                entity_id="steve_jobs"
            )

            print(f"Name: {details['name']}")
            print(f"Type: {details['type']}")
            print(f"Outgoing: {len(details['outgoing_relations'])}")
            print(f"Incoming: {len(details['incoming_relations'])}")
            ```
        """
        response = await self.query_graph(  # type: ignore[attr-defined]
            graph_id=graph_id,
            query_type="get_entity_details",
            params={"entity_id": entity_id},
        )

        if response.results:
            return cast(Dict[str, Any], response.results[0])
        return None

    async def ask(
        self,
        query: str,
        graph_id: str,
    ) -> str:
        """
        Graph RAG 질의 (간단한 답변 반환)

        Args:
            query: 사용자 질의
            graph_id: 그래프 ID

        Returns:
            str: 답변 (hybrid results의 요약)

        Example:
            ```python
            answer = await kg.ask(
                query="Who founded Apple?",
                graph_id="tech_companies"
            )
            print(answer)
            ```
        """
        response = await self.graph_rag(  # type: ignore[attr-defined]
            query=query, graph_id=graph_id
        )

        if response.hybrid_results:
            top_results = response.hybrid_results[:3]
            answer_parts = []

            for result in top_results:
                entity = result.get("entity", {})
                if not isinstance(entity, dict):
                    entity = {}
                name = entity.get("name", "Unknown")
                entity_type = entity.get("type", "UNKNOWN")
                score = result.get("score", 0.0)

                answer_parts.append(f"- {name} ({entity_type}): {score:.2f}")

            return "Based on the knowledge graph:\n" + "\n".join(answer_parts)

        return "No relevant information found in the knowledge graph."

    async def merge_graphs(
        self,
        graph_ids: List[str],
        new_graph_id: Optional[str] = None,
    ) -> BuildGraphResponse:
        """
        여러 그래프 병합

        Args:
            graph_ids: 병합할 그래프 ID 목록
            new_graph_id: 새 그래프 ID (optional)

        Returns:
            BuildGraphResponse: 병합된 그래프 정보

        Example:
            ```python
            response = await kg.merge_graphs(
                graph_ids=["graph1", "graph2", "graph3"],
                new_graph_id="merged_graph"
            )
            ```
        """
        raise NotImplementedError("merge_graphs is not yet implemented")
