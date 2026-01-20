"""
GraphRAG - 그래프 기반 RAG
SOLID 원칙:
- SRP: 그래프 기반 검색만 담당
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import networkx as nx

from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class GraphRAG:
    """
    그래프 기반 RAG

    Example:
        ```python
        graph_rag = GraphRAG(graph, querier)

        # Entity-centric retrieval
        results = graph_rag.entity_centric_retrieval(
            query="Tell me about Steve Jobs",
            top_k=5
        )

        # Path-based reasoning
        results = graph_rag.path_reasoning(
            query="How is Steve Jobs related to Apple?",
            max_path_length=3
        )
        ```
    """

    def __init__(
        self,
        graph: nx.Graph,
        querier: Optional[Any] = None,
    ) -> None:
        """Initialize GraphRAG"""
        self.graph = graph
        self.querier = querier
        logger.info("GraphRAG initialized")

    def entity_centric_retrieval(
        self,
        query: str,
        top_k: int = 5,
        max_hops: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        엔티티 중심 검색

        Args:
            query: 쿼리
            top_k: 반환할 결과 수
            max_hops: 최대 홉 수

        Returns:
            List[Dict]: 검색 결과
        """
        logger.info(f"Entity-centric retrieval: {query}")

        # Step 1: Extract entities from query (placeholder)
        # In production, use NER or LLM
        query_entities = self._extract_query_entities(query)

        # Step 2: Find entities in graph
        relevant_entities = []
        for entity_name in query_entities:
            if self.querier:
                matches = self.querier.find_entities_by_name(entity_name, fuzzy=True)
                relevant_entities.extend(matches)

        # Step 3: Expand to neighbors
        expanded_entities = set()
        for entity in relevant_entities:
            entity_id = entity["id"]

            if self.querier:
                neighbors = self.querier.find_related_entities(
                    entity_id, max_hops=max_hops
                )
                for neighbor in neighbors:
                    expanded_entities.add(neighbor["id"])

        # Step 4: Score and rank
        results = []
        for entity_id in list(expanded_entities)[:top_k]:
            if entity_id in self.graph:
                node_data = self.graph.nodes[entity_id]
                results.append({
                    "id": entity_id,
                    "name": node_data.get("name"),
                    "type": node_data.get("type"),
                    "description": node_data.get("description", ""),
                    "score": 1.0,  # Placeholder scoring
                })

        return results

    def path_reasoning(
        self,
        query: str,
        max_path_length: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        경로 기반 추론

        Args:
            query: 쿼리
            max_path_length: 최대 경로 길이

        Returns:
            List[Dict]: 추론 결과 (경로 포함)
        """
        logger.info(f"Path reasoning: {query}")

        # Extract entity pairs from query
        entity_pairs = self._extract_entity_pairs(query)

        results = []
        for source, target in entity_pairs:
            # Find source and target in graph
            source_matches = self.querier.find_entities_by_name(source, fuzzy=True) if self.querier else []
            target_matches = self.querier.find_entities_by_name(target, fuzzy=True) if self.querier else []

            for source_entity in source_matches[:1]:
                for target_entity in target_matches[:1]:
                    source_id = source_entity["id"]
                    target_id = target_entity["id"]

                    # Find path
                    path = self.querier.find_shortest_path(source_id, target_id) if self.querier else None

                    if path and len(path) <= max_path_length + 1:
                        # Build path description
                        path_desc = self._describe_path(path)

                        results.append({
                            "source": source,
                            "target": target,
                            "path": path,
                            "path_length": len(path) - 1,
                            "description": path_desc,
                        })

        return results

    def hybrid_retrieval(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        하이브리드 검색 (엔티티 + 경로)

        Args:
            query: 쿼리
            top_k: 반환할 결과 수

        Returns:
            List[Dict]: 검색 결과
        """
        # Entity-centric results
        entity_results = self.entity_centric_retrieval(query, top_k=top_k // 2)

        # Path-based results
        path_results = self.path_reasoning(query, max_path_length=3)

        # Combine
        combined = entity_results + path_results
        return combined[:top_k]

    def _extract_query_entities(self, query: str) -> List[str]:
        """쿼리에서 엔티티 추출 (placeholder)"""
        # Simple word extraction (placeholder)
        # In production, use NER or LLM
        words = query.split()
        entities = [w for w in words if w[0].isupper()]
        return entities

    def _extract_entity_pairs(self, query: str) -> List[tuple]:
        """쿼리에서 엔티티 쌍 추출 (placeholder)"""
        # Placeholder
        entities = self._extract_query_entities(query)
        if len(entities) >= 2:
            return [(entities[0], entities[1])]
        return []

    def _describe_path(self, path: List[str]) -> str:
        """경로 설명 생성"""
        descriptions = []

        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]

            source_name = self.graph.nodes[source].get("name", source)
            target_name = self.graph.nodes[target].get("name", target)

            if self.graph.has_edge(source, target):
                edge_type = self.graph.edges[source, target].get("type", "related_to")
                descriptions.append(f"{source_name} -{edge_type}-> {target_name}")

        return " → ".join(descriptions)
