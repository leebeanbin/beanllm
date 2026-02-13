"""
GraphQuerier - 그래프 쿼리 인터페이스
SOLID 원칙:
- SRP: 그래프 쿼리만 담당
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

import networkx as nx

from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class GraphQuerier:
    """
    그래프 쿼리 인터페이스

    Example:
        ```python
        querier = GraphQuerier(graph)

        # Find entities by type
        people = querier.find_entities_by_type("person")

        # Find related entities
        related = querier.find_related_entities("steve_jobs", max_hops=2)

        # Find paths
        path = querier.find_shortest_path("steve_jobs", "apple")
        ```
    """

    def __init__(self, graph: nx.Graph) -> None:
        """Initialize querier with graph"""
        self.graph = graph
        logger.info("GraphQuerier initialized")

    def find_entities_by_type(
        self,
        entity_type: str,
    ) -> List[Dict[str, Any]]:
        """엔티티 타입으로 검색"""
        results = []

        for node, data in self.graph.nodes(data=True):
            if data.get("type") == entity_type:
                results.append({"id": node, **data})

        return results

    def find_entities_by_name(
        self,
        name: str,
        fuzzy: bool = False,
    ) -> List[Dict[str, Any]]:
        """이름으로 엔티티 검색"""
        results = []
        search_name = name.lower()

        for node, data in self.graph.nodes(data=True):
            node_name = data.get("name", "").lower()

            if fuzzy:
                if search_name in node_name:
                    results.append({"id": node, **data})
            else:
                if node_name == search_name:
                    results.append({"id": node, **data})

        return results

    def find_related_entities(
        self,
        entity_id: str,
        relation_type: Optional[str] = None,
        max_hops: int = 1,
    ) -> List[Dict[str, Any]]:
        """관련 엔티티 검색"""
        if entity_id not in self.graph:
            return []

        related = []
        visited = set()
        queue = [(entity_id, 0)]

        while queue:
            current, depth = queue.pop(0)

            if depth > max_hops:
                continue

            if current in visited:
                continue

            visited.add(current)

            if current != entity_id:
                node_data = self.graph.nodes[current]
                related.append({"id": current, **node_data})

            # Add neighbors
            for neighbor in self.graph.neighbors(current):
                if neighbor not in visited:
                    # Check relation type if specified
                    if relation_type:
                        edge_data = self.graph.edges[current, neighbor]
                        if edge_data.get("type") == relation_type:
                            queue.append((neighbor, depth + 1))
                    else:
                        queue.append((neighbor, depth + 1))

        return related

    def find_shortest_path(
        self,
        source_id: str,
        target_id: str,
    ) -> Optional[List[str]]:
        """최단 경로 찾기"""
        if source_id not in self.graph or target_id not in self.graph:
            return None

        try:
            path = nx.shortest_path(self.graph, source_id, target_id)
            return cast(List[str], path)
        except nx.NetworkXNoPath:
            return None

    def get_entity_details(
        self,
        entity_id: str,
    ) -> Optional[Dict[str, Any]]:
        """엔티티 상세 정보"""
        if entity_id not in self.graph:
            return None

        data = self.graph.nodes[entity_id].copy()
        data["id"] = entity_id

        # Add relations
        data["outgoing_relations"] = []
        data["incoming_relations"] = []

        if isinstance(self.graph, nx.DiGraph):
            for target in self.graph.successors(entity_id):
                edge_data = self.graph.edges[entity_id, target]
                data["outgoing_relations"].append(
                    {
                        "target": target,
                        **edge_data,
                    }
                )

            for source in self.graph.predecessors(entity_id):
                edge_data = self.graph.edges[source, entity_id]
                data["incoming_relations"].append(
                    {
                        "source": source,
                        **edge_data,
                    }
                )
        else:
            for neighbor in self.graph.neighbors(entity_id):
                edge_data = self.graph.edges[entity_id, neighbor]
                data["outgoing_relations"].append(
                    {
                        "target": neighbor,
                        **edge_data,
                    }
                )

        return cast(Dict[str, Any], data)
