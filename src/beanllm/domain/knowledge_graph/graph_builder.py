"""
GraphBuilder - NetworkX 기반 지식 그래프 구축
SOLID 원칙:
- SRP: 그래프 구축만 담당
- OCP: 새로운 그래프 타입 추가 가능
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import networkx as nx

from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class GraphBuilder:
    """
    NetworkX 기반 지식 그래프 빌더

    책임:
    - 엔티티와 관계로부터 그래프 구축
    - 그래프 업데이트 (증분)
    - 중복 제거
    - 그래프 통계

    Example:
        ```python
        from beanllm.domain.knowledge_graph import (
            EntityExtractor,
            RelationExtractor,
            GraphBuilder
        )

        # Extract entities and relations
        entity_extractor = EntityExtractor()
        relation_extractor = RelationExtractor()

        entities = entity_extractor.extract_entities(text)
        relations = relation_extractor.extract_relations(entities, text)

        # Build graph
        builder = GraphBuilder()
        graph = builder.build_graph(entities, relations)

        # Query graph
        print(f"Nodes: {graph.number_of_nodes()}")
        print(f"Edges: {graph.number_of_edges()}")

        # Get neighbors
        neighbors = builder.get_neighbors(graph, entity_id)
        ```
    """

    def __init__(self, directed: bool = True) -> None:
        """
        Initialize graph builder

        Args:
            directed: 방향 그래프 여부 (default: True)
        """
        self.directed = directed
        logger.info(f"GraphBuilder initialized (directed={directed})")

    def build_graph(
        self,
        entities: List[Any],  # List[Entity]
        relations: List[Any],  # List[Relation]
    ) -> nx.Graph:
        """
        엔티티와 관계로부터 그래프 구축

        Args:
            entities: 엔티티 리스트
            relations: 관계 리스트

        Returns:
            nx.Graph: NetworkX 그래프
        """
        logger.info(f"Building graph: {len(entities)} entities, {len(relations)} relations")

        # Create graph
        if self.directed:
            graph = nx.DiGraph()
        else:
            graph = nx.Graph()

        # Add nodes (entities)
        for entity in entities:
            graph.add_node(
                entity.id,
                name=entity.name,
                type=entity.type.value if hasattr(entity.type, "value") else str(entity.type),
                description=entity.description,
                properties=entity.properties,
                confidence=entity.confidence,
            )

        # Add edges (relations)
        for relation in relations:
            graph.add_edge(
                relation.source_id,
                relation.target_id,
                type=relation.type.value if hasattr(relation.type, "value") else str(relation.type),
                description=relation.description,
                properties=relation.properties,
                confidence=relation.confidence,
            )

        logger.info(
            f"Graph built: {graph.number_of_nodes()} nodes, " f"{graph.number_of_edges()} edges"
        )

        return graph

    def add_entities(
        self,
        graph: nx.Graph,
        entities: List[Any],
    ) -> nx.Graph:
        """
        그래프에 엔티티 추가 (증분)

        Args:
            graph: 기존 그래프
            entities: 추가할 엔티티 리스트

        Returns:
            nx.Graph: 업데이트된 그래프
        """
        for entity in entities:
            if entity.id not in graph:
                graph.add_node(
                    entity.id,
                    name=entity.name,
                    type=entity.type.value if hasattr(entity.type, "value") else str(entity.type),
                    description=entity.description,
                    properties=entity.properties,
                    confidence=entity.confidence,
                )
            else:
                # Update existing node
                graph.nodes[entity.id].update(
                    {
                        "name": entity.name,
                        "description": entity.description,
                        "properties": entity.properties,
                        "confidence": entity.confidence,
                    }
                )

        return graph

    def add_relations(
        self,
        graph: nx.Graph,
        relations: List[Any],
    ) -> nx.Graph:
        """
        그래프에 관계 추가 (증분)

        Args:
            graph: 기존 그래프
            relations: 추가할 관계 리스트

        Returns:
            nx.Graph: 업데이트된 그래프
        """
        for relation in relations:
            if not graph.has_edge(relation.source_id, relation.target_id):
                graph.add_edge(
                    relation.source_id,
                    relation.target_id,
                    type=relation.type.value
                    if hasattr(relation.type, "value")
                    else str(relation.type),
                    description=relation.description,
                    properties=relation.properties,
                    confidence=relation.confidence,
                )
            else:
                # Update existing edge
                graph.edges[relation.source_id, relation.target_id].update(
                    {
                        "type": relation.type.value
                        if hasattr(relation.type, "value")
                        else str(relation.type),
                        "description": relation.description,
                        "properties": relation.properties,
                        "confidence": relation.confidence,
                    }
                )

        return graph

    def merge_graphs(
        self,
        graph1: nx.Graph,
        graph2: nx.Graph,
    ) -> nx.Graph:
        """
        두 그래프 병합

        Args:
            graph1: 첫 번째 그래프
            graph2: 두 번째 그래프

        Returns:
            nx.Graph: 병합된 그래프
        """
        logger.info(
            f"Merging graphs: G1({graph1.number_of_nodes()}, {graph1.number_of_edges()}), "
            f"G2({graph2.number_of_nodes()}, {graph2.number_of_edges()})"
        )

        # Create new graph
        if self.directed:
            merged = nx.DiGraph(graph1)
        else:
            merged = nx.Graph(graph1)

        # Add nodes from graph2
        for node, data in graph2.nodes(data=True):
            if node in merged:
                # Merge properties
                merged.nodes[node].update(data)
            else:
                merged.add_node(node, **data)

        # Add edges from graph2
        for u, v, data in graph2.edges(data=True):
            if merged.has_edge(u, v):
                # Merge properties
                merged.edges[u, v].update(data)
            else:
                merged.add_edge(u, v, **data)

        logger.info(
            f"Merged graph: {merged.number_of_nodes()} nodes, " f"{merged.number_of_edges()} edges"
        )

        return merged

    def get_neighbors(
        self,
        graph: nx.Graph,
        entity_id: str,
        max_hops: int = 1,
    ) -> List[str]:
        """
        이웃 노드 조회

        Args:
            graph: 그래프
            entity_id: 엔티티 ID
            max_hops: 최대 홉 수

        Returns:
            List[str]: 이웃 노드 ID 리스트
        """
        if entity_id not in graph:
            return []

        if max_hops == 1:
            # Direct neighbors
            if self.directed:
                predecessors = set(graph.predecessors(entity_id))
                successors = set(graph.successors(entity_id))
                return list(predecessors | successors)
            else:
                return list(graph.neighbors(entity_id))

        else:
            # Multi-hop neighbors (BFS)
            visited = set()
            queue = [(entity_id, 0)]
            neighbors = []

            while queue:
                current, depth = queue.pop(0)

                if depth >= max_hops:
                    continue

                if current in visited:
                    continue

                visited.add(current)

                if current != entity_id:
                    neighbors.append(current)

                # Add neighbors to queue
                if self.directed:
                    next_nodes = set(graph.predecessors(current)) | set(graph.successors(current))
                else:
                    next_nodes = set(graph.neighbors(current))

                for next_node in next_nodes:
                    if next_node not in visited:
                        queue.append((next_node, depth + 1))

            return neighbors

    def find_path(
        self,
        graph: nx.Graph,
        source_id: str,
        target_id: str,
        max_length: int = 5,
    ) -> Optional[List[str]]:
        """
        두 엔티티 간 경로 찾기

        Args:
            graph: 그래프
            source_id: 시작 엔티티 ID
            target_id: 목표 엔티티 ID
            max_length: 최대 경로 길이

        Returns:
            Optional[List[str]]: 경로 (노드 ID 리스트) 또는 None
        """
        if source_id not in graph or target_id not in graph:
            return None

        try:
            if self.directed:
                path = nx.shortest_path(graph, source_id, target_id, weight=None)
            else:
                path = nx.shortest_path(graph, source_id, target_id, weight=None)

            if len(path) <= max_length + 1:  # +1 for nodes count
                return path
            else:
                return None

        except nx.NetworkXNoPath:
            return None

    def get_subgraph(
        self,
        graph: nx.Graph,
        entity_ids: List[str],
        include_edges: bool = True,
    ) -> nx.Graph:
        """
        서브그래프 추출

        Args:
            graph: 원본 그래프
            entity_ids: 포함할 엔티티 ID 리스트
            include_edges: 엔티티 간 엣지 포함 여부

        Returns:
            nx.Graph: 서브그래프
        """
        if include_edges:
            return graph.subgraph(entity_ids).copy()
        else:
            # Only nodes, no edges
            if self.directed:
                subgraph = nx.DiGraph()
            else:
                subgraph = nx.Graph()

            for node_id in entity_ids:
                if node_id in graph:
                    subgraph.add_node(node_id, **graph.nodes[node_id])

            return subgraph

    def get_graph_statistics(
        self,
        graph: nx.Graph,
    ) -> Dict[str, Any]:
        """
        그래프 통계

        Args:
            graph: 그래프

        Returns:
            Dict[str, Any]: 통계 정보
        """
        stats = {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "is_directed": self.directed,
        }

        # Node types distribution
        node_types: Dict[str, int] = {}
        for node, data in graph.nodes(data=True):
            node_type = data.get("type", "unknown")
            node_types[node_type] = node_types.get(node_type, 0) + 1

        stats["node_type_distribution"] = node_types

        # Edge types distribution
        edge_types: Dict[str, int] = {}
        for u, v, data in graph.edges(data=True):
            edge_type = data.get("type", "unknown")
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

        stats["edge_type_distribution"] = edge_types

        # Connected components
        if self.directed:
            stats["num_weakly_connected_components"] = nx.number_weakly_connected_components(graph)
            stats["num_strongly_connected_components"] = nx.number_strongly_connected_components(
                graph
            )
        else:
            stats["num_connected_components"] = nx.number_connected_components(graph)

        # Degree statistics
        if graph.number_of_nodes() > 0:
            degrees = [d for n, d in graph.degree()]
            stats["avg_degree"] = sum(degrees) / len(degrees)
            stats["max_degree"] = max(degrees)
            stats["min_degree"] = min(degrees)

        return stats

    def export_to_dict(
        self,
        graph: nx.Graph,
    ) -> Dict[str, Any]:
        """
        그래프를 딕셔너리로 변환

        Args:
            graph: 그래프

        Returns:
            Dict[str, Any]: 그래프 데이터
        """
        return {
            "nodes": [{"id": node, **data} for node, data in graph.nodes(data=True)],
            "edges": [{"source": u, "target": v, **data} for u, v, data in graph.edges(data=True)],
            "directed": self.directed,
        }

    def import_from_dict(
        self,
        data: Dict[str, Any],
    ) -> nx.Graph:
        """
        딕셔너리로부터 그래프 생성

        Args:
            data: 그래프 데이터

        Returns:
            nx.Graph: 그래프
        """
        if data.get("directed", True):
            graph = nx.DiGraph()
        else:
            graph = nx.Graph()

        # Add nodes
        for node_data in data.get("nodes", []):
            node_id = node_data.pop("id")
            graph.add_node(node_id, **node_data)

        # Add edges
        for edge_data in data.get("edges", []):
            source = edge_data.pop("source")
            target = edge_data.pop("target")
            graph.add_edge(source, target, **edge_data)

        return graph


def build_graph_simple(
    entities: List[Any],
    relations: List[Any],
    directed: bool = True,
) -> nx.Graph:
    """
    간단한 그래프 구축 (편의 함수)

    Args:
        entities: 엔티티 리스트
        relations: 관계 리스트
        directed: 방향 그래프 여부

    Returns:
        nx.Graph: 구축된 그래프
    """
    builder = GraphBuilder(directed=directed)
    return builder.build_graph(entities, relations)
