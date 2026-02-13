"""
Knowledge Graph 그래프 연산: 시각화, 통계, 목록, 삭제.

책임:
- 그래프 시각화 (ASCII)
- 그래프 통계 조회
- 그래프 목록/삭제
"""

from __future__ import annotations

from typing import Any, Dict, List, cast

import networkx as nx  # type: ignore[import-untyped]

from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


def _get_graph(service: Any, graph_id: str) -> nx.DiGraph:
    """
    그래프 가져오기.

    Args:
        service: 서비스 인스턴스 (_graphs 속성 보유)
        graph_id: 그래프 ID

    Returns:
        nx.DiGraph: NetworkX 그래프

    Raises:
        ValueError: 그래프가 없는 경우
    """
    if graph_id not in service._graphs:
        raise ValueError(f"Graph not found: {graph_id}")
    return service._graphs[graph_id]


async def visualize_graph(service: Any, graph_id: str) -> str:
    """
    그래프 시각화 (ASCII).

    Args:
        service: 서비스 인스턴스 (_graphs 속성 보유)
        graph_id: 그래프 ID

    Returns:
        str: ASCII 그래프 다이어그램
    """
    graph = _get_graph(service, graph_id)

    lines = []
    lines.append(f"Graph: {graph_id}")
    lines.append("=" * 50)
    lines.append(f"Nodes: {graph.number_of_nodes()}")
    lines.append(f"Edges: {graph.number_of_edges()}")
    lines.append("")
    lines.append("Entities:")
    lines.append("-" * 50)

    for node_id, node_data in list(graph.nodes(data=True))[:20]:
        name = node_data.get("name", node_id)
        entity_type = node_data.get("type", "UNKNOWN")
        lines.append(f"  [{entity_type}] {name} (id: {node_id})")

    if graph.number_of_nodes() > 20:
        lines.append(f"  ... and {graph.number_of_nodes() - 20} more")

    lines.append("")
    lines.append("Relations:")
    lines.append("-" * 50)

    for source, target, edge_data in list(graph.edges(data=True))[:20]:
        source_name = graph.nodes[source].get("name", source)
        target_name = graph.nodes[target].get("name", target)
        relation_type = edge_data.get("type", "UNKNOWN")
        lines.append(f"  {source_name} --[{relation_type}]--> {target_name}")

    if graph.number_of_edges() > 20:
        lines.append(f"  ... and {graph.number_of_edges() - 20} more")

    ascii_diagram = "\n".join(lines)
    logger.info(f"Graph visualized: {graph_id}")
    return ascii_diagram


async def get_graph_stats(service: Any, graph_id: str) -> Dict[str, Any]:
    """
    그래프 통계 (노드 수, 엣지 수, 밀도 등).

    Args:
        service: 서비스 인스턴스 (_graphs, _graph_metadata, _graph_builder 보유)
        graph_id: 그래프 ID

    Returns:
        Dict: 그래프 통계
    """
    graph = _get_graph(service, graph_id)

    stats = service._graph_builder.get_graph_statistics(graph)
    stats["graph_id"] = graph_id

    if graph_id in service._graph_metadata:
        stats["metadata"] = service._graph_metadata[graph_id]

    entity_type_counts: Dict[str, int] = {}
    for node_id, node_data in graph.nodes(data=True):
        entity_type = node_data.get("type", "UNKNOWN")
        entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
    stats["entity_type_counts"] = entity_type_counts

    relation_type_counts: Dict[str, int] = {}
    for source, target, edge_data in graph.edges(data=True):
        relation_type = edge_data.get("type", "UNKNOWN")
        relation_type_counts[relation_type] = relation_type_counts.get(relation_type, 0) + 1
    stats["relation_type_counts"] = relation_type_counts

    logger.info(f"Graph stats calculated: {graph_id}")
    return cast(Dict[str, Any], stats)


async def list_graphs(service: Any) -> List[Dict[str, Any]]:
    """
    모든 그래프 목록 조회.

    Args:
        service: 서비스 인스턴스 (_graphs, _graph_metadata 보유)

    Returns:
        List[Dict]: 그래프 목록 [{id, name, num_nodes, num_edges, ...}]
    """
    graphs: List[Dict[str, Any]] = []
    for graph_id, graph in service._graphs.items():
        metadata = service._graph_metadata.get(graph_id, {})
        graphs.append(
            {
                "id": graph_id,
                "name": metadata.get("name", graph_id),
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "metadata": metadata,
            }
        )
    return graphs


def delete_graph(service: Any, graph_id: str) -> None:
    """
    그래프 삭제.

    Args:
        service: 서비스 인스턴스 (_graphs, _graph_metadata 보유)
        graph_id: 그래프 ID
    """
    if graph_id in service._graphs:
        del service._graphs[graph_id]
        logger.info(f"Graph deleted: {graph_id}")

    if graph_id in service._graph_metadata:
        del service._graph_metadata[graph_id]
