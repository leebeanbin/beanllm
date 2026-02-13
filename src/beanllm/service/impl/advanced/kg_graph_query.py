"""
Knowledge Graph 쿼리 로직 (query_graph).

책임:
- 쿼리 타입별 라우팅 (find_entities_by_type, find_related_entities, cypher 등)
- GraphQuerier 및 Neo4j 어댑터 호출
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import networkx as nx

from beanllm.domain.knowledge_graph import GraphQuerier, Neo4jAdapter
from beanllm.dto.request.graph.kg_request import QueryGraphRequest
from beanllm.dto.response.graph.kg_response import QueryGraphResponse
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


def execute_graph_query(
    *,
    graph: nx.DiGraph,
    neo4j_adapter: Optional[Neo4jAdapter],
    request: QueryGraphRequest,
) -> QueryGraphResponse:
    """
    그래프 쿼리 실행 (Cypher-like, 타입별 라우팅).

    Args:
        graph: NetworkX 그래프
        neo4j_adapter: Neo4j 어댑터 (Cypher 쿼리용)
        request: 쿼리 요청 DTO

    Returns:
        QueryGraphResponse: 쿼리 결과

    Raises:
        ValueError: 그래프 없음, 알 수 없는 쿼리 타입, Cypher without Neo4j
    """
    graph_id = request.graph_id
    querier = GraphQuerier(graph=graph)

    query_type = request.query_type or "cypher"
    results: List[Dict[str, Any]] = []
    params = request.params or {}

    if query_type == "find_entities_by_type":
        entity_type = str(params.get("entity_type", ""))
        results = querier.find_entities_by_type(entity_type=entity_type)

    elif query_type == "find_entities_by_name":
        name = str(params.get("name", ""))
        fuzzy = bool(params.get("fuzzy", False))
        results = querier.find_entities_by_name(name=name, fuzzy=fuzzy)

    elif query_type == "find_related_entities":
        entity_id = str(params.get("entity_id", ""))
        relation_type = (
            str(params.get("relation_type", "")) if params.get("relation_type") else None
        )
        max_hops_val = params.get("max_hops", 1)
        max_hops = int(max_hops_val) if isinstance(max_hops_val, (int, float, str)) else 1
        results = querier.find_related_entities(
            entity_id=entity_id,
            relation_type=relation_type,
            max_hops=max_hops,
        )

    elif query_type == "find_shortest_path":
        source_id = str(params.get("source_id", ""))
        target_id = str(params.get("target_id", ""))
        path = querier.find_shortest_path(
            source_id=source_id,
            target_id=target_id,
        )
        results = [{"path": path}] if path else []

    elif query_type == "get_entity_details":
        entity_id = str(params.get("entity_id", ""))
        details = querier.get_entity_details(entity_id=entity_id)
        results = [details] if details else []

    elif query_type == "cypher":
        if neo4j_adapter:
            results = neo4j_adapter.query(
                cypher_query=request.query,
                parameters=request.params,
            )
        else:
            raise ValueError("Cypher queries require Neo4j adapter")

    else:
        raise ValueError(f"Unknown query type: {query_type}")

    logger.info(f"Graph query executed: {graph_id} ({len(results)} results)")

    return QueryGraphResponse(
        graph_id=graph_id,
        query=request.query,
        results=results,
        num_results=len(results),
        execution_time=0.0,
    )
