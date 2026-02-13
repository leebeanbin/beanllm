"""
Knowledge Graph RAG 로직 (graph_rag).

책임:
- Entity-centric retrieval
- Path reasoning
- Hybrid retrieval
"""

from __future__ import annotations

from typing import Any, Dict, List

import networkx as nx

from beanllm.domain.knowledge_graph import GraphRAG
from beanllm.dto.response.graph.kg_response import GraphRAGResponse
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


def execute_graph_rag(
    *,
    graph: nx.DiGraph,
    graph_id: str,
    query: str,
    top_k: int = 5,
    max_hops: int = 2,
    max_path_length: int = 3,
) -> GraphRAGResponse:
    """
    그래프 기반 RAG 실행 (entity-centric, path reasoning, hybrid).

    Args:
        graph: NetworkX 그래프
        graph_id: 그래프 ID
        query: 사용자 질의
        top_k: 엔티티 검색 상위 k개
        max_hops: 관계 탐색 최대 홉 수
        max_path_length: 경로 추론 최대 길이

    Returns:
        GraphRAGResponse: RAG 응답
    """
    graph_rag = GraphRAG(graph=graph)

    entity_results = graph_rag.entity_centric_retrieval(
        query=query,
        top_k=top_k,
        max_hops=max_hops,
    )

    path_results = graph_rag.path_reasoning(
        query=query,
        max_path_length=max_path_length,
    )

    hybrid_results = graph_rag.hybrid_retrieval(
        query=query,
        top_k=top_k,
    )

    logger.info(f"Graph RAG executed: {graph_id} ({len(hybrid_results)} results)")

    entities_used: List[str] = [e.get("name", e.get("id", "")) for e in entity_results]
    reasoning_paths: List[List[str]] = [p.get("path", []) for p in path_results]
    graph_context = f"Graph {graph_id}: {len(entity_results)} entities, {len(path_results)} paths"

    return GraphRAGResponse(
        answer=f"Found {len(hybrid_results)} results for query: {query}",
        entities_used=entities_used,
        reasoning_paths=reasoning_paths,
        graph_context=graph_context,
        graph_id=graph_id,
        num_results=len(hybrid_results),
        entity_results=entity_results,
        path_results=path_results,
        hybrid_results=hybrid_results,
    )
