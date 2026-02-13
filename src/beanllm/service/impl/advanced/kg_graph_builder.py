"""
Knowledge Graph 구축 로직 (build_graph).

책임:
- 문서별 엔티티/관계 추출 오케스트레이션
- 배치/순차 처리 분기
- 그래프 병합 및 Neo4j 저장
"""

from __future__ import annotations

import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional

import networkx as nx

from beanllm.domain.knowledge_graph import Entity, Neo4jAdapter, Relation
from beanllm.dto.request.graph.kg_request import BuildGraphRequest
from beanllm.dto.response.graph.kg_response import BuildGraphResponse
from beanllm.infrastructure.distributed.task_processor import BatchProcessor
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


async def build_graph_logic(
    *,
    graphs: Dict[str, nx.DiGraph],
    graph_metadata: Dict[str, Dict[str, Any]],
    graph_builder: Any,
    neo4j_adapter: Optional[Neo4jAdapter],
    batch_processor: BatchProcessor,
    process_single_doc_fn: Callable[[str], Awaitable[Dict[str, Any]]],
    request: BuildGraphRequest,
) -> BuildGraphResponse:
    """
    Knowledge Graph 구축 비즈니스 로직.

    Args:
        graphs: 그래프 저장소 (graph_id -> graph)
        graph_metadata: 그래프 메타데이터
        graph_builder: GraphBuilder 도메인 객체
        neo4j_adapter: Neo4j 어댑터 (optional)
        batch_processor: 배치 처리기
        process_single_doc_fn: 단일 문서 처리 함수 (async)
        request: 그래프 구축 요청 DTO

    Returns:
        BuildGraphResponse: 그래프 정보

    Raises:
        ValueError: documents가 비어있는 경우
        RuntimeError: 그래프 구축 실패 시
    """
    graph_id = request.graph_id or str(uuid.uuid4())

    # 기존 그래프가 있으면 가져오기
    if graph_id in graphs:
        graph = graphs[graph_id]
        logger.info(f"Loading existing graph: {graph_id}")
    else:
        graph = nx.DiGraph()
        logger.info(f"Creating new graph: {graph_id}")

    # 문서별로 엔티티/관계 추출
    all_entities: List[Entity] = []
    all_relations: List[Relation] = []

    documents = request.documents or []
    if not documents:
        raise ValueError("documents is required for graph building")

    use_batch = len(documents) >= 5

    if use_batch:
        logger.info(f"Using batch processing for {len(documents)} documents")

        results = await batch_processor.process_items(
            items=documents,
            handler=process_single_doc_fn,
        )

        for result in results:
            if isinstance(result, dict) and "error" not in result:
                all_entities.extend(result.get("entities", []))
                all_relations.extend(result.get("relations", []))
            elif isinstance(result, dict) and "error" in result:
                logger.warning(f"Document processing error: {result['error']}")
    else:
        logger.info(f"Using sequential processing for {len(documents)} documents")

        for doc in documents:
            result = await process_single_doc_fn(doc)
            all_entities.extend(result.get("entities", []))
            all_relations.extend(result.get("relations", []))

    # 그래프 구축
    graph = graph_builder.build_graph(
        entities=all_entities,
        relations=all_relations,
    )

    # 기존 그래프와 병합
    if graph_id in graphs:
        graph = graph_builder.merge_graphs(graphs[graph_id], graph)

    # 상태 저장
    graphs[graph_id] = graph
    graph_metadata[graph_id] = {
        "num_documents": len(documents),
        "entity_types": request.entity_types or [],
        "relation_types": request.relation_types or [],
    }

    # Neo4j에 저장 (optional)
    if neo4j_adapter and request.persist_to_neo4j:
        try:
            neo4j_adapter.export_graph(
                graph=graph,
                clear_existing=request.clear_existing,
            )
            logger.info(f"Graph exported to Neo4j: {graph_id}")
        except Exception as e:
            logger.warning(f"Failed to export to Neo4j: {e}")

    stats = graph_builder.get_graph_statistics(graph)

    if "num_connected_components" not in stats:
        stats["num_connected_components"] = stats.get("num_weakly_connected_components", 0)

    logger.info(f"Graph built: {graph_id} ({stats['num_nodes']} nodes, {stats['num_edges']} edges)")

    return BuildGraphResponse(
        graph_id=graph_id,
        graph_name=request.graph_name or graph_id,
        num_nodes=stats["num_nodes"],
        num_edges=stats["num_edges"],
        backend=request.backend,
        document_ids=[],
        created_at=graph_id,
        statistics=stats,
        metadata=request.config,
    )
