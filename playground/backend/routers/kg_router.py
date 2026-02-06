"""
Knowledge Graph Router

Knowledge Graph endpoints for building, querying, and visualizing graphs.
Uses Python best practices with type hints and dataclasses.
"""

import logging
from typing import Any, Dict

from common import get_kg
from fastapi import APIRouter, HTTPException
from schemas.kg import BuildGraphRequest, GraphRAGRequest, QueryGraphRequest
from schemas.responses.kg import (
    BuildGraphResponse,
    QueryGraphResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/kg", tags=["Knowledge Graph"])


# ============================================================================
# Helper Functions
# ============================================================================


def _extract_entity(entity: Any, index: int) -> Dict[str, Any]:
    """Extract entity data from various formats using duck typing"""
    if not isinstance(entity, dict):
        return {
            "id": f"entity-{index}",
            "name": str(entity),
            "type": "UNKNOWN",
            "metadata": {},
        }

    # Use getattr pattern for flexibility
    entity_id = entity.get("id") or entity.get("entity_id") or f"entity-{index}"
    name = entity.get("name") or entity.get("text") or str(entity)
    entity_type = entity.get("type") or entity.get("entity_type") or "UNKNOWN"

    # Extract metadata (exclude known keys)
    known_keys = {"id", "name", "type", "text", "entity_id", "entity_type"}
    metadata = {k: v for k, v in entity.items() if k not in known_keys}

    return {
        "id": entity_id,
        "name": name,
        "type": entity_type,
        "metadata": metadata,
    }


def _extract_relation(relation: Any, index: int) -> Dict[str, Any]:
    """Extract relation data from various formats"""
    if not isinstance(relation, dict):
        return {
            "source": f"source-{index}",
            "target": f"target-{index}",
            "type": "RELATED_TO",
            "label": None,
        }

    return {
        "source": relation.get("source") or relation.get("source_id") or f"source-{index}",
        "target": relation.get("target") or relation.get("target_id") or f"target-{index}",
        "type": relation.get("type") or relation.get("relation_type") or "RELATED_TO",
        "label": relation.get("label") or relation.get("description") or relation.get("type"),
    }


# ============================================================================
# Endpoints
# ============================================================================


@router.post("/build", response_model=BuildGraphResponse)
async def kg_build(request: BuildGraphRequest) -> BuildGraphResponse:
    """
    Build knowledge graph from documents.

    Uses beanllm's KnowledgeGraph facade with optional model selection.
    """
    try:
        from beanllm.facade.advanced import KnowledgeGraph
        from beanllm.facade.core import Client

        # Create KG with requested model or use default (dependency injection pattern)
        if request.model:
            client = Client(model=request.model)
            kg = KnowledgeGraph(client=client)
        else:
            kg = get_kg()

        # Use quick_build (builder pattern)
        response = await kg.quick_build(documents=request.documents)

        # Get entities and relations for visualization
        entities_list = []
        relations_list = []

        try:
            # Query all entities using async/await pattern
            entities_response = await kg.query_graph(
                graph_id=response.graph_id,
                query_type="all_entities",
            )

            # Use list comprehension with slice for performance
            entities_list = [
                _extract_entity(entity, idx)
                for idx, entity in enumerate(entities_response.results[:50])
            ]

            # Query all relations
            relations_response = await kg.query_graph(
                graph_id=response.graph_id,
                query_type="all_relations",
            )

            relations_list = [
                _extract_relation(relation, idx)
                for idx, relation in enumerate(relations_response.results[:50])
            ]

        except Exception as e:
            logger.warning(f"Failed to get entities/relations for visualization: {e}")

        return BuildGraphResponse(
            graph_id=response.graph_id,
            num_nodes=response.num_nodes,
            num_edges=response.num_edges,
            entities=entities_list,
            relations=relations_list,
            statistics=getattr(response, "statistics", {}),
        )

    except Exception as e:
        logger.error(f"KG build error: {e}", exc_info=True)
        raise HTTPException(500, f"KG build error: {str(e)}")


@router.post("/query", response_model=QueryGraphResponse)
async def kg_query(request: QueryGraphRequest) -> QueryGraphResponse:
    """Query knowledge graph with Cypher or predefined query types."""
    try:
        from beanllm.facade.advanced import KnowledgeGraph
        from beanllm.facade.core import Client

        # Create KG with requested model or use default
        kg = KnowledgeGraph(client=Client(model=request.model)) if request.model else get_kg()

        # Query based on parameters
        if not request.query:
            # Return all entities (default behavior)
            response = await kg.query_graph(
                graph_id=request.graph_id,
                query_type="all_entities",
            )
        else:
            response = await kg.query_graph(
                graph_id=request.graph_id,
                query_type=request.query_type,
                query=request.query,
                params=request.params or {},
            )

        # Limit results using slice (more Pythonic than indexing)
        limited_results = response.results[:20]

        return QueryGraphResponse(
            graph_id=response.graph_id,
            results=limited_results,
            num_results=len(response.results),
        )

    except Exception as e:
        logger.error(f"KG query error: {e}", exc_info=True)
        raise HTTPException(500, f"KG query error: {str(e)}")


@router.post("/graph_rag")
async def kg_graph_rag(request: GraphRAGRequest) -> Dict[str, Any]:
    """Graph-based RAG query - combines knowledge graph with LLM."""
    try:
        from beanllm.facade.advanced import KnowledgeGraph
        from beanllm.facade.core import Client

        # Conditional initialization pattern
        kg = KnowledgeGraph(client=Client(model=request.model)) if request.model else get_kg()

        # Use ask method (simplified graph RAG)
        answer = await kg.ask(
            query=request.query,
            graph_id=request.graph_id,
        )

        return {
            "query": request.query,
            "graph_id": request.graph_id,
            "answer": answer,
        }

    except Exception as e:
        logger.error(f"Graph RAG error: {e}", exc_info=True)
        raise HTTPException(500, f"Graph RAG error: {str(e)}")


@router.get("/visualize/{graph_id}")
async def kg_visualize(graph_id: str) -> Dict[str, Any]:
    """Get graph visualization (ASCII art or structured data)."""
    try:
        kg = get_kg()
        visualization = await kg.visualize_graph(graph_id=graph_id)

        return {
            "graph_id": graph_id,
            "visualization": visualization,
        }

    except Exception as e:
        logger.error(f"Visualization error: {e}", exc_info=True)
        raise HTTPException(500, f"Visualization error: {str(e)}")
