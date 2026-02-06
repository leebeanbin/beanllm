"""
Knowledge Graph Request Schemas
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class BuildGraphRequest(BaseModel):
    """Request to build a knowledge graph"""

    documents: List[str]
    graph_id: Optional[str] = None
    entity_types: Optional[List[str]] = None
    relation_types: Optional[List[str]] = None
    model: Optional[str] = None


class QueryGraphRequest(BaseModel):
    """Request to query a knowledge graph"""

    graph_id: str
    query_type: str = "cypher"
    query: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    model: Optional[str] = None


class GraphRAGRequest(BaseModel):
    """Request for GraphRAG query"""

    query: str
    graph_id: str
    model: Optional[str] = None
