"""
Knowledge Graph Response Schemas
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EntityResponse(BaseModel):
    """Entity in the knowledge graph"""

    id: str
    name: str
    type: str = "UNKNOWN"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


class RelationResponse(BaseModel):
    """Relation between entities"""

    source: str
    target: str
    type: str = "RELATED_TO"
    label: Optional[str] = None


class BuildGraphResponse(BaseModel):
    """Response from building a knowledge graph"""

    graph_id: str
    num_nodes: int
    num_edges: int
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    relations: List[Dict[str, Any]] = Field(default_factory=list)
    statistics: Dict[str, Any] = Field(default_factory=dict)


class QueryGraphResponse(BaseModel):
    """Response from querying a knowledge graph"""

    graph_id: str
    results: List[Any]
    num_results: int
