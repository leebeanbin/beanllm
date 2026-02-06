"""
RAG Response Schemas
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RAGBuildResponse(BaseModel):
    """Response from building RAG index"""

    collection_name: str
    num_documents: int
    status: str = "success"


class RAGQueryResponse(BaseModel):
    """Response from RAG query"""

    query: str
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    relevance_score: float = 0.85


class CollectionInfo(BaseModel):
    """Information about a RAG collection"""

    name: str
    document_count: int
    created_at: Optional[str] = None


class CollectionListResponse(BaseModel):
    """Response for listing collections"""

    collections: List[CollectionInfo]
    total: int
