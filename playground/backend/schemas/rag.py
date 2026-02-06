"""
RAG Request Schemas
"""

from typing import List, Optional

from pydantic import BaseModel


class RAGBuildRequest(BaseModel):
    """Request to build RAG index"""

    documents: List[str]
    collection_name: Optional[str] = "default"
    model: Optional[str] = None


class RAGQueryRequest(BaseModel):
    """Request to query RAG"""

    query: str
    collection_name: Optional[str] = "default"
    top_k: int = 5
    model: Optional[str] = None


class RAGDebugRequest(BaseModel):
    """Request to debug RAG pipeline"""

    query: str
    documents: List[str]
    collection_name: Optional[str] = None
    debug_mode: str = "full"
    model: Optional[str] = None
