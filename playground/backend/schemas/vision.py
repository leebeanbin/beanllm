"""
Vision RAG Request Schemas
"""

from typing import List, Optional
from pydantic import BaseModel


class VisionRAGBuildRequest(BaseModel):
    """Request to build Vision RAG index"""
    images: List[str]  # Base64 encoded images or URLs
    texts: Optional[List[str]] = None
    collection_name: Optional[str] = "default"
    model: Optional[str] = None
    generate_captions: bool = False


class VisionRAGQueryRequest(BaseModel):
    """Request to query Vision RAG"""
    query: str
    image: Optional[str] = None  # Base64 encoded image or URL
    collection_name: Optional[str] = "default"
    top_k: int = 5
    model: Optional[str] = None
