"""
Graph Request DTOs - 그래프 관련 요청 DTO
"""

from .graph_request import GraphRequest
from .kg_request import ExtractEntitiesRequest, ExtractRelationsRequest

__all__ = [
    "GraphRequest",
    "ExtractEntitiesRequest",
    "ExtractRelationsRequest",
]

