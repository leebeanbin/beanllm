"""
Knowledge Graph Request DTOs - Knowledge Graph 요청 데이터 전송 객체
책임: 요청 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExtractEntitiesRequest:
    """
    엔티티 추출 요청 DTO

    책임:
    - 데이터 구조 정의만
    - 검증 없음 (Handler에서 처리)
    """

    document_id: str
    entity_types: Optional[List[str]] = None  # ["PERSON", "ORG", "LOCATION", ...]
    use_coreference: bool = True
    llm_model: str = "gpt-4o-mini"

    def __post_init__(self):
        if self.entity_types is None:
            self.entity_types = ["PERSON", "ORG", "LOCATION", "DATE", "EVENT"]


@dataclass
class ExtractRelationsRequest:
    """
    관계 추출 요청 DTO
    """

    document_id: str
    entity_pairs: Optional[List[tuple]] = None  # [(entity1, entity2), ...]
    relation_types: Optional[List[str]] = None
    bidirectional: bool = True
    llm_model: str = "gpt-4o-mini"

    def __post_init__(self):
        if self.entity_pairs is None:
            self.entity_pairs = []
        if self.relation_types is None:
            self.relation_types = []


@dataclass
class BuildGraphRequest:
    """
    그래프 구축 요청 DTO
    """

    graph_name: str
    document_ids: List[str]
    backend: str = "networkx"  # "networkx" or "neo4j"
    incremental: bool = True
    deduplicate: bool = True
    config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.config is None:
            self.config = {}


@dataclass
class QueryGraphRequest:
    """
    그래프 쿼리 요청 DTO
    """

    graph_id: str
    query: str  # Cypher-like query or natural language
    query_type: str = "cypher"  # "cypher" or "natural"
    limit: int = 10
