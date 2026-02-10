"""
Knowledge Graph Request DTOs - Knowledge Graph 요청 데이터 전송 객체
책임: 요청 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True, kw_only=True)
class ExtractEntitiesRequest:
    """
    엔티티 추출 요청 DTO

    책임:
    - 데이터 구조 정의만
    - 검증 없음 (Handler에서 처리)
    """

    document_id: str
    text: Optional[str] = None
    entity_types: list[str] = field(
        default_factory=lambda: ["PERSON", "ORG", "LOCATION", "DATE", "EVENT"]
    )
    use_coreference: bool = True
    resolve_coreferences: bool = True
    llm_model: str = "gpt-4o-mini"

    def __post_init__(self) -> None:
        if not self.resolve_coreferences:
            self.use_coreference = False


@dataclass(slots=True, kw_only=True)
class ExtractRelationsRequest:
    """
    관계 추출 요청 DTO
    """

    document_id: str
    text: Optional[str] = None
    entities: list[dict[str, object]] = field(default_factory=list)
    entity_pairs: list[tuple[str, ...]] = field(default_factory=list)
    relation_types: list[str] = field(default_factory=list)
    bidirectional: bool = True
    infer_implicit: bool = False
    llm_model: str = "gpt-4o-mini"


@dataclass(slots=True, kw_only=True)
class BuildGraphRequest:
    """
    그래프 구축 요청 DTO
    """

    graph_name: str = ""
    graph_id: Optional[str] = None
    documents: list[str] = field(default_factory=list)
    document_ids: list[str] = field(default_factory=list)
    entity_types: Optional[list[str]] = None
    relation_types: Optional[list[str]] = None
    backend: str = "networkx"
    persist_to_neo4j: bool = False
    clear_existing: bool = False
    incremental: bool = True
    deduplicate: bool = True
    config: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, kw_only=True)
class QueryGraphRequest:
    """
    그래프 쿼리 요청 DTO
    """

    graph_id: str
    query: str
    query_type: str = "cypher"
    limit: int = 10
    params: dict[str, object] = field(default_factory=dict)
