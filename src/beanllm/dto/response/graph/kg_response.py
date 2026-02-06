"""
Knowledge Graph Response DTOs - Knowledge Graph 응답 데이터 전송 객체
책임: 응답 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class EntitiesResponse:
    """
    엔티티 추출 응답 DTO

    책임:
    - 응답 데이터 구조 정의만
    - 변환 로직 없음 (Service에서 처리)
    """

    document_id: str
    entities: List[Dict[str, Any]]  # [{"text": "Apple", "type": "ORG", "start": 0, "end": 5}]
    num_entities: int
    entity_counts_by_type: Dict[str, int]  # {"PERSON": 10, "ORG": 5}
    coreference_chains: Optional[List[List[str]]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RelationsResponse:
    """
    관계 추출 응답 DTO
    """

    document_id: str
    relations: List[Dict[str, Any]]  # [{"source": "Apple", "target": "iPhone", "type": "PRODUCES"}]
    num_relations: int
    relation_counts_by_type: Dict[str, int]
    confidence_scores: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BuildGraphResponse:
    """
    그래프 구축 응답 DTO
    """

    graph_id: str
    graph_name: str
    num_nodes: int
    num_edges: int
    backend: str
    document_ids: List[str]
    created_at: str
    statistics: Dict[str, Any]  # {"density": 0.1, "avg_degree": 2.5}
    density: float = 0.0
    num_connected_components: int = 1
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class QueryGraphResponse:
    """
    그래프 쿼리 응답 DTO
    """

    graph_id: str
    query: str
    results: List[Dict[str, Any]]
    num_results: int
    execution_time: float
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class GraphRAGResponse:
    """
    그래프 기반 RAG 응답 DTO
    """

    answer: str
    entities_used: List[str]
    reasoning_paths: List[List[str]]  # [[entity1, relation, entity2, ...]]
    graph_context: str
    graph_id: Optional[str] = None
    num_results: int = 0
    traditional_rag_context: Optional[str] = None
    hybrid_score: Optional[float] = None
    sources: Optional[List[Any]] = None
    entity_results: Optional[List[Dict[str, Any]]] = None
    path_results: Optional[List[Dict[str, Any]]] = None
    hybrid_results: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
