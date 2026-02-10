"""
Knowledge Graph Response DTOs - Knowledge Graph 응답 데이터 전송 객체
책임: 응답 데이터만 전달
"""

from __future__ import annotations

from typing import Optional

from pydantic import ConfigDict

from beanllm.dto.response.base_response import BaseResponse


class EntitiesResponse(BaseResponse):
    """
    엔티티 추출 응답 DTO

    책임:
    - 응답 데이터 구조 정의만
    - 변환 로직 없음 (Service에서 처리)
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    document_id: str
    entities: list[dict[str, object]]
    num_entities: int
    entity_counts_by_type: dict[str, int]
    coreference_chains: Optional[list[list[str]]] = None
    metadata: dict[str, object] = {}


class RelationsResponse(BaseResponse):
    """
    관계 추출 응답 DTO
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    document_id: str
    relations: list[dict[str, object]]
    num_relations: int
    relation_counts_by_type: dict[str, int]
    confidence_scores: Optional[list[float]] = None
    metadata: dict[str, object] = {}


class BuildGraphResponse(BaseResponse):
    """
    그래프 구축 응답 DTO
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    graph_id: str
    graph_name: str
    num_nodes: int
    num_edges: int
    backend: str
    document_ids: list[str]
    created_at: str
    statistics: dict[str, object]
    density: float = 0.0
    num_connected_components: int = 1
    metadata: dict[str, object] = {}


class QueryGraphResponse(BaseResponse):
    """
    그래프 쿼리 응답 DTO
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    graph_id: str
    query: str
    results: list[dict[str, object]]
    num_results: int
    execution_time: float
    metadata: dict[str, object] = {}


class GraphRAGResponse(BaseResponse):
    """
    그래프 기반 RAG 응답 DTO
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    answer: str
    entities_used: list[str]
    reasoning_paths: list[list[str]]
    graph_context: str
    graph_id: Optional[str] = None
    num_results: int = 0
    traditional_rag_context: Optional[str] = None
    hybrid_score: Optional[float] = None
    sources: Optional[list[object]] = None
    entity_results: Optional[list[dict[str, object]]] = None
    path_results: Optional[list[dict[str, object]]] = None
    hybrid_results: Optional[list[dict[str, object]]] = None
    metadata: dict[str, object] = {}
