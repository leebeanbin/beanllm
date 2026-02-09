"""
Knowledge Graph Response DTOs - Knowledge Graph 응답 데이터 전송 객체
책임: 응답 데이터만 전달
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import ConfigDict

from beanllm.dto.response.base_response import BaseResponse


class EntitiesResponse(BaseResponse):
    """
    엔티티 추출 응답 DTO

    책임:
    - 응답 데이터 구조 정의만
    - 변환 로직 없음 (Service에서 처리)
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    entities: List[Dict[str, Any]]
    num_entities: int
    entity_counts_by_type: Dict[str, int]
    coreference_chains: Optional[List[List[str]]] = None
    metadata: Dict[str, Any] = {}


class RelationsResponse(BaseResponse):
    """
    관계 추출 응답 DTO
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    relations: List[Dict[str, Any]]
    num_relations: int
    relation_counts_by_type: Dict[str, int]
    confidence_scores: Optional[List[float]] = None
    metadata: Dict[str, Any] = {}


class BuildGraphResponse(BaseResponse):
    """
    그래프 구축 응답 DTO
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    graph_id: str
    graph_name: str
    num_nodes: int
    num_edges: int
    backend: str
    document_ids: List[str]
    created_at: str
    statistics: Dict[str, Any]
    density: float = 0.0
    num_connected_components: int = 1
    metadata: Dict[str, Any] = {}


class QueryGraphResponse(BaseResponse):
    """
    그래프 쿼리 응답 DTO
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    graph_id: str
    query: str
    results: List[Dict[str, Any]]
    num_results: int
    execution_time: float
    metadata: Dict[str, Any] = {}


class GraphRAGResponse(BaseResponse):
    """
    그래프 기반 RAG 응답 DTO
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: str
    entities_used: List[str]
    reasoning_paths: List[List[str]]
    graph_context: str
    graph_id: Optional[str] = None
    num_results: int = 0
    traditional_rag_context: Optional[str] = None
    hybrid_score: Optional[float] = None
    sources: Optional[List[Any]] = None
    entity_results: Optional[List[Dict[str, Any]]] = None
    path_results: Optional[List[Dict[str, Any]]] = None
    hybrid_results: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = {}
