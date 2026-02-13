"""
KnowledgeGraphHandler - Knowledge Graph 핸들러

책임:
- 입력 검증
- 에러 처리 (@handle_errors 데코레이터)
- DTO 변환

SOLID:
- SRP: 입력 검증과 에러 처리만
- DIP: Service 인터페이스에 의존
"""

from __future__ import annotations

from typing import Any, Dict

from beanllm.decorators.error_handler import handle_errors
from beanllm.dto.request.graph.kg_request import (
    BuildGraphRequest,
    ExtractEntitiesRequest,
    ExtractRelationsRequest,
    QueryGraphRequest,
)
from beanllm.dto.response.graph.kg_response import (
    BuildGraphResponse,
    EntitiesResponse,
    GraphRAGResponse,
    QueryGraphResponse,
    RelationsResponse,
)
from beanllm.handler.base_handler import BaseHandler
from beanllm.service.knowledge_graph_service import IKnowledgeGraphService
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)

# 유효한 타입 상수
_VALID_ENTITY_TYPES = frozenset(
    {
        "person",
        "organization",
        "location",
        "event",
        "product",
        "concept",
        "date",
        "number",
        "other",
    }
)

_VALID_RELATION_TYPES = frozenset(
    {
        "founded",
        "works_for",
        "located_in",
        "part_of",
        "owns",
        "created",
        "produces",
        "subsidiary_of",
        "competitor",
        "partner",
        "acquired",
        "invested_in",
        "similar_to",
        "causes",
        "affects",
        "other",
    }
)

_VALID_QUERY_TYPES = frozenset(
    {
        "cypher",
        "find_entities_by_type",
        "find_entities_by_name",
        "find_related_entities",
        "find_shortest_path",
        "get_entity_details",
    }
)

# 쿼리 타입별 필수 파라미터 매핑
_QUERY_REQUIRED_PARAMS: Dict[str, list[str]] = {
    "find_entities_by_type": ["entity_type"],
    "find_entities_by_name": ["name"],
    "find_related_entities": ["entity_id"],
    "find_shortest_path": ["source_id", "target_id"],
    "get_entity_details": ["entity_id"],
}


class KnowledgeGraphHandler(BaseHandler[IKnowledgeGraphService]):
    """
    Knowledge Graph 핸들러

    Example:
        ```python
        handler = KnowledgeGraphHandler(service=service)
        response = await handler.handle_extract_entities(
            ExtractEntitiesRequest(text="Apple was founded by Steve Jobs.")
        )
        ```
    """

    def __init__(self, service: IKnowledgeGraphService) -> None:
        super().__init__(service)

    @handle_errors(error_message="Failed to extract entities")
    async def handle_extract_entities(self, request: ExtractEntitiesRequest) -> EntitiesResponse:
        """엔티티 추출 핸들러"""
        if not request.text or not request.text.strip():
            raise ValueError("text is required and cannot be empty")

        self._validate_types(request.entity_types, _VALID_ENTITY_TYPES, "entity type")

        return await self._service.extract_entities(request)

    @handle_errors(error_message="Failed to extract relations")
    async def handle_extract_relations(self, request: ExtractRelationsRequest) -> RelationsResponse:
        """관계 추출 핸들러"""
        if not request.text or not request.text.strip():
            raise ValueError("text is required and cannot be empty")

        if not request.entities:
            raise ValueError("entities are required")

        for entity in request.entities:
            if not entity.get("name"):
                raise ValueError("Entity must have 'name' field")
            if not entity.get("type"):
                raise ValueError("Entity must have 'type' field")

        self._validate_types(request.relation_types, _VALID_RELATION_TYPES, "relation type")

        return await self._service.extract_relations(request)

    @handle_errors(error_message="Failed to build graph")
    async def handle_build_graph(self, request: BuildGraphRequest) -> BuildGraphResponse:
        """그래프 구축 핸들러"""
        if not request.documents:
            raise ValueError("documents are required")

        for doc in request.documents:
            if not doc or not doc.strip():
                raise ValueError("Documents cannot contain empty strings")

        self._validate_types(request.entity_types, _VALID_ENTITY_TYPES, "entity type")
        self._validate_types(request.relation_types, _VALID_RELATION_TYPES, "relation type")

        return await self._service.build_graph(request)

    @handle_errors(error_message="Failed to query graph")
    async def handle_query_graph(self, request: QueryGraphRequest) -> QueryGraphResponse:
        """그래프 쿼리 핸들러"""
        if not request.graph_id:
            raise ValueError("graph_id is required")

        query_type = request.query_type or "cypher"
        if query_type not in _VALID_QUERY_TYPES:
            raise ValueError(
                f"Invalid query type: {query_type}. "
                f"Valid types: {', '.join(sorted(_VALID_QUERY_TYPES))}"
            )

        # 쿼리 타입별 필수 파라미터 검증 (Registry 기반)
        required_params = _QUERY_REQUIRED_PARAMS.get(query_type)
        if required_params:
            for param in required_params:
                if not request.params or param not in request.params:
                    raise ValueError(f"{query_type} requires '{param}' parameter")
        elif query_type == "cypher" and not request.query:
            raise ValueError("Cypher query type requires 'query' field")

        return await self._service.query_graph(request)

    @handle_errors(error_message="Failed to execute graph RAG")
    async def handle_graph_rag(self, query: str, graph_id: str) -> GraphRAGResponse:
        """그래프 기반 RAG 핸들러"""
        if not query or not query.strip():
            raise ValueError("query is required and cannot be empty")

        if not graph_id:
            raise ValueError("graph_id is required")

        return await self._service.graph_rag(query=query, graph_id=graph_id)

    @handle_errors(error_message="Failed to visualize graph")
    async def handle_visualize_graph(self, graph_id: str) -> str:
        """그래프 시각화 핸들러"""
        if not graph_id:
            raise ValueError("graph_id is required")

        return await self._service.visualize_graph(graph_id)

    @handle_errors(error_message="Failed to get graph stats")
    async def handle_get_graph_stats(self, graph_id: str) -> Dict[str, Any]:
        """그래프 통계 핸들러"""
        if not graph_id:
            raise ValueError("graph_id is required")

        return await self._service.get_graph_stats(graph_id)

    # ===== Private helpers =====

    @staticmethod
    def _validate_types(types: Any, valid_set: frozenset[str], label: str) -> None:
        """타입 목록 검증 헬퍼"""
        if not types:
            return
        for t in types:
            if t.lower() not in valid_set:
                raise ValueError(
                    f"Invalid {label}: {t}. Valid types: {', '.join(sorted(valid_set))}"
                )
