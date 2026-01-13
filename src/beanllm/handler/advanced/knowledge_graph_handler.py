"""
KnowledgeGraphHandler - Knowledge Graph 핸들러

책임:
- 입력 검증
- 에러 처리
- DTO 변환

SOLID:
- SRP: 입력 검증과 에러 처리만
- DIP: Service 인터페이스에 의존
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

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
from beanllm.service.knowledge_graph_service import IKnowledgeGraphService

logger = logging.getLogger(__name__)


class KnowledgeGraphHandler:
    """
    Knowledge Graph 핸들러

    Example:
        ```python
        handler = KnowledgeGraphHandler(service=service)

        # 엔티티 추출
        response = await handler.handle_extract_entities(
            ExtractEntitiesRequest(text="Apple was founded by Steve Jobs.")
        )

        # 그래프 구축
        response = await handler.handle_build_graph(
            BuildGraphRequest(
                documents=["Apple was founded by Steve Jobs."]
            )
        )
        ```
    """

    def __init__(self, service: IKnowledgeGraphService) -> None:
        """
        초기화

        Args:
            service: Knowledge Graph Service
        """
        self._service = service
        logger.info("KnowledgeGraphHandler initialized")

    async def handle_extract_entities(
        self, request: ExtractEntitiesRequest
    ) -> EntitiesResponse:
        """
        엔티티 추출 핸들러

        Args:
            request: 엔티티 추출 요청 DTO

        Returns:
            EntitiesResponse: 추출된 엔티티 목록

        Raises:
            ValueError: 입력 검증 실패
            RuntimeError: 서비스 실행 실패
        """
        # 입력 검증
        if not request.text or not request.text.strip():
            raise ValueError("text is required and cannot be empty")

        if request.entity_types:
            valid_types = [
                "person",
                "organization",
                "location",
                "event",
                "product",
                "concept",
                "date",
                "number",
                "other",
            ]
            for entity_type in request.entity_types:
                if entity_type.lower() not in valid_types:
                    raise ValueError(
                        f"Invalid entity type: {entity_type}. "
                        f"Valid types: {', '.join(valid_types)}"
                    )

        try:
            response = await self._service.extract_entities(request)
            logger.info(f"Entities extracted: {response.num_entities}")
            return response

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise

        except Exception as e:
            logger.error(f"Failed to extract entities: {e}", exc_info=True)
            raise RuntimeError(f"Failed to extract entities: {e}") from e

    async def handle_extract_relations(
        self, request: ExtractRelationsRequest
    ) -> RelationsResponse:
        """
        관계 추출 핸들러

        Args:
            request: 관계 추출 요청 DTO

        Returns:
            RelationsResponse: 추출된 관계 목록

        Raises:
            ValueError: 입력 검증 실패
            RuntimeError: 서비스 실행 실패
        """
        # 입력 검증
        if not request.text or not request.text.strip():
            raise ValueError("text is required and cannot be empty")

        if not request.entities:
            raise ValueError("entities are required")

        # 엔티티 검증
        for entity in request.entities:
            if not entity.get("name"):
                raise ValueError("Entity must have 'name' field")
            if not entity.get("type"):
                raise ValueError("Entity must have 'type' field")

        if request.relation_types:
            valid_types = [
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
            ]
            for relation_type in request.relation_types:
                if relation_type.lower() not in valid_types:
                    raise ValueError(
                        f"Invalid relation type: {relation_type}. "
                        f"Valid types: {', '.join(valid_types)}"
                    )

        try:
            response = await self._service.extract_relations(request)
            logger.info(f"Relations extracted: {response.num_relations}")
            return response

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise

        except Exception as e:
            logger.error(f"Failed to extract relations: {e}", exc_info=True)
            raise RuntimeError(f"Failed to extract relations: {e}") from e

    async def handle_build_graph(
        self, request: BuildGraphRequest
    ) -> BuildGraphResponse:
        """
        그래프 구축 핸들러

        Args:
            request: 그래프 구축 요청 DTO

        Returns:
            BuildGraphResponse: 그래프 정보

        Raises:
            ValueError: 입력 검증 실패
            RuntimeError: 서비스 실행 실패
        """
        # 입력 검증
        if not request.documents:
            raise ValueError("documents are required")

        for doc in request.documents:
            if not doc or not doc.strip():
                raise ValueError("Documents cannot contain empty strings")

        if request.entity_types:
            valid_types = [
                "person",
                "organization",
                "location",
                "event",
                "product",
                "concept",
                "date",
                "number",
                "other",
            ]
            for entity_type in request.entity_types:
                if entity_type.lower() not in valid_types:
                    raise ValueError(
                        f"Invalid entity type: {entity_type}. "
                        f"Valid types: {', '.join(valid_types)}"
                    )

        if request.relation_types:
            valid_types = [
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
            ]
            for relation_type in request.relation_types:
                if relation_type.lower() not in valid_types:
                    raise ValueError(
                        f"Invalid relation type: {relation_type}. "
                        f"Valid types: {', '.join(valid_types)}"
                    )

        try:
            response = await self._service.build_graph(request)
            logger.info(
                f"Graph built: {response.graph_id} "
                f"({response.num_nodes} nodes, {response.num_edges} edges)"
            )
            return response

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise

        except Exception as e:
            logger.error(f"Failed to build graph: {e}", exc_info=True)
            raise RuntimeError(f"Failed to build graph: {e}") from e

    async def handle_query_graph(
        self, request: QueryGraphRequest
    ) -> QueryGraphResponse:
        """
        그래프 쿼리 핸들러

        Args:
            request: 그래프 쿼리 요청 DTO

        Returns:
            QueryGraphResponse: 쿼리 결과

        Raises:
            ValueError: 입력 검증 실패
            RuntimeError: 서비스 실행 실패
        """
        # 입력 검증
        if not request.graph_id:
            raise ValueError("graph_id is required")

        # 쿼리 타입별 검증
        query_type = request.query_type or "cypher"
        valid_query_types = [
            "cypher",
            "find_entities_by_type",
            "find_entities_by_name",
            "find_related_entities",
            "find_shortest_path",
            "get_entity_details",
        ]

        if query_type not in valid_query_types:
            raise ValueError(
                f"Invalid query type: {query_type}. "
                f"Valid types: {', '.join(valid_query_types)}"
            )

        # 쿼리 타입별 파라미터 검증
        if query_type == "find_entities_by_type":
            if not request.params or "entity_type" not in request.params:
                raise ValueError(
                    "find_entities_by_type requires 'entity_type' parameter"
                )

        elif query_type == "find_entities_by_name":
            if not request.params or "name" not in request.params:
                raise ValueError("find_entities_by_name requires 'name' parameter")

        elif query_type == "find_related_entities":
            if not request.params or "entity_id" not in request.params:
                raise ValueError(
                    "find_related_entities requires 'entity_id' parameter"
                )

        elif query_type == "find_shortest_path":
            if not request.params or "source_id" not in request.params:
                raise ValueError(
                    "find_shortest_path requires 'source_id' parameter"
                )
            if "target_id" not in request.params:
                raise ValueError(
                    "find_shortest_path requires 'target_id' parameter"
                )

        elif query_type == "get_entity_details":
            if not request.params or "entity_id" not in request.params:
                raise ValueError(
                    "get_entity_details requires 'entity_id' parameter"
                )

        elif query_type == "cypher":
            if not request.query:
                raise ValueError("Cypher query type requires 'query' field")

        try:
            response = await self._service.query_graph(request)
            logger.info(
                f"Graph query executed: {response.graph_id} "
                f"({response.num_results} results)"
            )
            return response

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise

        except Exception as e:
            logger.error(f"Failed to query graph: {e}", exc_info=True)
            raise RuntimeError(f"Failed to query graph: {e}") from e

    async def handle_graph_rag(
        self, query: str, graph_id: str
    ) -> GraphRAGResponse:
        """
        그래프 기반 RAG 핸들러

        Args:
            query: 사용자 질의
            graph_id: 그래프 ID

        Returns:
            GraphRAGResponse: RAG 응답

        Raises:
            ValueError: 입력 검증 실패
            RuntimeError: 서비스 실행 실패
        """
        # 입력 검증
        if not query or not query.strip():
            raise ValueError("query is required and cannot be empty")

        if not graph_id:
            raise ValueError("graph_id is required")

        try:
            response = await self._service.graph_rag(
                query=query,
                graph_id=graph_id,
            )
            logger.info(
                f"Graph RAG executed: {response.graph_id} "
                f"({response.num_results} results)"
            )
            return response

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise

        except Exception as e:
            logger.error(f"Failed to execute graph RAG: {e}", exc_info=True)
            raise RuntimeError(f"Failed to execute graph RAG: {e}") from e

    async def handle_visualize_graph(self, graph_id: str) -> str:
        """
        그래프 시각화 핸들러

        Args:
            graph_id: 그래프 ID

        Returns:
            str: ASCII 그래프 다이어그램

        Raises:
            ValueError: 입력 검증 실패
            RuntimeError: 서비스 실행 실패
        """
        # 입력 검증
        if not graph_id:
            raise ValueError("graph_id is required")

        try:
            ascii_diagram = await self._service.visualize_graph(graph_id)
            logger.info(f"Graph visualized: {graph_id}")
            return ascii_diagram

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise

        except Exception as e:
            logger.error(f"Failed to visualize graph: {e}", exc_info=True)
            raise RuntimeError(f"Failed to visualize graph: {e}") from e

    async def handle_get_graph_stats(self, graph_id: str) -> Dict[str, Any]:
        """
        그래프 통계 핸들러

        Args:
            graph_id: 그래프 ID

        Returns:
            Dict: 그래프 통계

        Raises:
            ValueError: 입력 검증 실패
            RuntimeError: 서비스 실행 실패
        """
        # 입력 검증
        if not graph_id:
            raise ValueError("graph_id is required")

        try:
            stats = await self._service.get_graph_stats(graph_id)
            logger.info(f"Graph stats retrieved: {graph_id}")
            return stats

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise

        except Exception as e:
            logger.error(f"Failed to get graph stats: {e}", exc_info=True)
            raise RuntimeError(f"Failed to get graph stats: {e}") from e
