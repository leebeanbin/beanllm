"""
KnowledgeGraphServiceImpl - Knowledge Graph 서비스 구현

책임:
- 엔티티/관계 추출, 그래프 구축, 그래프 기반 RAG 비즈니스 로직
- Domain 객체 오케스트레이션
- 상태 관리 (그래프 저장)

SOLID:
- SRP: Knowledge Graph 비즈니스 로직만
- DIP: Client 인터페이스에 의존
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

import networkx as nx

from beanllm.domain.knowledge_graph import (
    Entity,
    EntityExtractor,
    EntityType,
    GraphBuilder,
    GraphQuerier,
    GraphRAG,
    Neo4jAdapter,
    Relation,
    RelationExtractor,
    RelationType,
)
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
from beanllm.infrastructure.distributed import with_distributed_features
from beanllm.infrastructure.distributed.task_processor import BatchProcessor
from beanllm.service.knowledge_graph_service import IKnowledgeGraphService

logger = logging.getLogger(__name__)


class KnowledgeGraphServiceImpl(IKnowledgeGraphService):
    """
    Knowledge Graph 서비스 구현

    Example:
        ```python
        service = KnowledgeGraphServiceImpl(client=client)

        # 엔티티 추출
        entities_response = await service.extract_entities(
            ExtractEntitiesRequest(text="Apple was founded by Steve Jobs.")
        )

        # 관계 추출
        relations_response = await service.extract_relations(
            ExtractRelationsRequest(
                text="Apple was founded by Steve Jobs.",
                entities=entities_response.entities
            )
        )

        # 그래프 구축
        graph_response = await service.build_graph(
            BuildGraphRequest(
                documents=["Apple was founded by Steve Jobs."],
                graph_id="tech_companies"
            )
        )

        # 그래프 쿼리
        query_response = await service.query_graph(
            QueryGraphRequest(
                graph_id="tech_companies",
                query="MATCH (p:PERSON)-[r:FOUNDED]->(o:ORGANIZATION) RETURN p, o"
            )
        )

        # Graph RAG
        rag_response = await service.graph_rag(
            query="Who founded Apple?",
            graph_id="tech_companies"
        )
        ```
    """

    def __init__(self, client: Optional[Any] = None) -> None:
        """
        초기화

        Args:
            client: LLM Client (optional)
        """
        # Domain 객체
        self._entity_extractor = EntityExtractor(client=client)
        self._relation_extractor = RelationExtractor(client=client)
        self._graph_builder = GraphBuilder(directed=True)

        # 상태 저장
        self._graphs: Dict[str, nx.DiGraph] = {}  # graph_id -> NetworkX Graph
        self._graph_metadata: Dict[str, Dict[str, Any]] = {}  # graph_id -> metadata

        # Neo4j Adapter (optional)
        self._neo4j_adapter: Optional[Neo4jAdapter] = None

        # BatchProcessor for parallel document processing
        self._batch_processor = BatchProcessor(
            task_type="knowledge_graph.extract",
            max_concurrent=10
        )

        logger.info("KnowledgeGraphServiceImpl initialized")

    def set_neo4j_adapter(
        self, uri: str, user: str, password: str
    ) -> None:
        """
        Neo4j Adapter 설정 (영구 저장소)

        Args:
            uri: Neo4j URI (e.g., "bolt://localhost:7687")
            user: 사용자 이름
            password: 비밀번호
        """
        try:
            self._neo4j_adapter = Neo4jAdapter(uri=uri, user=user, password=password)
            self._neo4j_adapter.connect()
            logger.info(f"Neo4j adapter connected: {uri}")
        except Exception as e:
            logger.warning(f"Failed to connect to Neo4j: {e}")
            self._neo4j_adapter = None

    @with_distributed_features(
        pipeline_type="knowledge_graph",
        enable_cache=True,
        enable_rate_limiting=True,
        enable_event_streaming=True,
        cache_key_prefix="kg:extract_entities",
        rate_limit_key="kg:llm_extraction",
        event_type="kg.extract_entities",
    )
    async def extract_entities(
        self, request: ExtractEntitiesRequest
    ) -> EntitiesResponse:
        """
        문서에서 엔티티 추출 (LLM-based NER)

        Args:
            request: 엔티티 추출 요청 DTO

        Returns:
            EntitiesResponse: 추출된 엔티티 목록
        """
        try:
            # 엔티티 타입 필터
            entity_types = None
            if request.entity_types:
                entity_types = [
                    EntityType(et) for et in request.entity_types
                ]

            # 엔티티 추출
            entities = self._entity_extractor.extract_entities(
                text=request.text,
                entity_types=entity_types,
            )

            # Coreference resolution
            if request.resolve_coreferences:
                entities = self._entity_extractor.resolve_coreferences(
                    entities=entities,
                    text=request.text,
                )

            # 직렬화
            entities_list = [
                {
                    "id": entity.id,
                    "name": entity.name,
                    "type": entity.type.value,
                    "description": entity.description,
                    "properties": entity.properties,
                    "aliases": entity.aliases,
                    "confidence": entity.confidence,
                    "mentions": entity.mentions,
                }
                for entity in entities
            ]

            logger.info(f"Extracted {len(entities)} entities from text")

            return EntitiesResponse(
                entities=entities_list,
                num_entities=len(entities),
            )

        except Exception as e:
            logger.error(f"Failed to extract entities: {e}", exc_info=True)
            raise RuntimeError(f"Failed to extract entities: {e}") from e

    @with_distributed_features(
        pipeline_type="knowledge_graph",
        enable_cache=True,
        enable_rate_limiting=True,
        enable_event_streaming=True,
        cache_key_prefix="kg:extract_relations",
        rate_limit_key="kg:llm_extraction",
        event_type="kg.extract_relations",
    )
    async def extract_relations(
        self, request: ExtractRelationsRequest
    ) -> RelationsResponse:
        """
        엔티티 간 관계 추출

        Args:
            request: 관계 추출 요청 DTO

        Returns:
            RelationsResponse: 추출된 관계 목록
        """
        try:
            # DTO → Domain Entity 변환
            entities = [
                Entity(
                    id=e.get("id", str(uuid.uuid4())),
                    name=e["name"],
                    type=EntityType(e["type"]),
                    description=e.get("description", ""),
                    properties=e.get("properties", {}),
                    aliases=e.get("aliases", []),
                    confidence=e.get("confidence", 1.0),
                    mentions=e.get("mentions", []),
                )
                for e in request.entities
            ]

            # 관계 타입 필터
            relation_types = None
            if request.relation_types:
                relation_types = [
                    RelationType(rt) for rt in request.relation_types
                ]

            # 관계 추출
            relations = self._relation_extractor.extract_relations(
                entities=entities,
                text=request.text,
                relation_types=relation_types,
            )

            # 암시적 관계 추론
            if request.infer_implicit:
                implicit_relations = self._relation_extractor.infer_implicit_relations(
                    relations=relations
                )
                relations.extend(implicit_relations)

            # 직렬화
            relations_list = [
                {
                    "source_id": relation.source_id,
                    "target_id": relation.target_id,
                    "type": relation.type.value,
                    "properties": relation.properties,
                    "confidence": relation.confidence,
                    "bidirectional": relation.bidirectional,
                }
                for relation in relations
            ]

            logger.info(f"Extracted {len(relations)} relations from text")

            return RelationsResponse(
                relations=relations_list,
                num_relations=len(relations),
            )

        except Exception as e:
            logger.error(f"Failed to extract relations: {e}", exc_info=True)
            raise RuntimeError(f"Failed to extract relations: {e}") from e

    async def _process_single_document(
        self,
        doc: str,
        entity_types: Optional[List[str]],
        relation_types: Optional[List[str]],
    ) -> Dict[str, Any]:
        """
        단일 문서 처리 (엔티티 + 관계 추출)

        Args:
            doc: 문서 텍스트
            entity_types: 추출할 엔티티 타입
            relation_types: 추출할 관계 타입

        Returns:
            Dict with 'entities' and 'relations' keys
        """
        try:
            # 엔티티 추출
            entities_response = await self.extract_entities(
                ExtractEntitiesRequest(
                    text=doc,
                    entity_types=entity_types,
                    resolve_coreferences=True,
                )
            )

            # Domain Entity로 변환
            entities = [
                Entity(
                    id=e["id"],
                    name=e["name"],
                    type=EntityType(e["type"]),
                    description=e.get("description", ""),
                    properties=e.get("properties", {}),
                    aliases=e.get("aliases", []),
                    confidence=e.get("confidence", 1.0),
                    mentions=e.get("mentions", []),
                )
                for e in entities_response.entities
            ]

            # 관계 추출
            relations = []
            if entities:
                relations_response = await self.extract_relations(
                    ExtractRelationsRequest(
                        text=doc,
                        entities=entities_response.entities,
                        relation_types=relation_types,
                        infer_implicit=True,
                    )
                )

                # Domain Relation으로 변환
                relations = [
                    Relation(
                        source_id=r["source_id"],
                        target_id=r["target_id"],
                        type=RelationType(r["type"]),
                        properties=r.get("properties", {}),
                        confidence=r.get("confidence", 1.0),
                        bidirectional=r.get("bidirectional", False),
                    )
                    for r in relations_response.relations
                ]

            return {"entities": entities, "relations": relations}
        except Exception as e:
            logger.error(f"Failed to process document: {e}", exc_info=True)
            return {"entities": [], "relations": [], "error": str(e)}

    @with_distributed_features(
        pipeline_type="knowledge_graph",
        enable_cache=True,
        enable_rate_limiting=False,  # Don't rate limit graph building (already slow)
        enable_event_streaming=True,
        cache_key_prefix="kg:build_graph",
        event_type="kg.build_graph",
    )
    async def build_graph(self, request: BuildGraphRequest) -> BuildGraphResponse:
        """
        Knowledge Graph 구축 (NetworkX/Neo4j)

        Args:
            request: 그래프 구축 요청 DTO

        Returns:
            BuildGraphResponse: 그래프 정보
        """
        try:
            graph_id = request.graph_id or str(uuid.uuid4())

            # 기존 그래프가 있으면 가져오기
            if graph_id in self._graphs:
                graph = self._graphs[graph_id]
                logger.info(f"Loading existing graph: {graph_id}")
            else:
                graph = nx.DiGraph()
                logger.info(f"Creating new graph: {graph_id}")

            # 문서별로 엔티티/관계 추출
            all_entities: List[Entity] = []
            all_relations: List[Relation] = []

            # 배치 처리 여부 결정 (5개 이상 문서면 병렬 처리)
            use_batch = len(request.documents) >= 5

            if use_batch:
                logger.info(
                    f"Using batch processing for {len(request.documents)} documents"
                )

                # 각 문서에 대한 처리 함수 생성
                async def process_doc_wrapper(doc: str) -> Dict[str, Any]:
                    return await self._process_single_document(
                        doc=doc,
                        entity_types=request.entity_types,
                        relation_types=request.relation_types,
                    )

                # 병렬 처리
                results = await self._batch_processor.process_batch(
                    items=request.documents,
                    handler=process_doc_wrapper,
                    max_concurrent=10,
                )

                # 결과 수집
                for result in results:
                    if isinstance(result, dict) and "error" not in result:
                        all_entities.extend(result.get("entities", []))
                        all_relations.extend(result.get("relations", []))
                    elif isinstance(result, dict) and "error" in result:
                        logger.warning(f"Document processing error: {result['error']}")
            else:
                logger.info(
                    f"Using sequential processing for {len(request.documents)} documents"
                )

                # 순차 처리 (문서가 적을 때)
                for doc in request.documents:
                    result = await self._process_single_document(
                        doc=doc,
                        entity_types=request.entity_types,
                        relation_types=request.relation_types,
                    )
                    all_entities.extend(result.get("entities", []))
                    all_relations.extend(result.get("relations", []))

            # 그래프 구축
            graph = self._graph_builder.build_graph(
                entities=all_entities,
                relations=all_relations,
            )

            # 기존 그래프와 병합
            if graph_id in self._graphs:
                graph = self._graph_builder.merge_graphs(
                    [self._graphs[graph_id], graph]
                )

            # 상태 저장
            self._graphs[graph_id] = graph
            self._graph_metadata[graph_id] = {
                "num_documents": len(request.documents),
                "entity_types": request.entity_types or [],
                "relation_types": request.relation_types or [],
            }

            # Neo4j에 저장 (optional)
            if self._neo4j_adapter and request.persist_to_neo4j:
                try:
                    self._neo4j_adapter.export_graph(
                        graph=graph,
                        clear_existing=request.clear_existing,
                    )
                    logger.info(f"Graph exported to Neo4j: {graph_id}")
                except Exception as e:
                    logger.warning(f"Failed to export to Neo4j: {e}")

            # 통계 계산
            stats = self._graph_builder.get_graph_statistics(graph)

            logger.info(
                f"Graph built: {graph_id} ({stats['num_nodes']} nodes, {stats['num_edges']} edges)"
            )

            return BuildGraphResponse(
                graph_id=graph_id,
                num_nodes=stats["num_nodes"],
                num_edges=stats["num_edges"],
                density=stats["density"],
                num_connected_components=stats["num_connected_components"],
            )

        except Exception as e:
            logger.error(f"Failed to build graph: {e}", exc_info=True)
            raise RuntimeError(f"Failed to build graph: {e}") from e

    @with_distributed_features(
        pipeline_type="knowledge_graph",
        enable_cache=True,
        enable_rate_limiting=True,
        enable_event_streaming=True,
        cache_key_prefix="kg:query_graph",
        rate_limit_key="kg:query",
        event_type="kg.query_graph",
    )
    async def query_graph(self, request: QueryGraphRequest) -> QueryGraphResponse:
        """
        그래프 쿼리 (Cypher-like)

        Args:
            request: 그래프 쿼리 요청 DTO

        Returns:
            QueryGraphResponse: 쿼리 결과
        """
        try:
            graph_id = request.graph_id

            # 그래프 확인
            if graph_id not in self._graphs:
                raise ValueError(f"Graph not found: {graph_id}")

            graph = self._graphs[graph_id]
            querier = GraphQuerier(graph=graph)

            # 쿼리 타입별 처리
            query_type = request.query_type or "cypher"
            results: List[Dict[str, Any]] = []

            if query_type == "find_entities_by_type":
                # 타입별 엔티티 검색
                entity_type = request.params.get("entity_type")
                results = querier.find_entities_by_type(entity_type=entity_type)

            elif query_type == "find_entities_by_name":
                # 이름 기반 검색
                name = request.params.get("name")
                fuzzy = request.params.get("fuzzy", False)
                results = querier.find_entities_by_name(name=name, fuzzy=fuzzy)

            elif query_type == "find_related_entities":
                # 관계 기반 탐색
                entity_id = request.params.get("entity_id")
                relation_type = request.params.get("relation_type")
                max_hops = request.params.get("max_hops", 1)
                results = querier.find_related_entities(
                    entity_id=entity_id,
                    relation_type=relation_type,
                    max_hops=max_hops,
                )

            elif query_type == "find_shortest_path":
                # 최단 경로
                source_id = request.params.get("source_id")
                target_id = request.params.get("target_id")
                path = querier.find_shortest_path(
                    source_id=source_id,
                    target_id=target_id,
                )
                results = [{"path": path}] if path else []

            elif query_type == "get_entity_details":
                # 엔티티 상세 정보
                entity_id = request.params.get("entity_id")
                details = querier.get_entity_details(entity_id=entity_id)
                results = [details] if details else []

            elif query_type == "cypher":
                # Cypher 쿼리 (Neo4j만 지원)
                if self._neo4j_adapter:
                    results = self._neo4j_adapter.query(
                        cypher_query=request.query,
                        parameters=request.params,
                    )
                else:
                    raise ValueError("Cypher queries require Neo4j adapter")

            else:
                raise ValueError(f"Unknown query type: {query_type}")

            logger.info(
                f"Graph query executed: {graph_id} ({len(results)} results)"
            )

            return QueryGraphResponse(
                graph_id=graph_id,
                results=results,
                num_results=len(results),
            )

        except Exception as e:
            logger.error(f"Failed to query graph: {e}", exc_info=True)
            raise RuntimeError(f"Failed to query graph: {e}") from e

    @with_distributed_features(
        pipeline_type="knowledge_graph",
        enable_cache=True,
        enable_rate_limiting=True,
        enable_event_streaming=True,
        cache_key_prefix="kg:graph_rag",
        rate_limit_key="kg:rag",
        event_type="kg.graph_rag",
    )
    async def graph_rag(self, query: str, graph_id: str) -> GraphRAGResponse:
        """
        그래프 기반 RAG (entity-centric retrieval, path reasoning)

        Args:
            query: 사용자 질의
            graph_id: 그래프 ID

        Returns:
            GraphRAGResponse: RAG 응답
        """
        try:
            # 그래프 확인
            if graph_id not in self._graphs:
                raise ValueError(f"Graph not found: {graph_id}")

            graph = self._graphs[graph_id]
            graph_rag = GraphRAG(graph=graph)

            # Entity-centric retrieval
            entity_results = graph_rag.entity_centric_retrieval(
                query=query,
                top_k=5,
                max_hops=2,
            )

            # Path reasoning
            path_results = graph_rag.path_reasoning(
                query=query,
                max_path_length=3,
            )

            # Hybrid retrieval
            hybrid_results = graph_rag.hybrid_retrieval(
                query=query,
                top_k=5,
            )

            logger.info(
                f"Graph RAG executed: {graph_id} ({len(hybrid_results)} results)"
            )

            return GraphRAGResponse(
                query=query,
                graph_id=graph_id,
                entity_results=entity_results,
                path_results=path_results,
                hybrid_results=hybrid_results,
                num_results=len(hybrid_results),
            )

        except Exception as e:
            logger.error(f"Failed to execute graph RAG: {e}", exc_info=True)
            raise RuntimeError(f"Failed to execute graph RAG: {e}") from e

    async def visualize_graph(self, graph_id: str) -> str:
        """
        그래프 시각화 (ASCII)

        Args:
            graph_id: 그래프 ID

        Returns:
            str: ASCII 그래프 다이어그램
        """
        try:
            # 그래프 확인
            if graph_id not in self._graphs:
                raise ValueError(f"Graph not found: {graph_id}")

            graph = self._graphs[graph_id]

            # ASCII 시각화 (간단한 구현)
            lines = []
            lines.append(f"Graph: {graph_id}")
            lines.append("=" * 50)
            lines.append(f"Nodes: {graph.number_of_nodes()}")
            lines.append(f"Edges: {graph.number_of_edges()}")
            lines.append("")
            lines.append("Entities:")
            lines.append("-" * 50)

            # 노드 (최대 20개)
            for i, (node_id, node_data) in enumerate(
                list(graph.nodes(data=True))[:20]
            ):
                name = node_data.get("name", node_id)
                entity_type = node_data.get("type", "UNKNOWN")
                lines.append(f"  [{entity_type}] {name} (id: {node_id})")

            if graph.number_of_nodes() > 20:
                lines.append(f"  ... and {graph.number_of_nodes() - 20} more")

            lines.append("")
            lines.append("Relations:")
            lines.append("-" * 50)

            # 엣지 (최대 20개)
            for i, (source, target, edge_data) in enumerate(
                list(graph.edges(data=True))[:20]
            ):
                source_name = graph.nodes[source].get("name", source)
                target_name = graph.nodes[target].get("name", target)
                relation_type = edge_data.get("type", "UNKNOWN")
                lines.append(f"  {source_name} --[{relation_type}]--> {target_name}")

            if graph.number_of_edges() > 20:
                lines.append(f"  ... and {graph.number_of_edges() - 20} more")

            ascii_diagram = "\n".join(lines)

            logger.info(f"Graph visualized: {graph_id}")

            return ascii_diagram

        except Exception as e:
            logger.error(f"Failed to visualize graph: {e}", exc_info=True)
            raise RuntimeError(f"Failed to visualize graph: {e}") from e

    async def get_graph_stats(self, graph_id: str) -> Dict[str, Any]:
        """
        그래프 통계 (노드 수, 엣지 수, 밀도 등)

        Args:
            graph_id: 그래프 ID

        Returns:
            Dict: 그래프 통계
        """
        try:
            # 그래프 확인
            if graph_id not in self._graphs:
                raise ValueError(f"Graph not found: {graph_id}")

            graph = self._graphs[graph_id]

            # 통계 계산
            stats = self._graph_builder.get_graph_statistics(graph)

            # 추가 통계
            stats["graph_id"] = graph_id

            # 메타데이터
            if graph_id in self._graph_metadata:
                stats["metadata"] = self._graph_metadata[graph_id]

            # 엔티티 타입별 개수
            entity_type_counts: Dict[str, int] = {}
            for node_id, node_data in graph.nodes(data=True):
                entity_type = node_data.get("type", "UNKNOWN")
                entity_type_counts[entity_type] = (
                    entity_type_counts.get(entity_type, 0) + 1
                )
            stats["entity_type_counts"] = entity_type_counts

            # 관계 타입별 개수
            relation_type_counts: Dict[str, int] = {}
            for source, target, edge_data in graph.edges(data=True):
                relation_type = edge_data.get("type", "UNKNOWN")
                relation_type_counts[relation_type] = (
                    relation_type_counts.get(relation_type, 0) + 1
                )
            stats["relation_type_counts"] = relation_type_counts

            logger.info(f"Graph stats calculated: {graph_id}")

            return stats

        except Exception as e:
            logger.error(f"Failed to get graph stats: {e}", exc_info=True)
            raise RuntimeError(f"Failed to get graph stats: {e}") from e

    # --- Helper Methods ---

    def _get_graph(self, graph_id: str) -> nx.DiGraph:
        """
        그래프 가져오기

        Args:
            graph_id: 그래프 ID

        Returns:
            nx.DiGraph: NetworkX 그래프

        Raises:
            ValueError: 그래프가 없는 경우
        """
        if graph_id not in self._graphs:
            raise ValueError(f"Graph not found: {graph_id}")
        return self._graphs[graph_id]

    def list_graphs(self) -> List[str]:
        """
        모든 그래프 ID 목록

        Returns:
            List[str]: 그래프 ID 목록
        """
        return list(self._graphs.keys())

    def delete_graph(self, graph_id: str) -> None:
        """
        그래프 삭제

        Args:
            graph_id: 그래프 ID
        """
        if graph_id in self._graphs:
            del self._graphs[graph_id]
            logger.info(f"Graph deleted: {graph_id}")

        if graph_id in self._graph_metadata:
            del self._graph_metadata[graph_id]
