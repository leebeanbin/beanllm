"""
Knowledge Graph 엔티티 추출 로직

KnowledgeGraphServiceImpl에서 extract_entities 비즈니스 로직을 분리하여
단일 책임 원칙(SRP)을 준수합니다.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from beanllm.domain.knowledge_graph import Entity, EntityType
from beanllm.dto.request.graph.kg_request import ExtractEntitiesRequest, ExtractRelationsRequest
from beanllm.dto.response.graph.kg_response import EntitiesResponse, RelationsResponse

from .kg_serialization import count_by_type, serialize_entity, serialize_relation

if TYPE_CHECKING:
    from beanllm.domain.knowledge_graph import (
        EntityExtractor,
        RelationExtractor,
        RelationType,
    )

from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


def extract_entities_logic(
    entity_extractor: "EntityExtractor",
    request: ExtractEntitiesRequest,
) -> EntitiesResponse:
    """
    엔티티 추출 비즈니스 로직

    Args:
        entity_extractor: 엔티티 추출기 도메인 객체
        request: 엔티티 추출 요청 DTO

    Returns:
        EntitiesResponse: 추출된 엔티티 목록

    Raises:
        ValueError: 텍스트가 비어있는 경우
    """
    # 엔티티 타입 필터
    entity_types = None
    if request.entity_types:
        entity_types = [EntityType(et) for et in request.entity_types]

    # 텍스트 검증
    text = request.text or ""
    if not text:
        raise ValueError("text is required for entity extraction")

    # 엔티티 추출
    entities = entity_extractor.extract_entities(
        text=text,
        entity_types=entity_types,
    )

    # Coreference resolution
    if request.resolve_coreferences:
        entities = entity_extractor.resolve_coreferences(
            entities=entities,
            text=text,
        )

    # 직렬화 + 통계
    entities_list = [serialize_entity(e) for e in entities]
    entity_counts_by_type = count_by_type(entities)

    logger.info(f"Extracted {len(entities)} entities from text")

    return EntitiesResponse(
        document_id=request.document_id,
        entities=entities_list,
        num_entities=len(entities),
        entity_counts_by_type=entity_counts_by_type,
    )


def extract_relations_logic(
    relation_extractor: "RelationExtractor",
    request: ExtractRelationsRequest,
) -> RelationsResponse:
    """
    관계 추출 비즈니스 로직

    Args:
        relation_extractor: 관계 추출기 도메인 객체
        request: 관계 추출 요청 DTO

    Returns:
        RelationsResponse: 추출된 관계 목록

    Raises:
        ValueError: 엔티티 또는 텍스트가 비어있는 경우
    """
    from beanllm.domain.knowledge_graph import RelationType

    # DTO → Domain Entity 변환
    if not request.entities:
        raise ValueError("entities is required for relation extraction")

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

    # 텍스트 검증
    text = request.text or ""
    if not text:
        raise ValueError("text is required for relation extraction")

    # 관계 타입 필터
    relation_types_list: Optional[List[RelationType]] = None
    if request.relation_types:
        relation_types_list = [RelationType(rt) for rt in request.relation_types]

    # 관계 추출
    relations = relation_extractor.extract_relations(
        entities=entities,
        text=text,
    )

    # 암시적 관계 추론
    if request.infer_implicit:
        implicit_relations = relation_extractor.infer_implicit_relations(relations=relations)
        relations.extend(implicit_relations)

    # 직렬화 + 통계
    relations_list = [serialize_relation(r) for r in relations]
    relation_counts_by_type = count_by_type(relations)

    logger.info(f"Extracted {len(relations)} relations from text")

    return RelationsResponse(
        document_id=request.document_id,
        relations=relations_list,
        num_relations=len(relations),
        relation_counts_by_type=relation_counts_by_type,
    )
