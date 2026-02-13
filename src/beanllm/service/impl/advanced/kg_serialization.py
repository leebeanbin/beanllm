"""
Knowledge Graph 직렬화 헬퍼

Entity/Relation 도메인 객체 → DTO 딕셔너리 변환 및 통계 유틸리티.
knowledge_graph_service_impl.py의 직렬화 로직을 분리하여
코드 중복 제거 및 단일 책임 원칙(SRP) 준수.
"""

from __future__ import annotations

from typing import Any, Dict, List

from beanllm.domain.knowledge_graph import Entity, Relation


def serialize_entity(entity: Entity) -> Dict[str, Any]:
    """
    Entity 도메인 객체를 DTO 딕셔너리로 변환합니다.

    Args:
        entity: Entity 도메인 객체

    Returns:
        직렬화된 엔티티 딕셔너리
    """
    return {
        "id": entity.id,
        "name": entity.name,
        "type": entity.type.value,
        "description": entity.description,
        "properties": entity.properties,
        "aliases": entity.aliases,
        "confidence": entity.confidence,
        "mentions": entity.mentions,
    }


def serialize_relation(relation: Relation) -> Dict[str, Any]:
    """
    Relation 도메인 객체를 DTO 딕셔너리로 변환합니다.

    Args:
        relation: Relation 도메인 객체

    Returns:
        직렬화된 관계 딕셔너리
    """
    return {
        "source_id": relation.source_id,
        "target_id": relation.target_id,
        "type": relation.type.value,
        "properties": relation.properties,
        "confidence": relation.confidence,
        "bidirectional": relation.bidirectional,
    }


def count_by_type(items: List[Any], type_attr: str = "type") -> Dict[str, int]:
    """
    항목들을 타입별로 카운트합니다.

    Args:
        items: Entity 또는 Relation 리스트
        type_attr: 타입 속성 이름

    Returns:
        {타입_값: 개수} 딕셔너리
    """
    counts: Dict[str, int] = {}
    for item in items:
        t = getattr(item, type_attr)
        type_value = t.value if hasattr(t, "value") else str(t)
        counts[type_value] = counts.get(type_value, 0) + 1
    return counts
