"""
Knowledge Graph 단일 문서 처리 (엔티티 + 관계 추출).

책임:
- 단일 문서에 대한 엔티티/관계 추출 오케스트레이션
- Service의 extract_entities / extract_relations 호출 및 Domain 변환
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, cast

from beanllm.domain.knowledge_graph import (
    Entity,
    EntityType,
    Relation,
    RelationType,
)
from beanllm.dto.request.graph.kg_request import (
    ExtractEntitiesRequest,
    ExtractRelationsRequest,
)
from beanllm.service.knowledge_graph_service import IKnowledgeGraphService
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


async def process_single_document(
    service: IKnowledgeGraphService,
    doc: str,
    entity_types: Optional[List[str]],
    relation_types: Optional[List[str]],
) -> Dict[str, Any]:
    """
    단일 문서 처리 (엔티티 + 관계 추출).

    Args:
        service: Knowledge Graph 서비스 (extract_entities, extract_relations 제공)
        doc: 문서 텍스트
        entity_types: 추출할 엔티티 타입
        relation_types: 추출할 관계 타입

    Returns:
        Dict with 'entities' and 'relations' keys; on error includes 'error' key.
    """
    try:
        doc_id = str(uuid.uuid4())

        entity_types_list: List[str] = entity_types if entity_types is not None else []
        entities_response = await service.extract_entities(
            ExtractEntitiesRequest(
                document_id=doc_id,
                text=doc,
                entity_types=entity_types_list,
                resolve_coreferences=True,
            )
        )

        entities = [
            Entity(
                id=cast(str, e.get("id", "")),
                name=cast(str, e.get("name", "")),
                type=EntityType(cast(str, e.get("type", "ENTITY"))),
                description=cast(str, e.get("description", "")),
                properties=cast(Dict[str, Any], e.get("properties", {})),
                aliases=cast(List[str], e.get("aliases", [])),
                confidence=cast(float, e.get("confidence", 1.0)),
                mentions=cast(List[Dict[str, Any]], e.get("mentions", [])),
            )
            for e in entities_response.entities
        ]

        relations: List[Relation] = []
        if entities:
            relation_types_list: List[str] = relation_types if relation_types is not None else []
            relations_response = await service.extract_relations(
                ExtractRelationsRequest(
                    document_id=doc_id,
                    text=doc,
                    entities=entities_response.entities,
                    relation_types=relation_types_list,
                    infer_implicit=True,
                )
            )
            relations = [
                Relation(
                    source_id=cast(str, r["source_id"]),
                    target_id=cast(str, r["target_id"]),
                    type=RelationType(cast(str, r["type"])),
                    properties=cast(Dict[str, Any], r.get("properties", {})),
                    confidence=cast(float, r.get("confidence", 1.0)),
                    bidirectional=cast(bool, r.get("bidirectional", False)),
                )
                for r in relations_response.relations
            ]

        return {"entities": entities, "relations": relations}
    except Exception as e:
        logger.error(f"Failed to process document: {e}", exc_info=True)
        return {"entities": [], "relations": [], "error": str(e)}
