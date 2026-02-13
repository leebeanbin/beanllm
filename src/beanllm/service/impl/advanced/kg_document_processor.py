"""
Knowledge Graph 단일 문서 처리 (엔티티 + 관계 추출).

책임:
- 단일 문서에 대한 엔티티/관계 추출 오케스트레이션
- Service의 extract_entities / extract_relations 호출 및 Domain 변환
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

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

logger = logging.getLogger(__name__)


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

        entities_response = await service.extract_entities(
            ExtractEntitiesRequest(
                document_id=doc_id,
                text=doc,
                entity_types=entity_types,
                resolve_coreferences=True,
            )
        )

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

        relations: List[Relation] = []
        if entities:
            relations_response = await service.extract_relations(
                ExtractRelationsRequest(
                    document_id=doc_id,
                    text=doc,
                    entities=entities_response.entities,
                    relation_types=relation_types,
                    infer_implicit=True,
                )
            )
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
