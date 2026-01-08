"""
EntityExtractor - LLM-based Named Entity Recognition
SOLID 원칙:
- SRP: 엔티티 추출만 담당
- OCP: 새로운 엔티티 타입 추가 가능
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from beanllm.utils.logger import get_logger

logger = get_logger(__name__)


class EntityType(Enum):
    """엔티티 타입"""

    PERSON = "person"  # 인물
    ORGANIZATION = "organization"  # 조직
    LOCATION = "location"  # 장소
    CONCEPT = "concept"  # 개념
    EVENT = "event"  # 이벤트
    DATE = "date"  # 날짜
    PRODUCT = "product"  # 제품
    TECHNOLOGY = "technology"  # 기술
    OTHER = "other"  # 기타


@dataclass
class Entity:
    """
    엔티티

    Attributes:
        id: 엔티티 ID
        name: 엔티티 이름
        type: 엔티티 타입
        description: 설명
        aliases: 별칭 리스트
        properties: 추가 속성
        confidence: 신뢰도 (0.0-1.0)
        mentions: 언급된 위치 [{doc_id, start, end}, ...]
    """

    id: str
    name: str
    type: EntityType
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    mentions: List[Dict[str, Any]] = field(default_factory=list)

    def add_alias(self, alias: str) -> None:
        """별칭 추가"""
        if alias not in self.aliases and alias != self.name:
            self.aliases.append(alias)

    def add_mention(self, doc_id: str, start: int, end: int, context: str = "") -> None:
        """언급 위치 추가"""
        self.mentions.append({
            "doc_id": doc_id,
            "start": start,
            "end": end,
            "context": context,
        })

    def merge_with(self, other: "Entity") -> None:
        """다른 엔티티와 병합 (coreference resolution)"""
        # Merge aliases
        for alias in other.aliases:
            self.add_alias(alias)

        # Merge properties
        self.properties.update(other.properties)

        # Merge mentions
        self.mentions.extend(other.mentions)

        # Update confidence (average)
        self.confidence = (self.confidence + other.confidence) / 2

        # Update description (keep longer one)
        if len(other.description) > len(self.description):
            self.description = other.description


class EntityExtractor:
    """
    LLM 기반 엔티티 추출기

    책임:
    - 문서에서 엔티티 추출
    - Coreference resolution (대명사 해결)
    - 엔티티 중복 제거 및 병합

    Example:
        ```python
        extractor = EntityExtractor()

        # Extract entities from text
        entities = extractor.extract_entities(
            text="Steve Jobs founded Apple Inc. in 1976.",
            entity_types=[EntityType.PERSON, EntityType.ORGANIZATION]
        )

        # Result:
        # [
        #   Entity(id="...", name="Steve Jobs", type=EntityType.PERSON),
        #   Entity(id="...", name="Apple Inc.", type=EntityType.ORGANIZATION)
        # ]

        # Resolve coreferences
        entities = extractor.resolve_coreferences(entities, text)
        ```
    """

    def __init__(self) -> None:
        """Initialize entity extractor"""
        self._entity_cache: Dict[str, Entity] = {}
        logger.info("EntityExtractor initialized")

    def extract_entities(
        self,
        text: str,
        entity_types: Optional[List[EntityType]] = None,
        min_confidence: float = 0.5,
    ) -> List[Entity]:
        """
        텍스트에서 엔티티 추출

        Args:
            text: 입력 텍스트
            entity_types: 추출할 엔티티 타입 (None이면 모두)
            min_confidence: 최소 신뢰도

        Returns:
            List[Entity]: 추출된 엔티티 리스트

        Note:
            실제 구현에서는 LLM의 structured output을 사용하여 엔티티 추출
            현재는 placeholder 로직
        """
        logger.info(f"Extracting entities from text ({len(text)} chars)")

        # Placeholder: 실제로는 LLM API 호출
        # Example prompt:
        # "Extract entities from the following text. Return as JSON:
        # [{"name": "...", "type": "person", "description": "..."}]"

        entities = []

        # Placeholder logic (실제로는 LLM 호출)
        # 여기서는 간단한 패턴 매칭으로 시뮬레이션
        import re

        # Simple pattern matching for demonstration
        patterns = {
            EntityType.PERSON: r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b",
            EntityType.ORGANIZATION: r"\b([A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd)\.?)\b",
            EntityType.DATE: r"\b(\d{4})\b",
        }

        entity_types_to_extract = entity_types or list(EntityType)

        for entity_type in entity_types_to_extract:
            if entity_type not in patterns:
                continue

            pattern = patterns[entity_type]
            matches = re.finditer(pattern, text)

            for match in matches:
                name = match.group(1)
                entity_id = self._generate_entity_id(name, entity_type)

                entity = Entity(
                    id=entity_id,
                    name=name,
                    type=entity_type,
                    confidence=0.8,  # Placeholder confidence
                )

                entities.append(entity)

        # Deduplicate
        entities = self._deduplicate_entities(entities)

        # Filter by confidence
        entities = [e for e in entities if e.confidence >= min_confidence]

        logger.info(f"Extracted {len(entities)} entities")
        return entities

    def extract_entities_from_documents(
        self,
        documents: List[Dict[str, Any]],
        entity_types: Optional[List[EntityType]] = None,
    ) -> List[Entity]:
        """
        여러 문서에서 엔티티 추출

        Args:
            documents: 문서 리스트 [{"id": "...", "content": "..."}, ...]
            entity_types: 추출할 엔티티 타입

        Returns:
            List[Entity]: 추출된 엔티티 리스트 (중복 제거됨)
        """
        logger.info(f"Extracting entities from {len(documents)} documents")

        all_entities = []

        for doc in documents:
            doc_id = doc.get("id", "unknown")
            content = doc.get("content", "")

            entities = self.extract_entities(content, entity_types)

            # Add document reference to entities
            for entity in entities:
                entity.add_mention(doc_id, 0, len(content), content[:100])

            all_entities.extend(entities)

        # Global deduplication
        all_entities = self._deduplicate_entities(all_entities)

        logger.info(f"Total {len(all_entities)} unique entities extracted")
        return all_entities

    def resolve_coreferences(
        self,
        entities: List[Entity],
        text: str,
    ) -> List[Entity]:
        """
        Coreference resolution (대명사 해결)

        Args:
            entities: 엔티티 리스트
            text: 원본 텍스트

        Returns:
            List[Entity]: Coreference가 해결된 엔티티 리스트

        Note:
            실제 구현에서는 spaCy, neuralcoref 등 사용 또는 LLM 활용
        """
        logger.info(f"Resolving coreferences for {len(entities)} entities")

        # Placeholder: 실제로는 coreference resolution 모델 사용
        # 예: "Steve Jobs founded Apple. He was a visionary."
        #     -> "He"를 "Steve Jobs"로 해결

        # Simple heuristic: merge entities with similar names
        resolved_entities = []
        entity_map: Dict[str, Entity] = {}

        for entity in entities:
            # Find similar entity
            canonical_name = self._canonicalize_name(entity.name)

            if canonical_name in entity_map:
                # Merge with existing entity
                entity_map[canonical_name].merge_with(entity)
            else:
                entity_map[canonical_name] = entity

        resolved_entities = list(entity_map.values())

        logger.info(f"Resolved to {len(resolved_entities)} entities")
        return resolved_entities

    def extract_entity_properties(
        self,
        entity: Entity,
        text: str,
    ) -> Dict[str, Any]:
        """
        엔티티 속성 추출

        Args:
            entity: 엔티티
            text: 텍스트

        Returns:
            Dict[str, Any]: 속성 딕셔너리

        Note:
            실제 구현에서는 LLM으로 structured output 추출
        """
        logger.info(f"Extracting properties for entity: {entity.name}")

        # Placeholder: LLM으로 속성 추출
        # Example: "Steve Jobs was the CEO of Apple from 1997 to 2011"
        #          -> {"role": "CEO", "company": "Apple", "tenure": "1997-2011"}

        properties = {}

        # Simple pattern matching (placeholder)
        if entity.type == EntityType.PERSON:
            properties["type"] = "person"

        elif entity.type == EntityType.ORGANIZATION:
            properties["type"] = "organization"

        entity.properties.update(properties)
        return properties

    def _generate_entity_id(self, name: str, entity_type: EntityType) -> str:
        """엔티티 ID 생성"""
        import hashlib

        canonical_name = self._canonicalize_name(name)
        id_string = f"{entity_type.value}:{canonical_name}"
        entity_id = hashlib.md5(id_string.encode()).hexdigest()[:16]
        return entity_id

    def _canonicalize_name(self, name: str) -> str:
        """이름 정규화 (소문자, 공백 제거)"""
        return name.lower().strip().replace("  ", " ")

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """엔티티 중복 제거"""
        entity_map: Dict[str, Entity] = {}

        for entity in entities:
            canonical_name = self._canonicalize_name(entity.name)

            if canonical_name in entity_map:
                # Merge with existing
                entity_map[canonical_name].merge_with(entity)
            else:
                entity_map[canonical_name] = entity

        return list(entity_map.values())

    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """ID로 엔티티 조회"""
        return self._entity_cache.get(entity_id)

    def get_entities_by_type(
        self,
        entities: List[Entity],
        entity_type: EntityType,
    ) -> List[Entity]:
        """타입별 엔티티 필터링"""
        return [e for e in entities if e.type == entity_type]

    def get_entity_statistics(
        self,
        entities: List[Entity],
    ) -> Dict[str, Any]:
        """엔티티 통계"""
        type_counts = {}
        for entity in entities:
            type_name = entity.type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        return {
            "total_entities": len(entities),
            "type_distribution": type_counts,
            "avg_confidence": sum(e.confidence for e in entities) / len(entities)
            if entities
            else 0.0,
            "total_mentions": sum(len(e.mentions) for e in entities),
        }


def extract_entities_simple(
    text: str,
    entity_types: Optional[List[EntityType]] = None,
) -> List[Entity]:
    """
    간단한 엔티티 추출 (편의 함수)

    Args:
        text: 입력 텍스트
        entity_types: 추출할 엔티티 타입

    Returns:
        List[Entity]: 추출된 엔티티
    """
    extractor = EntityExtractor()
    return extractor.extract_entities(text, entity_types)
