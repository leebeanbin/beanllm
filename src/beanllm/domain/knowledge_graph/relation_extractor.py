"""
RelationExtractor - 엔티티 간 관계 추출
SOLID 원칙:
- SRP: 관계 추출만 담당
- OCP: 새로운 관계 타입 추가 가능
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class RelationType(Enum):
    """관계 타입"""

    FOUNDED = "founded"  # 설립
    WORKS_FOR = "works_for"  # 근무
    LOCATED_IN = "located_in"  # 위치
    PART_OF = "part_of"  # 부분
    RELATED_TO = "related_to"  # 관련
    CREATED = "created"  # 생성
    MANAGES = "manages"  # 관리
    OWNS = "owns"  # 소유
    MEMBER_OF = "member_of"  # 멤버
    USES = "uses"  # 사용
    DEPENDS_ON = "depends_on"  # 의존
    INFLUENCES = "influences"  # 영향
    CAUSES = "causes"  # 원인
    LOCATED_AT = "located_at"  # 위치
    OCCURS_IN = "occurs_in"  # 발생
    OTHER = "other"  # 기타


@dataclass
class Relation:
    """
    엔티티 간 관계

    Attributes:
        source_id: 소스 엔티티 ID
        target_id: 타겟 엔티티 ID
        type: 관계 타입
        description: 설명
        properties: 추가 속성
        confidence: 신뢰도 (0.0-1.0)
        bidirectional: 양방향 관계 여부
    """

    source_id: str
    target_id: str
    type: RelationType
    description: str = ""
    properties: Dict[str, Any] = None
    confidence: float = 1.0
    bidirectional: bool = False

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}

    def reverse(self) -> "Relation":
        """역방향 관계 생성"""
        return Relation(
            source_id=self.target_id,
            target_id=self.source_id,
            type=self.type,
            description=self.description,
            properties=self.properties.copy(),
            confidence=self.confidence,
            bidirectional=True,
        )


class RelationExtractor:
    """
    관계 추출기

    책임:
    - 엔티티 간 관계 추출
    - 관계 타입 분류
    - 양방향 관계 처리

    Example:
        ```python
        from beanllm.domain.knowledge_graph import EntityExtractor, RelationExtractor

        # Extract entities first
        entity_extractor = EntityExtractor()
        entities = entity_extractor.extract_entities(
            "Steve Jobs founded Apple Inc. in 1976."
        )

        # Extract relations
        relation_extractor = RelationExtractor()
        relations = relation_extractor.extract_relations(
            entities=entities,
            text="Steve Jobs founded Apple Inc. in 1976."
        )

        # Result:
        # [Relation(
        #     source_id="steve_jobs_id",
        #     target_id="apple_id",
        #     type=RelationType.FOUNDED
        # )]
        ```
    """

    def __init__(self) -> None:
        """Initialize relation extractor"""
        logger.info("RelationExtractor initialized")

    def extract_relations(
        self,
        entities: List[Any],  # List[Entity]
        text: str,
        min_confidence: float = 0.5,
    ) -> List[Relation]:
        """
        엔티티 간 관계 추출

        Args:
            entities: 엔티티 리스트
            text: 원본 텍스트
            min_confidence: 최소 신뢰도

        Returns:
            List[Relation]: 추출된 관계 리스트

        Note:
            실제 구현에서는 LLM의 structured output 사용
            또는 dependency parsing + LLM verification
        """
        logger.info(f"Extracting relations from {len(entities)} entities")

        relations = []

        # Placeholder: 실제로는 LLM API 호출
        # Example prompt:
        # "Given these entities: {entities}, extract relationships from text.
        # Return as JSON: [{"source": "...", "target": "...", "type": "founded"}]"

        # Simple pattern matching for demonstration
        if len(entities) >= 2:
            # Look for common patterns
            import re

            patterns = {
                RelationType.FOUNDED: r"(.+?) founded (.+)",
                RelationType.WORKS_FOR: r"(.+?) works? for (.+)",
                RelationType.LOCATED_IN: r"(.+?) (?:in|at) (.+)",
            }

            for relation_type, pattern in patterns.items():
                matches = re.finditer(pattern, text, re.IGNORECASE)

                for match in matches:
                    source_name = match.group(1).strip()
                    target_name = match.group(2).strip()

                    # Find matching entities
                    source_entity = self._find_entity_by_name(entities, source_name)
                    target_entity = self._find_entity_by_name(entities, target_name)

                    if source_entity and target_entity:
                        relation = Relation(
                            source_id=source_entity.id,
                            target_id=target_entity.id,
                            type=relation_type,
                            confidence=0.7,  # Placeholder
                        )
                        relations.append(relation)

        # Filter by confidence
        relations = [r for r in relations if r.confidence >= min_confidence]

        logger.info(f"Extracted {len(relations)} relations")
        return relations

    def extract_relations_with_llm(
        self,
        entities: List[Any],
        text: str,
    ) -> List[Relation]:
        """
        LLM을 사용한 관계 추출 (placeholder)

        Args:
            entities: 엔티티 리스트
            text: 텍스트

        Returns:
            List[Relation]: 추출된 관계
        """
        # Placeholder for LLM-based extraction
        # In production, call LLM with structured output
        return self.extract_relations(entities, text)

    def infer_implicit_relations(
        self,
        relations: List[Relation],
    ) -> List[Relation]:
        """
        암시적 관계 추론

        Args:
            relations: 기존 관계 리스트

        Returns:
            List[Relation]: 추론된 관계를 포함한 리스트

        Example:
            A works_for B, B part_of C -> A works_for C (transitive)
        """
        logger.info(f"Inferring implicit relations from {len(relations)} relations")

        inferred = []

        # Transitive relations
        # Example: A -> B, B -> C => A -> C
        relation_map = {}
        for rel in relations:
            if rel.source_id not in relation_map:
                relation_map[rel.source_id] = []
            relation_map[rel.source_id].append(rel)

        # Simple transitivity check
        transitive_types = {
            RelationType.PART_OF,
            RelationType.LOCATED_IN,
            RelationType.MEMBER_OF,
        }

        for rel in relations:
            if rel.type in transitive_types:
                # Check if target has further relations
                if rel.target_id in relation_map:
                    for next_rel in relation_map[rel.target_id]:
                        if next_rel.type == rel.type:
                            # Create transitive relation
                            inferred_rel = Relation(
                                source_id=rel.source_id,
                                target_id=next_rel.target_id,
                                type=rel.type,
                                description="Inferred (transitive)",
                                confidence=min(rel.confidence, next_rel.confidence) * 0.8,
                            )
                            inferred.append(inferred_rel)

        all_relations = relations + inferred
        logger.info(f"Inferred {len(inferred)} additional relations")
        return all_relations

    def create_bidirectional_relations(
        self,
        relations: List[Relation],
    ) -> List[Relation]:
        """
        양방향 관계 생성

        Args:
            relations: 관계 리스트

        Returns:
            List[Relation]: 양방향 관계 포함
        """
        bidirectional_types = {
            RelationType.RELATED_TO,
            RelationType.MEMBER_OF,
        }

        all_relations = relations.copy()

        for rel in relations:
            if rel.type in bidirectional_types and not rel.bidirectional:
                reverse_rel = rel.reverse()
                all_relations.append(reverse_rel)

        return all_relations

    def _find_entity_by_name(
        self,
        entities: List[Any],
        name: str,
    ) -> Optional[Any]:
        """이름으로 엔티티 찾기"""
        normalized_name = name.lower().strip()

        for entity in entities:
            if entity.name.lower() == normalized_name:
                return entity

            # Check aliases
            if hasattr(entity, "aliases"):
                for alias in entity.aliases:
                    if alias.lower() == normalized_name:
                        return entity

        return None

    def get_relations_by_entity(
        self,
        relations: List[Relation],
        entity_id: str,
        direction: str = "both",
    ) -> List[Relation]:
        """
        엔티티 기준 관계 필터링

        Args:
            relations: 관계 리스트
            entity_id: 엔티티 ID
            direction: "source", "target", "both"

        Returns:
            List[Relation]: 필터링된 관계
        """
        if direction == "source":
            return [r for r in relations if r.source_id == entity_id]
        elif direction == "target":
            return [r for r in relations if r.target_id == entity_id]
        else:  # both
            return [
                r
                for r in relations
                if r.source_id == entity_id or r.target_id == entity_id
            ]

    def get_relations_by_type(
        self,
        relations: List[Relation],
        relation_type: RelationType,
    ) -> List[Relation]:
        """타입별 관계 필터링"""
        return [r for r in relations if r.type == relation_type]

    def get_relation_statistics(
        self,
        relations: List[Relation],
    ) -> Dict[str, Any]:
        """관계 통계"""
        type_counts = {}
        for relation in relations:
            type_name = relation.type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        return {
            "total_relations": len(relations),
            "type_distribution": type_counts,
            "avg_confidence": sum(r.confidence for r in relations) / len(relations)
            if relations
            else 0.0,
            "bidirectional_count": sum(1 for r in relations if r.bidirectional),
        }


def extract_relations_simple(
    entities: List[Any],
    text: str,
) -> List[Relation]:
    """
    간단한 관계 추출 (편의 함수)

    Args:
        entities: 엔티티 리스트
        text: 텍스트

    Returns:
        List[Relation]: 추출된 관계
    """
    extractor = RelationExtractor()
    return extractor.extract_relations(entities, text)
