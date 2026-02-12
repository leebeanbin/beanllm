"""
Entity Models - 엔티티 도메인 모델

EntityType Enum과 Entity 데이터클래스를 정의.
Knowledge Graph 전체에서 공유되는 핵심 도메인 모델.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


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
        self.mentions.append(
            {
                "doc_id": doc_id,
                "start": start,
                "end": end,
                "context": context,
            }
        )

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
