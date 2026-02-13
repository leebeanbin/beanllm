"""
Entity Patterns - 엔티티 추출용 패턴 & 매핑

Regex 패턴, NER 레이블 매핑, 문자열→EntityType 매핑을 중앙 관리.
패턴 추가/수정 시 이 파일만 변경하면 됨.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from beanllm.domain.knowledge_graph.entity_models import EntityType

# NER 엔진 레이블 -> EntityType 매핑
NER_LABEL_TO_ENTITY_TYPE: Dict[str, EntityType] = {
    "PERSON": EntityType.PERSON,
    "PER": EntityType.PERSON,
    "ORG": EntityType.ORGANIZATION,
    "ORGANIZATION": EntityType.ORGANIZATION,
    "LOC": EntityType.LOCATION,
    "GPE": EntityType.LOCATION,
    "LOCATION": EntityType.LOCATION,
    "DATE": EntityType.DATE,
    "TIME": EntityType.DATE,
    "PRODUCT": EntityType.PRODUCT,
    "TECHNOLOGY": EntityType.TECHNOLOGY,
    "EVENT": EntityType.EVENT,
    "MISC": EntityType.OTHER,
}

# 문자열 -> EntityType 매핑 (LLM 응답 파싱용)
STR_TO_ENTITY_TYPE: Dict[str, EntityType] = {
    "person": EntityType.PERSON,
    "organization": EntityType.ORGANIZATION,
    "org": EntityType.ORGANIZATION,
    "company": EntityType.ORGANIZATION,
    "location": EntityType.LOCATION,
    "place": EntityType.LOCATION,
    "concept": EntityType.CONCEPT,
    "event": EntityType.EVENT,
    "date": EntityType.DATE,
    "time": EntityType.DATE,
    "product": EntityType.PRODUCT,
    "technology": EntityType.TECHNOLOGY,
    "tech": EntityType.TECHNOLOGY,
}

# Regex 기반 엔티티 추출 패턴 (fallback용)
ENTITY_REGEX_PATTERNS: Dict[EntityType, List[str]] = {
    EntityType.PERSON: [
        r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b",  # First Last
        r"\b(Dr\.|Mr\.|Mrs\.|Ms\.|Prof\.) ([A-Z][a-z]+ [A-Z][a-z]+)\b",  # Title First Last
    ],
    EntityType.ORGANIZATION: [
        r"\b([A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Co|Company|Corporation)\.?)\b",
        r"\b([A-Z][A-Z]+(?:\s[A-Z][A-Z]+)*)\b",  # Acronyms like IBM, NASA
        r"\b([A-Z][a-z]+ (?:University|Institute|Foundation|Association))\b",
    ],
    EntityType.LOCATION: [
        r"\b(New York|Los Angeles|San Francisco|London|Paris|Tokyo|Seoul|Beijing)\b",
        r"\b([A-Z][a-z]+, [A-Z]{2})\b",  # City, STATE
    ],
    EntityType.DATE: [
        r"\b(\d{4})\b",  # Year
        r"\b(\d{1,2}/\d{1,2}/\d{4})\b",  # Date
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2},? \d{4}\b",
    ],
    EntityType.TECHNOLOGY: [
        r"\b(Python|JavaScript|Java|C\+\+|Rust|Go|TypeScript|Ruby|PHP)\b",
        r"\b(AI|ML|NLP|LLM|RAG|GPT|BERT|Transformer)\b",
    ],
    EntityType.PRODUCT: [
        r"\b(iPhone|iPad|MacBook|Galaxy|Pixel|Windows|macOS|Linux|Android|iOS)\b",
    ],
}

# 사전 컴파일된 정규표현식 패턴 (O(1) 매칭, 매번 컴파일 비용 제거)
ENTITY_REGEX_PATTERNS_COMPILED: Dict["EntityType", List[re.Pattern[str]]] = {
    entity_type: [re.compile(pattern) for pattern in patterns]
    for entity_type, patterns in ENTITY_REGEX_PATTERNS.items()
}

# Coreference 대명사 매핑
PRONOUN_TO_ENTITY_TYPE: Dict[str, Optional[EntityType]] = {
    "he": EntityType.PERSON,
    "him": EntityType.PERSON,
    "his": EntityType.PERSON,
    "she": EntityType.PERSON,
    "her": EntityType.PERSON,
    "they": None,  # Any type
    "them": None,
    "it": None,
    "its": None,
}
