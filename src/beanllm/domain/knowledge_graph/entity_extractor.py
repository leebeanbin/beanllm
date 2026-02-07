"""
EntityExtractor - LLM-based Named Entity Recognition

LLM을 활용한 고급 엔티티 추출기.
Structured output을 통해 정확한 Named Entity Recognition을 수행합니다.

SOLID 원칙:
- SRP: 엔티티 추출만 담당
- OCP: 새로운 엔티티 타입 추가 가능
- DIP: LLM 인터페이스를 통해 다양한 프로바이더 지원

Features:
- LLM 기반 Named Entity Recognition
- Coreference Resolution (대명사 해결)
- Entity Property Extraction (속성 추출)
- Entity Deduplication & Merging
- Batch Processing for Multiple Documents

Example:
    ```python
    from beanllm.domain.knowledge_graph import EntityExtractor, EntityType

    # LLM 함수와 함께 초기화
    extractor = EntityExtractor(llm_function=my_llm_call)

    # 엔티티 추출
    entities = extractor.extract_entities(
        text="Steve Jobs founded Apple Inc. in 1976. He was a visionary.",
        entity_types=[EntityType.PERSON, EntityType.ORGANIZATION]
    )

    # Coreference resolution
    resolved = extractor.resolve_coreferences(entities, text)
    # "He" -> "Steve Jobs" 해결됨
    ```
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


# LLM Prompts for Entity Extraction
ENTITY_EXTRACTION_PROMPT = """Extract named entities from the following text.

Text:
{text}

Entity types to extract: {entity_types}

Return a JSON array with the following structure:
[
  {{
    "name": "Entity name",
    "type": "person|organization|location|concept|event|date|product|technology|other",
    "description": "Brief description of the entity",
    "aliases": ["alternative names"],
    "confidence": 0.95
  }}
]

Rules:
1. Only extract entities that clearly appear in the text
2. Assign appropriate entity types based on context
3. Include common aliases if applicable
4. Set confidence based on how certain you are (0.0-1.0)
5. Return ONLY valid JSON, no explanation

JSON:"""

COREFERENCE_RESOLUTION_PROMPT = """Resolve coreferences in the following text.

Text:
{text}

Known entities:
{entities}

Identify pronouns (he, she, it, they, etc.) and other references that point to the known entities.

Return a JSON array mapping references to entity names:
[
  {{
    "reference": "He",
    "resolved_to": "Steve Jobs",
    "start_position": 45,
    "confidence": 0.95
  }}
]

Return ONLY valid JSON:"""

PROPERTY_EXTRACTION_PROMPT = """Extract properties for the entity "{entity_name}" from the following text.

Text:
{text}

Entity Type: {entity_type}

Extract relevant properties based on the entity type:
- For PERSON: role, title, affiliation, birth_date, nationality, etc.
- For ORGANIZATION: industry, founded, headquarters, ceo, employees, etc.
- For LOCATION: country, population, coordinates, etc.
- For PRODUCT: manufacturer, release_date, price, category, etc.
- For TECHNOLOGY: developer, version, release_date, etc.

Return a JSON object with extracted properties:
{{
  "property_name": "property_value",
  ...
}}

Return ONLY valid JSON:"""


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


class EntityExtractor:
    """
    멀티 엔진 엔티티 추출기

    다양한 NER 엔진(spaCy, HuggingFace, GLiNER, Flair, LLM)을 지원하며,
    Coreference Resolution, Property Extraction 등 고급 기능을 제공합니다.

    지원 엔진:
    - spacy: 빠르고 정확한 통계 기반 NER
    - huggingface: BERT/RoBERTa 기반 transformer NER
    - gliner: Zero-shot NER (커스텀 엔티티 타입)
    - flair: Contextual embeddings 기반 NER
    - llm: GPT/Claude 등 LLM 기반 NER

    책임:
    - 문서에서 엔티티 추출
    - Coreference resolution (대명사 해결)
    - 엔티티 속성 추출
    - 엔티티 중복 제거 및 병합

    Example:
        ```python
        # spaCy 엔진 사용 (빠름)
        extractor = EntityExtractor(engine="spacy")

        # HuggingFace 엔진 사용 (정확함)
        extractor = EntityExtractor(
            engine="huggingface",
            model="dslim/bert-base-NER"
        )

        # GLiNER 사용 (커스텀 엔티티 타입)
        extractor = EntityExtractor(
            engine="gliner",
            labels=["person", "company", "technology", "product"]
        )

        # LLM 사용 (가장 유연)
        extractor = EntityExtractor(llm_function=my_llm_call)

        # 엔티티 추출
        entities = extractor.extract_entities(
            text="Steve Jobs founded Apple Inc. in 1976.",
        )

        # Coreference resolution
        entities = extractor.resolve_coreferences(entities, text)

        # 벤치마크
        from beanllm.domain.knowledge_graph.ner_engines import NERBenchmark
        benchmark = NERBenchmark(engines=[...])
        results = benchmark.run(test_data)
        ```
    """

    def __init__(
        self,
        engine: Optional[str] = None,
        model: Optional[str] = None,
        llm_function: Optional[Callable[[str], str]] = None,
        labels: Optional[List[str]] = None,
        use_fallback: bool = True,
        max_text_length: int = 8000,
        batch_size: int = 5,
    ) -> None:
        """
        Initialize entity extractor

        Args:
            engine: NER 엔진 타입 ("spacy", "huggingface", "gliner", "flair")
                    None이면 llm_function 또는 regex 사용
            model: 엔진별 모델 이름
            llm_function: LLM 호출 함수 (prompt -> response)
            labels: GLiNER용 커스텀 레이블
            use_fallback: 실패 시 regex fallback 사용 여부
            max_text_length: 최대 텍스트 길이 (초과 시 청킹)
            batch_size: 배치 처리 시 문서 수
        """
        self._llm_function = llm_function
        self._use_fallback = use_fallback
        self._max_text_length = max_text_length
        self._batch_size = batch_size
        self._entity_cache: Dict[str, Entity] = {}
        self._ner_engine = None
        self._engine_type = engine

        # NER 엔진 초기화
        if engine:
            try:
                from beanllm.domain.knowledge_graph.ner_engines import NEREngineFactory

                kwargs = {}
                if model:
                    kwargs["model"] = model
                if labels and engine == "gliner":
                    kwargs["labels"] = labels

                self._ner_engine = NEREngineFactory.create(engine, **kwargs)
                logger.info(f"EntityExtractor initialized: engine={engine}")
            except ImportError as e:
                logger.warning(f"Failed to create NER engine {engine}: {e}")
                logger.info("Falling back to regex/LLM mode")
        elif llm_function:
            logger.info("EntityExtractor initialized: mode=LLM")
        else:
            logger.info("EntityExtractor initialized: mode=Regex (fallback)")

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
        """
        logger.info(f"Extracting entities from text ({len(text)} chars)")

        entity_types_to_extract = entity_types or list(EntityType)
        entities = []

        # 1. NER 엔진 기반 추출 시도
        if self._ner_engine:
            try:
                entities = self._extract_with_ner_engine(text, entity_types_to_extract)
                logger.info(f"NER engine extracted {len(entities)} entities")
            except Exception as e:
                logger.warning(f"NER engine extraction failed: {e}")
                if self._use_fallback and self._llm_function:
                    entities = self._extract_with_llm(text, entity_types_to_extract)
                elif self._use_fallback:
                    entities = self._extract_with_regex(text, entity_types_to_extract)

        # 2. LLM 기반 추출 시도
        elif self._llm_function:
            try:
                entities = self._extract_with_llm(text, entity_types_to_extract)
                logger.info(f"LLM extracted {len(entities)} entities")
            except Exception as e:
                logger.warning(f"LLM extraction failed: {e}")
                if self._use_fallback:
                    logger.info("Falling back to regex-based extraction")
                    entities = self._extract_with_regex(text, entity_types_to_extract)

        # 3. Fallback to regex
        else:
            entities = self._extract_with_regex(text, entity_types_to_extract)

        # Deduplicate
        entities = self._deduplicate_entities(entities)

        # Filter by confidence
        entities = [e for e in entities if e.confidence >= min_confidence]

        # Cache entities
        for entity in entities:
            self._entity_cache[entity.id] = entity

        logger.info(f"Extracted {len(entities)} entities (confidence >= {min_confidence})")
        return entities

    def _extract_with_llm(
        self,
        text: str,
        entity_types: List[EntityType],
    ) -> List[Entity]:
        """LLM을 사용한 엔티티 추출"""
        # 텍스트가 너무 길면 청킹
        if len(text) > self._max_text_length:
            return self._extract_with_llm_chunked(text, entity_types)

        # 프롬프트 생성
        type_names = [et.value for et in entity_types]
        prompt = ENTITY_EXTRACTION_PROMPT.format(
            text=text,
            entity_types=", ".join(type_names),
        )

        # LLM 호출
        assert self._llm_function is not None
        response = self._llm_function(prompt)

        # JSON 파싱
        entities = self._parse_entity_json(response)

        return entities

    def _extract_with_llm_chunked(
        self,
        text: str,
        entity_types: List[EntityType],
    ) -> List[Entity]:
        """긴 텍스트를 청킹하여 추출"""
        chunks = self._chunk_text(text, self._max_text_length)
        all_entities = []

        for chunk in chunks:
            chunk_entities = self._extract_with_llm(chunk, entity_types)
            all_entities.extend(chunk_entities)

        return self._deduplicate_entities(all_entities)

    def _extract_with_ner_engine(
        self,
        text: str,
        entity_types: List[EntityType],
    ) -> List[Entity]:
        """NER 엔진을 사용한 엔티티 추출"""
        assert self._ner_engine is not None
        result = self._ner_engine.extract_with_timing(text)
        entities = []

        # NER 엔진 레이블 -> EntityType 매핑
        label_to_type = {
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

        for ner_entity in result.entities:
            entity_type = label_to_type.get(ner_entity.label.upper(), EntityType.OTHER)

            # 타입 필터링
            if entity_type not in entity_types:
                continue

            entity_id = self._generate_entity_id(ner_entity.text, entity_type)

            entity = Entity(
                id=entity_id,
                name=ner_entity.text,
                type=entity_type,
                confidence=ner_entity.confidence,
            )
            entity.add_mention(
                doc_id="extracted",
                start=ner_entity.start,
                end=ner_entity.end,
                context=text[max(0, ner_entity.start - 20) : ner_entity.end + 20],
            )
            entities.append(entity)

        return entities

    def _chunk_text(self, text: str, max_length: int) -> List[str]:
        """텍스트를 문장 단위로 청킹"""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_length:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _parse_entity_json(self, response: str) -> List[Entity]:
        """LLM 응답에서 JSON 파싱하여 Entity 객체로 변환"""
        entities = []

        # JSON 추출 (코드 블록 내에 있을 수 있음)
        json_match = re.search(r"\[[\s\S]*\]", response)
        if not json_match:
            logger.warning("No JSON array found in LLM response")
            return entities

        try:
            data = json.loads(json_match.group())

            for item in data:
                name = item.get("name", "")
                type_str = item.get("type", "other").lower()
                description = item.get("description", "")
                aliases = item.get("aliases", [])
                confidence = item.get("confidence", 0.8)

                # EntityType 매핑
                entity_type = self._str_to_entity_type(type_str)
                entity_id = self._generate_entity_id(name, entity_type)

                entity = Entity(
                    id=entity_id,
                    name=name,
                    type=entity_type,
                    description=description,
                    aliases=aliases if isinstance(aliases, list) else [],
                    confidence=float(confidence),
                )
                entities.append(entity)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")

        return entities

    def _str_to_entity_type(self, type_str: str) -> EntityType:
        """문자열을 EntityType으로 변환"""
        type_map = {
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
        return type_map.get(type_str.lower(), EntityType.OTHER)

    def _extract_with_regex(
        self,
        text: str,
        entity_types: List[EntityType],
    ) -> List[Entity]:
        """Regex 기반 엔티티 추출 (fallback)"""
        entities = []

        # 확장된 패턴
        patterns = {
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

        for entity_type in entity_types:
            if entity_type not in patterns:
                continue

            for pattern in patterns[entity_type]:
                matches = re.finditer(pattern, text)

                for match in matches:
                    # 가장 긴 그룹 선택
                    name = max(match.groups(), key=lambda x: len(x) if x else 0)
                    if not name:
                        continue

                    entity_id = self._generate_entity_id(name, entity_type)

                    entity = Entity(
                        id=entity_id,
                        name=name.strip(),
                        type=entity_type,
                        confidence=0.6,  # Regex는 낮은 신뢰도
                    )
                    entities.append(entity)

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
        """
        logger.info(f"Resolving coreferences for {len(entities)} entities")

        # LLM 기반 coreference resolution
        if self._llm_function and entities:
            try:
                return self._resolve_with_llm(entities, text)
            except Exception as e:
                logger.warning(f"LLM coreference resolution failed: {e}")

        # Fallback: heuristic-based resolution
        return self._resolve_with_heuristics(entities, text)

    def _resolve_with_llm(
        self,
        entities: List[Entity],
        text: str,
    ) -> List[Entity]:
        """LLM 기반 coreference resolution"""
        entity_list = [{"name": e.name, "type": e.type.value} for e in entities]

        prompt = COREFERENCE_RESOLUTION_PROMPT.format(
            text=text,
            entities=json.dumps(entity_list, ensure_ascii=False),
        )

        assert self._llm_function is not None
        response = self._llm_function(prompt)

        # JSON 파싱
        json_match = re.search(r"\[[\s\S]*\]", response)
        if not json_match:
            return entities

        try:
            resolutions = json.loads(json_match.group())

            # 엔티티 이름 -> 엔티티 매핑
            entity_map = {e.name.lower(): e for e in entities}

            for resolution in resolutions:
                reference = resolution.get("reference", "")
                resolved_to = resolution.get("resolved_to", "")
                confidence = resolution.get("confidence", 0.8)

                if resolved_to.lower() in entity_map:
                    target_entity = entity_map[resolved_to.lower()]
                    target_entity.add_alias(reference)
                    # 언급 위치 추가
                    start = resolution.get("start_position", 0)
                    target_entity.add_mention("resolved", start, start + len(reference), reference)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse coreference JSON: {e}")

        return entities

    def _resolve_with_heuristics(
        self,
        entities: List[Entity],
        text: str,
    ) -> List[Entity]:
        """휴리스틱 기반 coreference resolution"""
        # 대명사 패턴
        pronouns = {
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

        # 엔티티를 타입별로 그룹화
        entities_by_type: Dict[EntityType, List[Entity]] = {}
        for entity in entities:
            if entity.type not in entities_by_type:
                entities_by_type[entity.type] = []
            entities_by_type[entity.type].append(entity)

        # 대명사 찾기 및 해결
        for pronoun, expected_type in pronouns.items():
            pattern = rf"\b{pronoun}\b"
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # 가장 가까운 이전 엔티티 찾기 (간단한 휴리스틱)
                if expected_type and expected_type in entities_by_type:
                    candidates = entities_by_type[expected_type]
                    if candidates:
                        # 첫 번째 매칭 엔티티에 alias 추가
                        candidates[0].add_alias(pronoun)

        # 중복 제거 및 병합
        entity_map: Dict[str, Entity] = {}
        for entity in entities:
            canonical = self._canonicalize_name(entity.name)
            if canonical in entity_map:
                entity_map[canonical].merge_with(entity)
            else:
                entity_map[canonical] = entity

        resolved = list(entity_map.values())
        logger.info(f"Resolved to {len(resolved)} entities")
        return resolved

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
