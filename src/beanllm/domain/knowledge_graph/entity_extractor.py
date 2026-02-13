"""
EntityExtractor - LLM-based Named Entity Recognition

LLM을 활용한 고급 엔티티 추출기.

타입/모델: entity_models.py
프롬프트: entity_prompts.py

Example:
    ```python
    from beanllm.domain.knowledge_graph import EntityExtractor, EntityType

    extractor = EntityExtractor(llm_function=my_llm_call)
    entities = extractor.extract_entities(
        text="Steve Jobs founded Apple Inc. in 1976.",
        entity_types=[EntityType.PERSON, EntityType.ORGANIZATION]
    )
    ```
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from beanllm.domain.knowledge_graph.coreference_resolver import (
    resolve_coreferences as _resolve_coreferences,
)
from beanllm.domain.knowledge_graph.entity_models import Entity, EntityType
from beanllm.domain.knowledge_graph.entity_patterns import (
    ENTITY_REGEX_PATTERNS_COMPILED,
    NER_LABEL_TO_ENTITY_TYPE,
    STR_TO_ENTITY_TYPE,
)
from beanllm.domain.knowledge_graph.entity_prompts import (
    COREFERENCE_RESOLUTION_PROMPT,
    ENTITY_EXTRACTION_PROMPT,
    PROPERTY_EXTRACTION_PROMPT,
)
from beanllm.utils.logging import get_logger

# Re-export for backward compatibility
__all__ = [
    "Entity",
    "EntityType",
    "EntityExtractor",
    "extract_entities_simple",
    "ENTITY_EXTRACTION_PROMPT",
    "COREFERENCE_RESOLUTION_PROMPT",
    "PROPERTY_EXTRACTION_PROMPT",
]

logger = get_logger(__name__)


class EntityExtractor:
    """
    멀티 엔진 엔티티 추출기

    지원 엔진: spacy, huggingface, gliner, flair, llm, regex(fallback)
    패턴 정의: entity_patterns.py | 프롬프트: entity_prompts.py

    Example:
        >>> extractor = EntityExtractor(engine="spacy")
        >>> entities = extractor.extract_entities("Steve Jobs founded Apple.")
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

                kwargs: Dict[str, Any] = {}
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

        for ner_entity in result.entities:
            entity_type = NER_LABEL_TO_ENTITY_TYPE.get(ner_entity.label.upper(), EntityType.OTHER)

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
        entities: List[Entity] = []

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
        return STR_TO_ENTITY_TYPE.get(type_str.lower(), EntityType.OTHER)

    def _extract_with_regex(
        self,
        text: str,
        entity_types: List[EntityType],
    ) -> List[Entity]:
        """Regex 기반 엔티티 추출 (fallback, 사전 컴파일 패턴 사용)"""
        entities = []

        for entity_type in entity_types:
            if entity_type not in ENTITY_REGEX_PATTERNS_COMPILED:
                continue

            for compiled_pattern in ENTITY_REGEX_PATTERNS_COMPILED[entity_type]:
                matches = compiled_pattern.finditer(text)

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
        """Coreference resolution (대명사 해결) - coreference_resolver 모듈에 위임"""
        return _resolve_coreferences(
            entities,
            text,
            llm_function=self._llm_function,
            canonicalize_fn=self._canonicalize_name,
        )

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
        type_counts: Dict[str, int] = {}
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
