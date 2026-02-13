"""
Coreference Resolver - 대명사 해결 로직

LLM 기반 또는 휴리스틱 기반으로 텍스트 내
대명사를 실제 엔티티에 매핑합니다.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from beanllm.domain.knowledge_graph.entity_models import Entity, EntityType
from beanllm.domain.knowledge_graph.entity_patterns import PRONOUN_TO_ENTITY_TYPE
from beanllm.domain.knowledge_graph.entity_prompts import COREFERENCE_RESOLUTION_PROMPT
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


def resolve_coreferences(
    entities: List[Entity],
    text: str,
    llm_function: Optional[Callable[[str], str]] = None,
    canonicalize_fn: Optional[Callable[[str], str]] = None,
) -> List[Entity]:
    """
    Coreference resolution (대명사 해결)

    Args:
        entities: 엔티티 리스트
        text: 원본 텍스트
        llm_function: LLM 호출 함수 (선택)
        canonicalize_fn: 이름 정규화 함수 (선택)

    Returns:
        List[Entity]: Coreference가 해결된 엔티티 리스트
    """
    logger.info(f"Resolving coreferences for {len(entities)} entities")

    # LLM 기반 coreference resolution
    if llm_function and entities:
        try:
            return _resolve_with_llm(entities, text, llm_function)
        except Exception as e:
            logger.warning(f"LLM coreference resolution failed: {e}")

    # Fallback: heuristic-based resolution
    return _resolve_with_heuristics(entities, text, canonicalize_fn)


def _resolve_with_llm(
    entities: List[Entity],
    text: str,
    llm_function: Callable[[str], str],
) -> List[Entity]:
    """LLM 기반 coreference resolution"""
    entity_list = [{"name": e.name, "type": e.type.value} for e in entities]

    prompt = COREFERENCE_RESOLUTION_PROMPT.format(
        text=text,
        entities=json.dumps(entity_list, ensure_ascii=False),
    )

    response = llm_function(prompt)

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
    entities: List[Entity],
    text: str,
    canonicalize_fn: Optional[Callable[[str], str]] = None,
) -> List[Entity]:
    """휴리스틱 기반 coreference resolution"""

    def _default_canonicalize(name: str) -> str:
        return name.lower().strip().replace("  ", " ")

    canonicalize = canonicalize_fn or _default_canonicalize

    # 엔티티를 타입별로 그룹화
    entities_by_type: Dict[EntityType, List[Entity]] = {}
    for entity in entities:
        if entity.type not in entities_by_type:
            entities_by_type[entity.type] = []
        entities_by_type[entity.type].append(entity)

    # 대명사 찾기 및 해결
    for pronoun, expected_type in PRONOUN_TO_ENTITY_TYPE.items():
        pattern = rf"\b{pronoun}\b"
        for match in re.finditer(pattern, text, re.IGNORECASE):
            # 가장 가까운 이전 엔티티 찾기 (간단한 휴리스틱)
            if expected_type and expected_type in entities_by_type:
                candidates = entities_by_type[expected_type]
                if candidates:
                    candidates[0].add_alias(pronoun)

    # 중복 제거 및 병합
    entity_map: Dict[str, Entity] = {}
    for entity in entities:
        canonical = canonicalize(entity.name)
        if canonical in entity_map:
            entity_map[canonical].merge_with(entity)
        else:
            entity_map[canonical] = entity

    resolved = list(entity_map.values())
    logger.info(f"Resolved to {len(resolved)} entities")
    return resolved
