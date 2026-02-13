"""LLM-based NER engine implementation."""

from __future__ import annotations

import json
import re
from typing import Callable, List, Optional

from beanllm.domain.knowledge_graph.ner_base import BaseNEREngine
from beanllm.domain.knowledge_graph.ner_models import NEREntity


class LLMNEREngine(BaseNEREngine):
    """
    LLM 기반 NER 엔진

    장점: 유연한 엔티티 타입, 높은 정확도
    단점: 비용, 속도
    """

    def __init__(
        self,
        llm_function: Callable[[str], str],
        labels: Optional[List[str]] = None,
    ) -> None:
        super().__init__("llm")
        self._llm_function = llm_function
        self.labels = labels or [
            "PERSON",
            "ORGANIZATION",
            "LOCATION",
            "DATE",
            "PRODUCT",
            "TECHNOLOGY",
        ]
        self._is_loaded = True

    def load(self) -> None:
        pass  # No loading needed

    def extract(self, text: str) -> List[NEREntity]:
        prompt = f"""Extract named entities from the following text.

Text: {text}

Entity types to extract: {", ".join(self.labels)}

Return as JSON array:
[{{"text": "entity text", "label": "ENTITY_TYPE", "start": 0, "end": 10, "confidence": 0.95}}]

JSON:"""

        response = self._llm_function(prompt)

        # Parse JSON
        entities = []
        json_match = re.search(r"\[[\s\S]*\]", response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                for item in data:
                    entities.append(
                        NEREntity(
                            text=item.get("text", ""),
                            label=item.get("label", "OTHER"),
                            start=item.get("start", 0),
                            end=item.get("end", 0),
                            confidence=item.get("confidence", 0.9),
                            source=self.name,
                        )
                    )
            except json.JSONDecodeError:
                pass

        return entities
