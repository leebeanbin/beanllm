"""spaCy NER engine implementation."""

from __future__ import annotations

from typing import List, Optional

from beanllm.domain.knowledge_graph.ner_base import BaseNEREngine
from beanllm.domain.knowledge_graph.ner_models import NEREntity
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class SpacyNEREngine(BaseNEREngine):
    """
    spaCy NER 엔진

    장점: 빠름, 정확함, 다국어 지원
    단점: 고정된 엔티티 타입

    Models:
    - en_core_web_sm (small, fast)
    - en_core_web_md (medium)
    - en_core_web_lg (large, accurate)
    - en_core_web_trf (transformer, most accurate)
    - ko_core_news_sm (Korean)
    """

    def __init__(
        self,
        model: str = "en_core_web_sm",
        disable: Optional[List[str]] = None,
    ) -> None:
        super().__init__(f"spacy:{model}")
        self.model_name = model
        self.disable = disable or ["parser", "tagger", "lemmatizer"]
        self._nlp = None

    def load(self) -> None:
        if self._is_loaded:
            return

        try:
            import spacy  # type: ignore[import-untyped]

            self._nlp = spacy.load(self.model_name, disable=self.disable)
            self._is_loaded = True
            logger.info(f"spaCy model loaded: {self.model_name}")
        except OSError:
            logger.warning(f"spaCy model not found: {self.model_name}")
            logger.info(f"Install with: python -m spacy download {self.model_name}")
            raise

    def extract(self, text: str) -> List[NEREntity]:
        if not self._is_loaded:
            self.load()
        assert self._nlp is not None

        doc = self._nlp(text)
        entities = []

        for ent in doc.ents:
            entities.append(
                NEREntity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=1.0,  # spaCy doesn't provide confidence
                    source=self.name,
                )
            )

        return entities
