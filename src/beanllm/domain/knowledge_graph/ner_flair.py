"""Flair NER engine implementation."""

from __future__ import annotations

from typing import Any, List, Optional

from beanllm.domain.knowledge_graph.ner_base import BaseNEREngine
from beanllm.domain.knowledge_graph.ner_models import NEREntity
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class FlairNEREngine(BaseNEREngine):
    """
    Flair NER 엔진

    장점: 높은 정확도, contextual embeddings
    단점: 무거움

    Models:
    - flair/ner-english (English)
    - flair/ner-english-large (English, accurate)
    - flair/ner-multi (Multilingual)
    """

    def __init__(self, model: str = "flair/ner-english") -> None:
        super().__init__(f"flair:{model.split('/')[-1]}")
        self.model_name = model
        self._tagger: Optional[Any] = None
        self._Sentence: Optional[Any] = None

    def load(self) -> None:
        if self._is_loaded:
            return

        try:
            from flair.data import Sentence  # type: ignore[import-untyped]
            from flair.models import SequenceTagger  # type: ignore[import-untyped]

            self._tagger = SequenceTagger.load(self.model_name)
            self._Sentence = Sentence
            self._is_loaded = True
            logger.info(f"Flair NER model loaded: {self.model_name}")
        except ImportError:
            logger.error("Flair not installed. Install with: pip install flair")
            raise
        except Exception as e:
            logger.error(f"Failed to load Flair model: {e}")
            raise

    def extract(self, text: str) -> List[NEREntity]:
        if not self._is_loaded:
            self.load()
        assert self._tagger is not None and self._Sentence is not None

        sentence = self._Sentence(text)
        self._tagger.predict(sentence)

        entities = []
        for entity in sentence.get_spans("ner"):
            entities.append(
                NEREntity(
                    text=entity.text,
                    label=entity.get_label("ner").value,
                    start=entity.start_position,
                    end=entity.end_position,
                    confidence=entity.get_label("ner").score,
                    source=self.name,
                )
            )

        return entities
