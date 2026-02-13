"""GLiNER NER engine implementation."""

from __future__ import annotations

from typing import List, Optional

from beanllm.domain.knowledge_graph.ner_base import BaseNEREngine
from beanllm.domain.knowledge_graph.ner_models import NEREntity
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class GLiNEREngine(BaseNEREngine):
    """
    GLiNER - Zero-shot NER 엔진

    장점: 커스텀 엔티티 타입 지원, 학습 불필요
    단점: 속도가 느릴 수 있음

    Models:
    - urchade/gliner_small (fast)
    - urchade/gliner_medium (balanced)
    - urchade/gliner_large (accurate)
    - urchade/gliner_multi (multilingual)
    """

    def __init__(
        self,
        model: str = "urchade/gliner_small",
        labels: Optional[List[str]] = None,
        threshold: float = 0.5,
    ) -> None:
        super().__init__(f"gliner:{model.split('/')[-1]}")
        self.model_name = model
        self.labels = labels or [
            "person",
            "organization",
            "location",
            "date",
            "product",
            "technology",
            "event",
        ]
        self.threshold = threshold
        self._model = None

    def load(self) -> None:
        if self._is_loaded:
            return

        try:
            from gliner import GLiNER  # type: ignore[import-untyped]

            self._model = GLiNER.from_pretrained(self.model_name)
            self._is_loaded = True
            logger.info(f"GLiNER model loaded: {self.model_name}")
        except ImportError:
            logger.error("GLiNER not installed. Install with: pip install gliner")
            raise
        except Exception as e:
            logger.error(f"Failed to load GLiNER model: {e}")
            raise

    def extract(self, text: str) -> List[NEREntity]:
        if not self._is_loaded:
            self.load()
        assert self._model is not None

        results = self._model.predict_entities(text, self.labels, threshold=self.threshold)
        entities = []

        for item in results:
            entities.append(
                NEREntity(
                    text=item["text"],
                    label=item["label"].upper(),
                    start=item["start"],
                    end=item["end"],
                    confidence=item.get("score", 1.0),
                    source=self.name,
                )
            )

        return entities

    def set_labels(self, labels: List[str]) -> None:
        """커스텀 레이블 설정"""
        self.labels = labels
