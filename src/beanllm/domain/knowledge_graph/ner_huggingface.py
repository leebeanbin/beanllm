"""HuggingFace NER engine implementation."""

from __future__ import annotations

from typing import List, Optional

from beanllm.domain.knowledge_graph.ner_base import BaseNEREngine
from beanllm.domain.knowledge_graph.ner_models import NEREntity
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class HuggingFaceNEREngine(BaseNEREngine):
    """
    Hugging Face NER 엔진

    장점: 다양한 모델, 높은 정확도
    단점: GPU 권장, 무거움

    Recommended Models:
    - dslim/bert-base-NER (English, general)
    - dslim/bert-large-NER (English, accurate)
    - Jean-Baptiste/camembert-ner (French)
    - xlm-roberta-large-finetuned-conll03-english (Multilingual)
    """

    def __init__(
        self,
        model: str = "dslim/bert-base-NER",
        device: Optional[str] = None,
        aggregation_strategy: str = "simple",
    ) -> None:
        super().__init__(f"hf:{model.split('/')[-1]}")
        self.model_name = model
        self.device = device
        self.aggregation_strategy = aggregation_strategy
        self._pipeline = None

    def load(self) -> None:
        if self._is_loaded:
            return

        try:
            from transformers import pipeline  # type: ignore[import-untyped]

            self._pipeline = pipeline(
                "ner",
                model=self.model_name,
                device=self.device,
                aggregation_strategy=self.aggregation_strategy,
            )
            self._is_loaded = True
            logger.info(f"HuggingFace NER model loaded: {self.model_name}")
        except ImportError:
            logger.error("transformers not installed. Install with: pip install transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            raise

    def extract(self, text: str) -> List[NEREntity]:
        if not self._is_loaded:
            self.load()
        assert self._pipeline is not None

        results = self._pipeline(text)
        entities = []

        for item in results:
            entities.append(
                NEREntity(
                    text=item.get("word", item.get("entity_group", "")),
                    label=item.get("entity_group", item.get("entity", ""))
                    .replace("B-", "")
                    .replace("I-", ""),
                    start=item.get("start", 0),
                    end=item.get("end", 0),
                    confidence=item.get("score", 1.0),
                    source=self.name,
                )
            )

        return entities
