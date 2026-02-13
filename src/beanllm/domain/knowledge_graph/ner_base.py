"""Base NER engine abstract class."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import List

from beanllm.domain.knowledge_graph.ner_models import NEREntity, NERResult


class BaseNEREngine(ABC):
    """NER 엔진 기본 클래스"""

    def __init__(self, name: str) -> None:
        self.name = name
        self._is_loaded = False

    @abstractmethod
    def load(self) -> None:
        """모델 로드 (lazy loading)"""
        pass

    @abstractmethod
    def extract(self, text: str) -> List[NEREntity]:
        """엔티티 추출"""
        pass

    def extract_with_timing(self, text: str) -> NERResult:
        """타이밍 포함 추출"""
        if not self._is_loaded:
            self.load()

        start = time.time()
        entities = self.extract(text)
        latency = (time.time() - start) * 1000

        return NERResult(
            entities=entities,
            engine_name=self.name,
            latency_ms=latency,
        )
