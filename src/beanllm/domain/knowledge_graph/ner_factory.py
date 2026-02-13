"""NER engine factory."""

from __future__ import annotations

from typing import Any, Callable, List, cast

from beanllm.domain.knowledge_graph.ner_base import BaseNEREngine
from beanllm.domain.knowledge_graph.ner_flair import FlairNEREngine
from beanllm.domain.knowledge_graph.ner_gliner import GLiNEREngine
from beanllm.domain.knowledge_graph.ner_huggingface import HuggingFaceNEREngine
from beanllm.domain.knowledge_graph.ner_llm import LLMNEREngine
from beanllm.domain.knowledge_graph.ner_spacy import SpacyNEREngine


class NEREngineFactory:
    """NER 엔진 팩토리"""

    _engines = {
        "spacy": SpacyNEREngine,
        "huggingface": HuggingFaceNEREngine,
        "hf": HuggingFaceNEREngine,
        "gliner": GLiNEREngine,
        "flair": FlairNEREngine,
    }

    @classmethod
    def create(cls, engine_type: str, **kwargs: Any) -> BaseNEREngine:
        """
        NER 엔진 생성

        Args:
            engine_type: "spacy", "huggingface", "gliner", "flair"
            **kwargs: 엔진별 설정

        Returns:
            BaseNEREngine 인스턴스
        """
        engine_class = cls._engines.get(engine_type.lower())
        if not engine_class:
            raise ValueError(
                f"Unknown engine type: {engine_type}. Available: {list(cls._engines.keys())}"
            )

        return cast(BaseNEREngine, engine_class(**kwargs))

    @classmethod
    def create_llm(cls, llm_function: Callable[[str], str], **kwargs: Any) -> LLMNEREngine:
        """LLM NER 엔진 생성"""
        return LLMNEREngine(llm_function=llm_function, **kwargs)

    @classmethod
    def list_available(cls) -> List[str]:
        """사용 가능한 엔진 목록"""
        return list(cls._engines.keys())
