"""
NER Engines - Named Entity Recognition 엔진 모음 (Re-export hub)

다양한 NER 모델을 통합하여 사용할 수 있습니다.
정확도 벤치마크 기능도 제공합니다.

지원 엔진:
1. spaCy NER - 빠르고 정확한 rule-based + statistical NER
2. Hugging Face NER - BERT/RoBERTa 기반 transformer NER
3. GLiNER - Zero-shot NER (새로운 엔티티 타입에 유연)
4. Flair NER - Contextual string embeddings 기반
5. LLM NER - GPT/Claude 등 LLM 기반 (고비용, 고정확도)

Example:
    ```python
    from beanllm.domain.knowledge_graph.ner_engines import (
        NEREngineFactory,
        NERBenchmark,
    )

    # 엔진 생성
    spacy_engine = NEREngineFactory.create("spacy")
    hf_engine = NEREngineFactory.create("huggingface", model="dslim/bert-base-NER")
    gliner_engine = NEREngineFactory.create("gliner")

    # 엔티티 추출
    entities = spacy_engine.extract("Steve Jobs founded Apple.")

    # 벤치마크
    benchmark = NERBenchmark(engines=[spacy_engine, hf_engine, gliner_engine])
    results = benchmark.run(test_data)
    print(benchmark.get_report())
    ```
"""

from __future__ import annotations

from beanllm.domain.knowledge_graph.ner_benchmark import (
    NERBenchmark,
    SAMPLE_TEST_DATA,
    quick_benchmark,
)
from beanllm.domain.knowledge_graph.ner_base import BaseNEREngine
from beanllm.domain.knowledge_graph.ner_factory import NEREngineFactory
from beanllm.domain.knowledge_graph.ner_flair import FlairNEREngine
from beanllm.domain.knowledge_graph.ner_gliner import GLiNEREngine
from beanllm.domain.knowledge_graph.ner_huggingface import HuggingFaceNEREngine
from beanllm.domain.knowledge_graph.ner_llm import LLMNEREngine
from beanllm.domain.knowledge_graph.ner_models import (
    BenchmarkResult,
    BenchmarkSample,
    NEREntity,
    NERResult,
)
from beanllm.domain.knowledge_graph.ner_spacy import SpacyNEREngine

__all__ = [
    "BaseNEREngine",
    "SpacyNEREngine",
    "HuggingFaceNEREngine",
    "GLiNEREngine",
    "FlairNEREngine",
    "LLMNEREngine",
    "NEREngineFactory",
    "NEREntity",
    "NERResult",
    "BenchmarkSample",
    "BenchmarkResult",
    "NERBenchmark",
    "SAMPLE_TEST_DATA",
    "quick_benchmark",
]
