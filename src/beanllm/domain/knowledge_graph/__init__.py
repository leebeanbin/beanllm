"""
Knowledge Graph Domain - 지식 그래프 구축 및 쿼리

Phase 5: Knowledge Graph Builder
- EntityExtractor: 멀티 엔진 엔티티 추출 (spaCy, HuggingFace, GLiNER, Flair, LLM)
- RelationExtractor: 엔티티 간 관계 추출
- GraphBuilder: NetworkX 기반 그래프 구축
- GraphQuerier: 그래프 쿼리 인터페이스
- GraphRAG: 그래프 기반 RAG
- Neo4jAdapter: Neo4j 데이터베이스 연동 (optional)

NER Engines:
- SpacyNEREngine: 빠르고 정확한 통계 기반 NER
- HuggingFaceNEREngine: BERT/RoBERTa transformer NER
- GLiNEREngine: Zero-shot NER (커스텀 엔티티 타입)
- FlairNEREngine: Contextual embeddings NER
- LLMNEREngine: LLM 기반 NER

Benchmark:
- NERBenchmark: NER 엔진 비교 벤치마크
- quick_benchmark: 빠른 벤치마크 실행

Example:
    ```python
    from beanllm.domain.knowledge_graph import (
        EntityExtractor,
        NEREngineFactory,
        NERBenchmark,
    )

    # spaCy 엔진 사용
    extractor = EntityExtractor(engine="spacy")
    entities = extractor.extract_entities("Steve Jobs founded Apple.")

    # 벤치마크
    benchmark = NERBenchmark(engines=[
        NEREngineFactory.create("spacy"),
        NEREngineFactory.create("huggingface"),
    ])
    results = benchmark.run(test_data)
    print(benchmark.get_report())
    ```
"""

from .entity_extractor import (
    Entity,
    EntityExtractor,
    EntityType,
    extract_entities_simple,
)
from .graph_builder import GraphBuilder, build_graph_simple
from .graph_querier import GraphQuerier
from .graph_rag import GraphRAG
from .neo4j_adapter import Neo4jAdapter
from .relation_extractor import (
    Relation,
    RelationExtractor,
    RelationType,
    extract_relations_simple,
)

# NER Engines (optional - may require additional dependencies)
try:
    from .ner_engines import (
        SAMPLE_TEST_DATA,
        BaseNEREngine,
        BenchmarkResult,
        BenchmarkSample,
        FlairNEREngine,
        GLiNEREngine,
        HuggingFaceNEREngine,
        LLMNEREngine,
        NERBenchmark,
        NEREngineFactory,
        NEREntity,
        NERResult,
        SpacyNEREngine,
        quick_benchmark,
    )

    _NER_AVAILABLE = True
except ImportError:
    _NER_AVAILABLE = False

__all__ = [
    # Entity Extraction
    "EntityExtractor",
    "Entity",
    "EntityType",
    "extract_entities_simple",
    # Relation Extraction
    "RelationExtractor",
    "Relation",
    "RelationType",
    "extract_relations_simple",
    # Graph Building
    "GraphBuilder",
    "build_graph_simple",
    # Graph Querying
    "GraphQuerier",
    # Graph RAG
    "GraphRAG",
    # Neo4j Adapter
    "Neo4jAdapter",
    # NER Engines
    "BaseNEREngine",
    "SpacyNEREngine",
    "HuggingFaceNEREngine",
    "GLiNEREngine",
    "FlairNEREngine",
    "LLMNEREngine",
    "NEREngineFactory",
    "NERBenchmark",
    "BenchmarkSample",
    "BenchmarkResult",
    "NEREntity",
    "NERResult",
    "quick_benchmark",
    "SAMPLE_TEST_DATA",
]
