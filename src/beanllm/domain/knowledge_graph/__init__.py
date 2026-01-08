"""
Knowledge Graph Domain - 지식 그래프 구축 및 쿼리

Phase 5: Knowledge Graph Builder
- EntityExtractor: LLM 기반 엔티티 추출
- RelationExtractor: 엔티티 간 관계 추출
- GraphBuilder: NetworkX 기반 그래프 구축
- GraphQuerier: 그래프 쿼리 인터페이스
- GraphRAG: 그래프 기반 RAG
- Neo4jAdapter: Neo4j 데이터베이스 연동 (optional)
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
]
