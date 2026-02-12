"""
Neo4jAdapter - 역호환 래퍼

실제 구현은 infrastructure/knowledge_graph/neo4j_adapter.py로 이동됨.
Clean Architecture: 외부 DB 연동은 Infrastructure 레이어에 속함.
"""

from beanllm.infrastructure.knowledge_graph.neo4j_adapter import Neo4jAdapter

__all__ = ["Neo4jAdapter"]
