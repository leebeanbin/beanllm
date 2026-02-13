"""
Neo4jAdapter - 역호환 래퍼 (deprecated)

실제 구현은 infrastructure/knowledge_graph/neo4j_adapter.py에 있습니다.
Clean Architecture: 외부 DB 연동은 Infrastructure 레이어에 속합니다.

이 파일은 역호환성을 위해 유지하되, domain 레이어에서 infrastructure를
직접 import하지 않도록 lazy import로 처리합니다.

사용자는 아래 경로에서 직접 import하는 것을 권장합니다:
    from beanllm.infrastructure.knowledge_graph.neo4j_adapter import Neo4jAdapter
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from beanllm.infrastructure.knowledge_graph.neo4j_adapter import (
        Neo4jAdapter as _Neo4jAdapter,
    )


def __getattr__(name: str) -> Any:
    if name == "Neo4jAdapter":
        from beanllm.infrastructure.knowledge_graph.neo4j_adapter import Neo4jAdapter

        return Neo4jAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Neo4jAdapter"]  # noqa: F822 (lazy import via __getattr__)
