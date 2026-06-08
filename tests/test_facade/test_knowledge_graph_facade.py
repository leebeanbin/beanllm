"""
KnowledgeGraph Facade 테스트
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from beanllm.dto.response.graph.kg_response import (
    BuildGraphResponse,
    EntitiesResponse,
    GraphRAGResponse,
    QueryGraphResponse,
    RelationsResponse,
)
from beanllm.facade.advanced.knowledge_graph_facade import KnowledgeGraph
from beanllm.handler.advanced.knowledge_graph_handler import KnowledgeGraphHandler


def _make_handler() -> MagicMock:
    handler = AsyncMock(spec=KnowledgeGraphHandler)
    handler.handle_extract_entities.return_value = EntitiesResponse(
        document_id="doc-1",
        entities=[{"name": "Apple", "type": "organization", "id": "e1"}],
        num_entities=1,
        entity_counts_by_type={"organization": 1},
    )
    handler.handle_extract_relations.return_value = RelationsResponse(
        document_id="doc-1",
        relations=[{"source_id": "e1", "target_id": "e2", "type": "founded"}],
        num_relations=1,
        relation_counts_by_type={"founded": 1},
    )
    handler.handle_build_graph.return_value = BuildGraphResponse(
        graph_id="graph-1",
        graph_name="test",
        num_nodes=3,
        num_edges=2,
        backend="networkx",
        document_ids=["doc-1"],
        created_at="2026-01-01T00:00:00",
        statistics={},
    )
    handler.handle_query_graph.return_value = QueryGraphResponse(
        graph_id="graph-1",
        query="test",
        results=[{"name": "Apple"}],
        num_results=1,
        execution_time=0.01,
    )
    handler.handle_graph_rag.return_value = GraphRAGResponse(
        answer="Apple was founded by Steve Jobs.",
        entities_used=["Apple"],
        reasoning_paths=[["Apple", "founded_by", "Steve Jobs"]],
        graph_context="context",
    )
    handler.handle_visualize_graph.return_value = "[Apple]"
    handler.handle_get_graph_stats.return_value = {"num_nodes": 3, "num_edges": 2, "density": 0.5}
    return handler


class TestKnowledgeGraphFacade:
    @pytest.fixture
    def kg(self) -> KnowledgeGraph:
        handler = _make_handler()
        return KnowledgeGraph(handler=handler)

    async def test_extract_entities(self, kg: KnowledgeGraph) -> None:
        result = await kg.extract_entities(text="Apple was founded by Steve Jobs.")
        assert isinstance(result, EntitiesResponse)
        assert result.num_entities == 1

    async def test_extract_entities_with_types(self, kg: KnowledgeGraph) -> None:
        result = await kg.extract_entities(
            text="Apple was founded by Steve Jobs.",
            entity_types=["person", "organization"],
        )
        assert isinstance(result, EntitiesResponse)

    async def test_extract_relations(self, kg: KnowledgeGraph) -> None:
        entities = [
            {"name": "Apple", "type": "organization"},
            {"name": "Steve Jobs", "type": "person"},
        ]
        result = await kg.extract_relations(
            text="Apple was founded by Steve Jobs.",
            entities=entities,
        )
        assert isinstance(result, RelationsResponse)
        assert result.num_relations == 1

    async def test_build_graph(self, kg: KnowledgeGraph) -> None:
        result = await kg.build_graph(documents=["Apple was founded by Steve Jobs in 1976."])
        assert isinstance(result, BuildGraphResponse)
        assert result.graph_id == "graph-1"
        assert result.num_nodes == 3

    async def test_build_graph_with_options(self, kg: KnowledgeGraph) -> None:
        result = await kg.build_graph(
            documents=["Apple was founded by Steve Jobs."],
            graph_id="custom-id",
            entity_types=["person", "organization"],
        )
        assert isinstance(result, BuildGraphResponse)

    async def test_query_graph(self, kg: KnowledgeGraph) -> None:
        result = await kg.query_graph(
            graph_id="graph-1",
            query_type="find_entities_by_type",
            params={"entity_type": "organization"},
        )
        assert isinstance(result, QueryGraphResponse)
        assert result.num_results == 1

    async def test_graph_rag(self, kg: KnowledgeGraph) -> None:
        result = await kg.graph_rag(
            query="Who founded Apple?",
            graph_id="graph-1",
        )
        assert isinstance(result, GraphRAGResponse)
        assert "Apple" in result.answer

    async def test_visualize_graph(self, kg: KnowledgeGraph) -> None:
        result = await kg.visualize_graph("graph-1")
        assert isinstance(result, str)
        assert len(result) > 0

    async def test_get_graph_stats(self, kg: KnowledgeGraph) -> None:
        result = await kg.get_graph_stats("graph-1")
        assert isinstance(result, dict)
        assert "num_nodes" in result
        assert result["num_nodes"] == 3

    async def test_kg_has_handler(self, kg: KnowledgeGraph) -> None:
        assert kg._handler is not None
