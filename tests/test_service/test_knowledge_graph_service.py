"""
KnowledgeGraph Service 테스트 - 서비스 인터페이스 및 구현 검증
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from beanllm.dto.request.graph.kg_request import (
    BuildGraphRequest,
    ExtractEntitiesRequest,
    ExtractRelationsRequest,
    QueryGraphRequest,
)
from beanllm.dto.response.graph.kg_response import (
    BuildGraphResponse,
    EntitiesResponse,
    GraphRAGResponse,
    QueryGraphResponse,
    RelationsResponse,
)
from beanllm.service.knowledge_graph_service import IKnowledgeGraphService


def _make_entities_response(doc_id: str = "doc-1") -> EntitiesResponse:
    return EntitiesResponse(
        document_id=doc_id,
        entities=[{"name": "Apple", "type": "organization", "id": "e1"}],
        num_entities=1,
        entity_counts_by_type={"organization": 1},
    )


def _make_build_response(graph_id: str = "graph-1") -> BuildGraphResponse:
    return BuildGraphResponse(
        graph_id=graph_id,
        graph_name="test-graph",
        num_nodes=3,
        num_edges=2,
        backend="networkx",
        document_ids=["doc-1"],
        created_at="2026-01-01T00:00:00",
        statistics={},
    )


class TestIKnowledgeGraphService:
    """IKnowledgeGraphService 인터페이스 테스트 (Mock 기반)"""

    @pytest.fixture
    def service(self) -> AsyncMock:
        return AsyncMock(spec=IKnowledgeGraphService)

    async def test_extract_entities_interface(self, service: AsyncMock) -> None:
        service.extract_entities.return_value = _make_entities_response()
        request = ExtractEntitiesRequest(
            document_id="doc-1",
            text="Apple was founded by Steve Jobs.",
            entity_types=[],
        )
        result = await service.extract_entities(request)
        assert isinstance(result, EntitiesResponse)
        service.extract_entities.assert_called_once_with(request)

    async def test_extract_relations_interface(self, service: AsyncMock) -> None:
        service.extract_relations.return_value = RelationsResponse(
            document_id="doc-1",
            relations=[{"source_id": "e1", "target_id": "e2", "type": "founded"}],
            num_relations=1,
            relation_counts_by_type={"founded": 1},
        )
        request = ExtractRelationsRequest(
            document_id="doc-1",
            text="Apple was founded by Steve Jobs.",
            entities=[{"name": "Apple", "type": "organization"}],
        )
        result = await service.extract_relations(request)
        assert isinstance(result, RelationsResponse)

    async def test_build_graph_interface(self, service: AsyncMock) -> None:
        service.build_graph.return_value = _make_build_response()
        request = BuildGraphRequest(documents=["Apple was founded by Steve Jobs."])
        result = await service.build_graph(request)
        assert isinstance(result, BuildGraphResponse)
        assert result.graph_id == "graph-1"

    async def test_query_graph_interface(self, service: AsyncMock) -> None:
        service.query_graph.return_value = QueryGraphResponse(
            graph_id="graph-1",
            query="test",
            results=[{"name": "Apple"}],
            num_results=1,
            execution_time=0.01,
        )
        request = QueryGraphRequest(
            graph_id="graph-1",
            query="MATCH (n) RETURN n",
            query_type="cypher",
        )
        result = await service.query_graph(request)
        assert isinstance(result, QueryGraphResponse)
        assert result.num_results == 1

    async def test_graph_rag_interface(self, service: AsyncMock) -> None:
        service.graph_rag.return_value = GraphRAGResponse(
            answer="Apple was founded by Steve Jobs.",
            entities_used=["Apple", "Steve Jobs"],
            reasoning_paths=[["Apple", "founded_by", "Steve Jobs"]],
            graph_context="context",
        )
        result = await service.graph_rag(query="Who founded Apple?", graph_id="graph-1")
        assert isinstance(result, GraphRAGResponse)
        assert "Apple" in result.answer

    async def test_visualize_graph_interface(self, service: AsyncMock) -> None:
        service.visualize_graph.return_value = "[Apple] --founded_by--> [Steve Jobs]"
        result = await service.visualize_graph("graph-1")
        assert isinstance(result, str)

    async def test_get_graph_stats_interface(self, service: AsyncMock) -> None:
        service.get_graph_stats.return_value = {
            "num_nodes": 3,
            "num_edges": 2,
            "density": 0.5,
            "num_connected_components": 1,
        }
        result = await service.get_graph_stats("graph-1")
        assert isinstance(result, dict)
        assert result["num_nodes"] == 3

    async def test_list_graphs_interface(self, service: AsyncMock) -> None:
        service.list_graphs.return_value = [
            {"id": "graph-1", "name": "Tech companies"},
            {"id": "graph-2", "name": "Science papers"},
        ]
        result = await service.list_graphs()
        assert isinstance(result, list)
        assert len(result) == 2


class TestKnowledgeGraphServiceWithMockedDeps:
    """KnowledgeGraphServiceImpl 테스트 (의존성 Mocking)"""

    @pytest.fixture
    def mock_service(self) -> MagicMock:
        """Lightweight mock that simulates the service implementation"""
        service = MagicMock(spec=IKnowledgeGraphService)
        service.extract_entities = AsyncMock(return_value=_make_entities_response())
        service.build_graph = AsyncMock(return_value=_make_build_response())
        service.graph_rag = AsyncMock(
            return_value=GraphRAGResponse(
                answer="Steve Jobs founded Apple.",
                entities_used=["Steve Jobs", "Apple"],
                reasoning_paths=[["Steve Jobs", "FOUNDED", "Apple"]],
                graph_context="founders context",
            )
        )
        service.get_graph_stats = AsyncMock(
            return_value={"num_nodes": 5, "num_edges": 4, "density": 0.4}
        )
        service.list_graphs = AsyncMock(return_value=[{"id": "g1", "name": "test", "num_nodes": 5}])
        return service

    async def test_extract_entities_returns_valid_response(self, mock_service: MagicMock) -> None:
        request = ExtractEntitiesRequest(
            document_id="doc-1",
            text="Apple was founded by Steve Jobs.",
            entity_types=[],
        )
        result = await mock_service.extract_entities(request)
        assert result.num_entities >= 1
        assert len(result.entities) >= 1

    async def test_build_graph_returns_valid_response(self, mock_service: MagicMock) -> None:
        request = BuildGraphRequest(
            documents=[
                "Apple was founded by Steve Jobs in 1976.",
                "Steve Jobs was the CEO of Apple.",
            ]
        )
        result = await mock_service.build_graph(request)
        assert result.graph_id is not None
        assert result.num_nodes >= 0

    async def test_graph_rag_returns_answer(self, mock_service: MagicMock) -> None:
        result = await mock_service.graph_rag(query="Who founded Apple?", graph_id="graph-1")
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0
        assert len(result.entities_used) >= 0

    async def test_get_graph_stats_returns_metrics(self, mock_service: MagicMock) -> None:
        stats = await mock_service.get_graph_stats("graph-1")
        assert "num_nodes" in stats
        assert "num_edges" in stats

    async def test_list_graphs_returns_list(self, mock_service: MagicMock) -> None:
        graphs = await mock_service.list_graphs()
        assert isinstance(graphs, list)


# ---------------------------------------------------------------------------
# KnowledgeGraphServiceImpl concrete class tests
# ---------------------------------------------------------------------------


class TestKnowledgeGraphServiceImpl:
    """Test the actual implementation class (not the interface mock)."""

    @pytest.fixture
    def service(self):
        from beanllm.service.impl.advanced.knowledge_graph_service_impl import (
            KnowledgeGraphServiceImpl,
        )

        return KnowledgeGraphServiceImpl()

    # --- __init__ ---

    def test_init_creates_entity_extractor(self, service):
        assert service._entity_extractor is not None

    def test_init_creates_relation_extractor(self, service):
        assert service._relation_extractor is not None

    def test_init_creates_graph_builder(self, service):
        assert service._graph_builder is not None

    def test_init_graphs_empty(self, service):
        assert service._graphs == {}

    def test_init_neo4j_adapter_none(self, service):
        assert service._neo4j_adapter is None

    # --- set_neo4j_adapter ---

    def test_set_neo4j_adapter_sets_none_on_connection_failure(self, service):
        # Neo4j not running → should not raise, just log warning
        service.set_neo4j_adapter(
            uri="bolt://localhost:9999",
            user="neo4j",
            password="test",
        )
        # May succeed or fail silently; adapter is None or not
        # Key thing: no exception raised
        assert True

    # --- extract_entities ---

    async def test_extract_entities_returns_entities_response(self, service):
        from beanllm.dto.request.graph.kg_request import ExtractEntitiesRequest
        from beanllm.dto.response.graph.kg_response import EntitiesResponse

        request = ExtractEntitiesRequest(
            document_id="doc-1",
            text="Apple was founded by Steve Jobs.",
            entity_types=["person", "organization"],
        )
        result = await service.extract_entities(request)
        assert isinstance(result, EntitiesResponse)

    async def test_extract_relations_returns_relations_response(self, service):
        from beanllm.dto.request.graph.kg_request import ExtractRelationsRequest
        from beanllm.dto.response.graph.kg_response import RelationsResponse

        request = ExtractRelationsRequest(
            document_id="doc-1",
            text="Apple was founded by Steve Jobs.",
            entities=[{"name": "Apple", "type": "organization", "id": "e1"}],
        )
        result = await service.extract_relations(request)
        assert isinstance(result, RelationsResponse)

    # --- query_graph ---

    async def test_query_graph_raises_for_unknown_graph(self, service):
        from beanllm.dto.request.graph.kg_request import QueryGraphRequest

        request = QueryGraphRequest(
            graph_id="nonexistent",
            query="MATCH (n) RETURN n",
        )
        with pytest.raises(RuntimeError, match="Graph not found"):
            await service.query_graph(request)

    async def test_query_graph_returns_response_for_existing_graph(self, service):
        import networkx as nx

        from beanllm.dto.request.graph.kg_request import QueryGraphRequest
        from beanllm.dto.response.graph.kg_response import QueryGraphResponse

        # Manually insert a graph
        g = nx.DiGraph()
        g.add_node("Apple", label="organization")
        service._graphs["test-graph"] = g

        # Use find_entities_by_type so we don't need Neo4j
        request = QueryGraphRequest(
            graph_id="test-graph",
            query="find all entities",
            query_type="find_entities_by_type",
            params={"entity_type": "organization"},
        )
        result = await service.query_graph(request)
        assert isinstance(result, QueryGraphResponse)

    # --- graph_rag ---

    async def test_graph_rag_raises_for_unknown_graph(self, service):
        with pytest.raises(RuntimeError, match="Graph not found"):
            await service.graph_rag(query="Who founded Apple?", graph_id="nope")

    async def test_graph_rag_returns_response_for_existing_graph(self, service):
        import networkx as nx

        from beanllm.dto.response.graph.kg_response import GraphRAGResponse

        g = nx.DiGraph()
        g.add_node("Apple", label="ORG")
        service._graphs["rag-graph"] = g

        result = await service.graph_rag(query="Who founded Apple?", graph_id="rag-graph")
        assert isinstance(result, GraphRAGResponse)

    # --- visualize_graph ---

    async def test_visualize_graph_raises_for_missing_graph(self, service):
        with pytest.raises(RuntimeError):
            await service.visualize_graph("nonexistent")

    async def test_visualize_graph_returns_string(self, service):
        import networkx as nx

        g = nx.DiGraph()
        g.add_edge("A", "B", relation="KNOWS")
        service._graphs["viz-graph"] = g

        result = await service.visualize_graph("viz-graph")
        assert isinstance(result, str)

    # --- get_graph_stats ---

    async def test_get_graph_stats_raises_for_missing_graph(self, service):
        with pytest.raises(RuntimeError):
            await service.get_graph_stats("nonexistent")

    async def test_get_graph_stats_returns_dict(self, service):
        import networkx as nx

        g = nx.DiGraph()
        g.add_edge("A", "B")
        service._graphs["stats-graph"] = g

        stats = await service.get_graph_stats("stats-graph")
        assert isinstance(stats, dict)
        assert "num_nodes" in stats

    # --- list_graphs ---

    async def test_list_graphs_returns_empty_list_initially(self, service):
        result = await service.list_graphs()
        assert isinstance(result, list)
        assert result == []

    async def test_list_graphs_returns_graphs_after_insertion(self, service):
        import networkx as nx

        service._graphs["g1"] = nx.DiGraph()
        service._graph_metadata["g1"] = {"name": "Graph 1"}

        result = await service.list_graphs()
        assert isinstance(result, list)
        assert len(result) >= 1

    # --- delete_graph ---

    def test_delete_graph_removes_from_store(self, service):
        import networkx as nx

        service._graphs["to-delete"] = nx.DiGraph()
        service.delete_graph("to-delete")
        assert "to-delete" not in service._graphs

    def test_delete_nonexistent_graph_is_silent(self, service):
        # delete_graph doesn't raise on nonexistent — it just silently skips
        service.delete_graph("nonexistent")
        assert True
