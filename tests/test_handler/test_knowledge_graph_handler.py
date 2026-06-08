"""
KnowledgeGraphHandler 테스트
"""

from unittest.mock import AsyncMock

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
from beanllm.handler.advanced.knowledge_graph_handler import KnowledgeGraphHandler
from beanllm.service.knowledge_graph_service import IKnowledgeGraphService


def _make_entities_response(document_id: str = "doc-1") -> EntitiesResponse:
    return EntitiesResponse(
        document_id=document_id,
        entities=[{"name": "Apple", "type": "organization", "id": "e1"}],
        num_entities=1,
        entity_counts_by_type={"organization": 1},
    )


def _make_relations_response(document_id: str = "doc-1") -> RelationsResponse:
    return RelationsResponse(
        document_id=document_id,
        relations=[{"source_id": "e1", "target_id": "e2", "type": "founded"}],
        num_relations=1,
        relation_counts_by_type={"founded": 1},
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
        statistics={"density": 0.5},
    )


def _make_query_response(graph_id: str = "graph-1") -> QueryGraphResponse:
    return QueryGraphResponse(
        graph_id=graph_id,
        query="test",
        results=[{"name": "Apple"}],
        num_results=1,
        execution_time=0.01,
    )


def _make_rag_response() -> GraphRAGResponse:
    return GraphRAGResponse(
        answer="Apple was founded by Steve Jobs.",
        entities_used=["Apple", "Steve Jobs"],
        reasoning_paths=[["Apple", "founded_by", "Steve Jobs"]],
        graph_context="Apple organization...",
    )


class TestKnowledgeGraphHandler:
    @pytest.fixture
    def mock_service(self) -> AsyncMock:
        service = AsyncMock(spec=IKnowledgeGraphService)
        service.extract_entities.return_value = _make_entities_response()
        service.extract_relations.return_value = _make_relations_response()
        service.build_graph.return_value = _make_build_response()
        service.query_graph.return_value = _make_query_response()
        service.graph_rag.return_value = _make_rag_response()
        service.visualize_graph.return_value = "[Apple] --founded_by--> [Steve Jobs]"
        service.get_graph_stats.return_value = {"num_nodes": 3, "num_edges": 2}
        return service

    @pytest.fixture
    def handler(self, mock_service: AsyncMock) -> KnowledgeGraphHandler:
        return KnowledgeGraphHandler(service=mock_service)

    async def test_handle_extract_entities(self, handler: KnowledgeGraphHandler) -> None:
        request = ExtractEntitiesRequest(
            document_id="doc-1",
            text="Apple was founded by Steve Jobs.",
            entity_types=[],  # empty = no filter validation
        )
        result = await handler.handle_extract_entities(request)
        assert isinstance(result, EntitiesResponse)
        assert result.num_entities == 1

    async def test_handle_extract_entities_empty_text_raises(
        self, handler: KnowledgeGraphHandler
    ) -> None:
        request = ExtractEntitiesRequest(document_id="doc-1", text="   ")
        with pytest.raises(Exception):
            await handler.handle_extract_entities(request)

    async def test_handle_extract_entities_none_text_raises(
        self, handler: KnowledgeGraphHandler
    ) -> None:
        request = ExtractEntitiesRequest(document_id="doc-1", text=None)
        with pytest.raises(Exception):
            await handler.handle_extract_entities(request)

    async def test_handle_extract_entities_invalid_type_raises(
        self, handler: KnowledgeGraphHandler
    ) -> None:
        request = ExtractEntitiesRequest(
            document_id="doc-1",
            text="Apple was founded.",
            entity_types=["invalid_type_xyz"],
        )
        with pytest.raises(Exception):
            await handler.handle_extract_entities(request)

    async def test_handle_extract_relations(self, handler: KnowledgeGraphHandler) -> None:
        request = ExtractRelationsRequest(
            document_id="doc-1",
            text="Apple was founded by Steve Jobs.",
            entities=[
                {"name": "Apple", "type": "organization"},
                {"name": "Steve Jobs", "type": "person"},
            ],
        )
        result = await handler.handle_extract_relations(request)
        assert isinstance(result, RelationsResponse)
        assert result.num_relations == 1

    async def test_handle_extract_relations_empty_text_raises(
        self, handler: KnowledgeGraphHandler
    ) -> None:
        request = ExtractRelationsRequest(
            document_id="doc-1",
            text="",
            entities=[{"name": "Apple", "type": "organization"}],
        )
        with pytest.raises(Exception):
            await handler.handle_extract_relations(request)

    async def test_handle_extract_relations_no_entities_raises(
        self, handler: KnowledgeGraphHandler
    ) -> None:
        request = ExtractRelationsRequest(
            document_id="doc-1",
            text="Apple was founded by Steve Jobs.",
            entities=[],
        )
        with pytest.raises(Exception):
            await handler.handle_extract_relations(request)

    async def test_handle_extract_relations_entity_missing_name_raises(
        self, handler: KnowledgeGraphHandler
    ) -> None:
        request = ExtractRelationsRequest(
            document_id="doc-1",
            text="Apple was founded by Steve Jobs.",
            entities=[{"type": "organization"}],  # missing name
        )
        with pytest.raises(Exception):
            await handler.handle_extract_relations(request)

    async def test_handle_build_graph(self, handler: KnowledgeGraphHandler) -> None:
        request = BuildGraphRequest(documents=["Apple was founded by Steve Jobs."])
        result = await handler.handle_build_graph(request)
        assert isinstance(result, BuildGraphResponse)
        assert result.num_nodes == 3

    async def test_handle_build_graph_no_documents_raises(
        self, handler: KnowledgeGraphHandler
    ) -> None:
        request = BuildGraphRequest(documents=[])
        with pytest.raises(Exception):
            await handler.handle_build_graph(request)

    async def test_handle_build_graph_empty_doc_raises(
        self, handler: KnowledgeGraphHandler
    ) -> None:
        request = BuildGraphRequest(documents=["  "])
        with pytest.raises(Exception):
            await handler.handle_build_graph(request)

    async def test_handle_query_graph_find_by_type(self, handler: KnowledgeGraphHandler) -> None:
        request = QueryGraphRequest(
            graph_id="graph-1",
            query="",
            query_type="find_entities_by_type",
            params={"entity_type": "organization"},
        )
        result = await handler.handle_query_graph(request)
        assert isinstance(result, QueryGraphResponse)
        assert result.num_results == 1

    async def test_handle_query_graph_cypher(self, handler: KnowledgeGraphHandler) -> None:
        request = QueryGraphRequest(
            graph_id="graph-1",
            query="MATCH (n) RETURN n",
            query_type="cypher",
        )
        result = await handler.handle_query_graph(request)
        assert isinstance(result, QueryGraphResponse)

    async def test_handle_query_graph_no_graph_id_raises(
        self, handler: KnowledgeGraphHandler
    ) -> None:
        request = QueryGraphRequest(graph_id="", query="MATCH (n) RETURN n", query_type="cypher")
        with pytest.raises(Exception):
            await handler.handle_query_graph(request)

    async def test_handle_query_graph_invalid_type_raises(
        self, handler: KnowledgeGraphHandler
    ) -> None:
        request = QueryGraphRequest(
            graph_id="graph-1",
            query="",
            query_type="invalid_query_type",
        )
        with pytest.raises(Exception):
            await handler.handle_query_graph(request)

    async def test_handle_query_graph_cypher_no_query_raises(
        self, handler: KnowledgeGraphHandler
    ) -> None:
        request = QueryGraphRequest(
            graph_id="graph-1",
            query="",
            query_type="cypher",
        )
        with pytest.raises(Exception):
            await handler.handle_query_graph(request)

    async def test_handle_graph_rag(self, handler: KnowledgeGraphHandler) -> None:
        result = await handler.handle_graph_rag(query="Who founded Apple?", graph_id="graph-1")
        assert isinstance(result, GraphRAGResponse)
        assert "Apple" in result.answer

    async def test_handle_graph_rag_empty_query_raises(
        self, handler: KnowledgeGraphHandler
    ) -> None:
        with pytest.raises(Exception):
            await handler.handle_graph_rag(query="  ", graph_id="graph-1")

    async def test_handle_graph_rag_no_graph_id_raises(
        self, handler: KnowledgeGraphHandler
    ) -> None:
        with pytest.raises(Exception):
            await handler.handle_graph_rag(query="Who founded Apple?", graph_id="")

    async def test_handle_visualize_graph(self, handler: KnowledgeGraphHandler) -> None:
        result = await handler.handle_visualize_graph("graph-1")
        assert isinstance(result, str)
        assert len(result) > 0

    async def test_handle_visualize_graph_no_id_raises(
        self, handler: KnowledgeGraphHandler
    ) -> None:
        with pytest.raises(Exception):
            await handler.handle_visualize_graph("")

    async def test_handle_get_graph_stats(self, handler: KnowledgeGraphHandler) -> None:
        result = await handler.handle_get_graph_stats("graph-1")
        assert isinstance(result, dict)
        assert "num_nodes" in result

    async def test_handle_get_graph_stats_no_id_raises(
        self, handler: KnowledgeGraphHandler
    ) -> None:
        with pytest.raises(Exception):
            await handler.handle_get_graph_stats("")
