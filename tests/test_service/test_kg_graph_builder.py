"""Tests for service/impl/advanced/kg_graph_builder.py (build_graph_logic)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import networkx as nx
import pytest

from beanllm.dto.request.graph.kg_request import BuildGraphRequest
from beanllm.service.impl.advanced.kg_graph_builder import build_graph_logic


def _make_graph_builder(entities=None, relations=None, stats=None) -> MagicMock:
    """Create a mock GraphBuilder with configurable return values."""
    gb = MagicMock()
    gb.build_graph.return_value = nx.DiGraph()
    gb.merge_graphs.return_value = nx.DiGraph()
    gb.get_graph_statistics.return_value = stats or {
        "num_nodes": 3,
        "num_edges": 2,
        "num_weakly_connected_components": 1,
    }
    return gb


def _make_request(**kwargs) -> BuildGraphRequest:
    defaults = dict(
        documents=["doc1", "doc2"],
        graph_name="test_graph",
        entity_types=["PERSON"],
        relation_types=["KNOWS"],
        backend="networkx",
        persist_to_neo4j=False,
    )
    defaults.update(kwargs)
    return BuildGraphRequest(**defaults)


async def _process_single_doc(doc: str):
    """Default per-document handler: returns empty entities/relations."""
    return {"entities": [], "relations": []}


# ---------------------------------------------------------------------------
# Basic paths
# ---------------------------------------------------------------------------


class TestBuildGraphLogicBasic:
    async def test_creates_new_graph_when_no_graph_id_in_graphs(self):
        graphs = {}
        graph_metadata = {}
        gb = _make_graph_builder()
        request = _make_request()
        batch_processor = MagicMock()
        batch_processor.process_items = AsyncMock(
            return_value=[
                {"entities": [], "relations": []},
                {"entities": [], "relations": []},
            ]
        )

        response = await build_graph_logic(
            graphs=graphs,
            graph_metadata=graph_metadata,
            graph_builder=gb,
            neo4j_adapter=None,
            batch_processor=batch_processor,
            process_single_doc_fn=_process_single_doc,
            request=request,
        )

        assert response is not None
        assert response.graph_id is not None
        assert response.num_nodes == 3
        assert response.num_edges == 2

    async def test_uses_provided_graph_id(self):
        graphs = {}
        graph_metadata = {}
        gb = _make_graph_builder()
        batch_processor = MagicMock()
        batch_processor.process_items = AsyncMock(return_value=[{}, {}])
        request = _make_request(graph_id="my-graph-001")

        response = await build_graph_logic(
            graphs=graphs,
            graph_metadata=graph_metadata,
            graph_builder=gb,
            neo4j_adapter=None,
            batch_processor=batch_processor,
            process_single_doc_fn=_process_single_doc,
            request=request,
        )

        assert response.graph_id == "my-graph-001"

    async def test_raises_value_error_when_no_documents(self):
        graphs = {}
        graph_metadata = {}
        gb = _make_graph_builder()
        batch_processor = MagicMock()
        request = _make_request(documents=[])

        with pytest.raises(ValueError, match="documents is required"):
            await build_graph_logic(
                graphs=graphs,
                graph_metadata=graph_metadata,
                graph_builder=gb,
                neo4j_adapter=None,
                batch_processor=batch_processor,
                process_single_doc_fn=_process_single_doc,
                request=request,
            )

    async def test_raises_value_error_when_documents_is_none(self):
        graphs = {}
        graph_metadata = {}
        gb = _make_graph_builder()
        batch_processor = MagicMock()
        request = _make_request(documents=None)

        with pytest.raises(ValueError, match="documents is required"):
            await build_graph_logic(
                graphs=graphs,
                graph_metadata=graph_metadata,
                graph_builder=gb,
                neo4j_adapter=None,
                batch_processor=batch_processor,
                process_single_doc_fn=_process_single_doc,
                request=request,
            )


# ---------------------------------------------------------------------------
# Sequential vs batch processing
# ---------------------------------------------------------------------------


class TestBuildGraphLogicProcessing:
    async def test_sequential_processing_for_less_than_5_docs(self):
        """< 5 documents → sequential (not batch)."""
        graphs = {}
        graph_metadata = {}
        gb = _make_graph_builder()
        batch_processor = MagicMock()
        batch_processor.process_items = AsyncMock(return_value=[])

        call_log = []

        async def process_doc(doc: str):
            call_log.append(doc)
            return {"entities": [], "relations": []}

        request = _make_request(documents=["a", "b", "c"])

        await build_graph_logic(
            graphs=graphs,
            graph_metadata=graph_metadata,
            graph_builder=gb,
            neo4j_adapter=None,
            batch_processor=batch_processor,
            process_single_doc_fn=process_doc,
            request=request,
        )

        batch_processor.process_items.assert_not_called()
        assert call_log == ["a", "b", "c"]

    async def test_batch_processing_for_5_or_more_docs(self):
        """>=5 documents → batch processing."""
        graphs = {}
        graph_metadata = {}
        gb = _make_graph_builder()
        batch_processor = MagicMock()
        batch_processor.process_items = AsyncMock(
            return_value=[
                {"entities": [], "relations": []},
            ]
            * 5
        )

        request = _make_request(documents=["d1", "d2", "d3", "d4", "d5"])

        await build_graph_logic(
            graphs=graphs,
            graph_metadata=graph_metadata,
            graph_builder=gb,
            neo4j_adapter=None,
            batch_processor=batch_processor,
            process_single_doc_fn=_process_single_doc,
            request=request,
        )

        batch_processor.process_items.assert_called_once()

    async def test_batch_results_with_error_key_are_skipped(self):
        """Batch results containing 'error' key are skipped."""
        graphs = {}
        graph_metadata = {}
        gb = _make_graph_builder()
        batch_processor = MagicMock()
        batch_processor.process_items = AsyncMock(
            return_value=[
                {"entities": [], "relations": []},
                {"error": "doc processing failed"},
                {"entities": [], "relations": []},
                {"entities": [], "relations": []},
                {"entities": [], "relations": []},
            ]
        )

        request = _make_request(documents=["d1", "d2", "d3", "d4", "d5"])

        # Should not raise — errors are logged and skipped
        response = await build_graph_logic(
            graphs=graphs,
            graph_metadata=graph_metadata,
            graph_builder=gb,
            neo4j_adapter=None,
            batch_processor=batch_processor,
            process_single_doc_fn=_process_single_doc,
            request=request,
        )

        assert response is not None


# ---------------------------------------------------------------------------
# Graph merging
# ---------------------------------------------------------------------------


class TestBuildGraphLogicMerging:
    async def test_merges_with_existing_graph(self):
        existing_graph = nx.DiGraph()
        graph_id = "existing-123"
        graphs = {graph_id: existing_graph}
        graph_metadata = {}
        gb = _make_graph_builder()
        batch_processor = MagicMock()
        batch_processor.process_items = AsyncMock(return_value=[{}, {}, {}, {}, {}])

        request = _make_request(graph_id=graph_id, documents=["d"] * 5)

        await build_graph_logic(
            graphs=graphs,
            graph_metadata=graph_metadata,
            graph_builder=gb,
            neo4j_adapter=None,
            batch_processor=batch_processor,
            process_single_doc_fn=_process_single_doc,
            request=request,
        )

        gb.merge_graphs.assert_called_once()

    async def test_stores_graph_and_metadata(self):
        graphs = {}
        graph_metadata = {}
        gb = _make_graph_builder()
        batch_processor = MagicMock()
        batch_processor.process_items = AsyncMock(return_value=[{}, {}])

        request = _make_request(graph_id="stored-graph", documents=["a", "b"])

        await build_graph_logic(
            graphs=graphs,
            graph_metadata=graph_metadata,
            graph_builder=gb,
            neo4j_adapter=None,
            batch_processor=batch_processor,
            process_single_doc_fn=_process_single_doc,
            request=request,
        )

        assert "stored-graph" in graphs
        assert "stored-graph" in graph_metadata
        assert graph_metadata["stored-graph"]["num_documents"] == 2


# ---------------------------------------------------------------------------
# Neo4j integration
# ---------------------------------------------------------------------------


class TestBuildGraphLogicNeo4j:
    async def test_exports_to_neo4j_when_flag_set(self):
        graphs = {}
        graph_metadata = {}
        gb = _make_graph_builder()
        neo4j = MagicMock()
        batch_processor = MagicMock()
        batch_processor.process_items = AsyncMock(return_value=[{}, {}])

        request = _make_request(documents=["a", "b"], persist_to_neo4j=True)

        await build_graph_logic(
            graphs=graphs,
            graph_metadata=graph_metadata,
            graph_builder=gb,
            neo4j_adapter=neo4j,
            batch_processor=batch_processor,
            process_single_doc_fn=_process_single_doc,
            request=request,
        )

        neo4j.export_graph.assert_called_once()

    async def test_skips_neo4j_when_adapter_is_none(self):
        graphs = {}
        graph_metadata = {}
        gb = _make_graph_builder()
        batch_processor = MagicMock()
        batch_processor.process_items = AsyncMock(return_value=[{}, {}])

        request = _make_request(documents=["a", "b"], persist_to_neo4j=True)

        # No neo4j_adapter — should not crash
        response = await build_graph_logic(
            graphs=graphs,
            graph_metadata=graph_metadata,
            graph_builder=gb,
            neo4j_adapter=None,
            batch_processor=batch_processor,
            process_single_doc_fn=_process_single_doc,
            request=request,
        )

        assert response is not None

    async def test_neo4j_export_failure_is_logged_not_raised(self):
        graphs = {}
        graph_metadata = {}
        gb = _make_graph_builder()
        neo4j = MagicMock()
        neo4j.export_graph.side_effect = RuntimeError("Neo4j connection failed")
        batch_processor = MagicMock()
        batch_processor.process_items = AsyncMock(return_value=[{}, {}])

        request = _make_request(documents=["a", "b"], persist_to_neo4j=True)

        # Should not raise despite Neo4j error
        response = await build_graph_logic(
            graphs=graphs,
            graph_metadata=graph_metadata,
            graph_builder=gb,
            neo4j_adapter=neo4j,
            batch_processor=batch_processor,
            process_single_doc_fn=_process_single_doc,
            request=request,
        )

        assert response is not None


# ---------------------------------------------------------------------------
# Statistics handling
# ---------------------------------------------------------------------------


class TestBuildGraphLogicStats:
    async def test_adds_num_connected_components_from_weakly_connected(self):
        graphs = {}
        graph_metadata = {}
        stats = {
            "num_nodes": 5,
            "num_edges": 4,
            "num_weakly_connected_components": 2,
        }
        gb = _make_graph_builder(stats=stats)
        batch_processor = MagicMock()
        batch_processor.process_items = AsyncMock(return_value=[{}, {}])

        request = _make_request(documents=["a", "b"])

        response = await build_graph_logic(
            graphs=graphs,
            graph_metadata=graph_metadata,
            graph_builder=gb,
            neo4j_adapter=None,
            batch_processor=batch_processor,
            process_single_doc_fn=_process_single_doc,
            request=request,
        )

        assert "num_connected_components" in response.statistics
        assert response.statistics["num_connected_components"] == 2

    async def test_stats_already_has_num_connected_components_not_overwritten(self):
        graphs = {}
        graph_metadata = {}
        stats = {
            "num_nodes": 5,
            "num_edges": 4,
            "num_connected_components": 3,  # Already present
            "num_weakly_connected_components": 1,
        }
        gb = _make_graph_builder(stats=stats)
        batch_processor = MagicMock()
        batch_processor.process_items = AsyncMock(return_value=[{}, {}])

        request = _make_request(documents=["a", "b"])

        response = await build_graph_logic(
            graphs=graphs,
            graph_metadata=graph_metadata,
            graph_builder=gb,
            neo4j_adapter=None,
            batch_processor=batch_processor,
            process_single_doc_fn=_process_single_doc,
            request=request,
        )

        # Should keep existing value (3), not overwrite with 1
        assert response.statistics["num_connected_components"] == 3

    async def test_response_graph_name_falls_back_to_graph_id(self):
        graphs = {}
        graph_metadata = {}
        gb = _make_graph_builder()
        batch_processor = MagicMock()
        batch_processor.process_items = AsyncMock(return_value=[{}, {}])

        request = _make_request(graph_id="fallback-id", graph_name=None, documents=["a", "b"])

        response = await build_graph_logic(
            graphs=graphs,
            graph_metadata=graph_metadata,
            graph_builder=gb,
            neo4j_adapter=None,
            batch_processor=batch_processor,
            process_single_doc_fn=_process_single_doc,
            request=request,
        )

        assert response.graph_name == "fallback-id"
