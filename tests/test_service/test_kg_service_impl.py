"""
Knowledge Graph Service Impl 테스트 - kg_serialization, kg_entity_extraction,
kg_graph_operations, kg_graph_query, kg_graph_builder, kg_graph_rag
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import networkx as nx
import pytest

from beanllm.domain.knowledge_graph import Entity, EntityType, Relation, RelationType
from beanllm.dto.request.graph.kg_request import (
    ExtractEntitiesRequest,
    ExtractRelationsRequest,
    QueryGraphRequest,
)
from beanllm.service.impl.advanced.kg_entity_extraction import (
    extract_entities_logic,
    extract_relations_logic,
)
from beanllm.service.impl.advanced.kg_graph_operations import (
    _get_graph,
    delete_graph,
    get_graph_stats,
    list_graphs,
    visualize_graph,
)
from beanllm.service.impl.advanced.kg_graph_query import execute_graph_query
from beanllm.service.impl.advanced.kg_serialization import (
    count_by_type,
    serialize_entity,
    serialize_relation,
)

# ─── Helpers ──────────────────────────────────────────────────────────────────


def make_entity(name: str = "Alice", etype: EntityType = EntityType.PERSON) -> Entity:
    return Entity(id=str(uuid.uuid4()), name=name, type=etype)


def make_relation(src: str | None = None, tgt: str | None = None) -> Relation:
    return Relation(
        source_id=src or str(uuid.uuid4()),
        target_id=tgt or str(uuid.uuid4()),
        type=RelationType.RELATED_TO,
    )


def make_service_stub(graphs: dict | None = None) -> MagicMock:
    """Minimal service object with _graphs/_graph_metadata/_graph_builder."""
    svc = MagicMock()
    svc._graphs = graphs or {}
    svc._graph_metadata = {}
    builder = MagicMock()
    builder.get_graph_statistics.return_value = {"num_nodes": 0, "num_edges": 0, "density": 0.0}
    svc._graph_builder = builder
    return svc


def make_nx_graph(entities: list[Entity] | None = None) -> nx.DiGraph:
    g = nx.DiGraph()
    for e in entities or []:
        g.add_node(e.id, name=e.name, type=e.type.value)
    return g


# ─── Serialization ────────────────────────────────────────────────────────────


class TestSerializeEntity:
    def test_all_fields_present(self) -> None:
        e = make_entity()
        d = serialize_entity(e)
        for key in (
            "id",
            "name",
            "type",
            "description",
            "properties",
            "aliases",
            "confidence",
            "mentions",
        ):
            assert key in d

    def test_type_is_string_value(self) -> None:
        e = make_entity(etype=EntityType.ORGANIZATION)
        d = serialize_entity(e)
        assert isinstance(d["type"], str)
        assert d["type"] == EntityType.ORGANIZATION.value

    def test_name_preserved(self) -> None:
        e = make_entity(name="Bob")
        assert serialize_entity(e)["name"] == "Bob"

    def test_confidence_preserved(self) -> None:
        e = Entity(id="x", name="X", type=EntityType.PERSON, confidence=0.75)
        assert serialize_entity(e)["confidence"] == 0.75


class TestSerializeRelation:
    def test_all_fields_present(self) -> None:
        r = make_relation()
        d = serialize_relation(r)
        for key in ("source_id", "target_id", "type", "properties", "confidence", "bidirectional"):
            assert key in d

    def test_type_is_string(self) -> None:
        r = make_relation()
        assert isinstance(serialize_relation(r)["type"], str)


class TestCountByType:
    def test_empty_list(self) -> None:
        assert count_by_type([]) == {}

    def test_single_entity(self) -> None:
        e = make_entity(etype=EntityType.PERSON)
        result = count_by_type([e])
        assert result.get(EntityType.PERSON.value, 0) == 1

    def test_mixed_types(self) -> None:
        entities = [
            make_entity(etype=EntityType.PERSON),
            make_entity(etype=EntityType.PERSON),
            make_entity(etype=EntityType.ORGANIZATION),
        ]
        result = count_by_type(entities)
        assert result[EntityType.PERSON.value] == 2
        assert result[EntityType.ORGANIZATION.value] == 1

    def test_relations(self) -> None:
        r1 = make_relation()
        r2 = make_relation()
        result = count_by_type([r1, r2])
        assert result.get(RelationType.RELATED_TO.value, 0) == 2


# ─── Entity Extraction Logic ──────────────────────────────────────────────────


class TestExtractEntitiesLogic:
    def test_empty_text_raises(self) -> None:
        extractor = MagicMock()
        # entity_types=[] to skip enum conversion and reach text validation
        req = ExtractEntitiesRequest(document_id="d1", text="", entity_types=[])
        with pytest.raises(ValueError, match="text is required"):
            extract_entities_logic(extractor, req)

    def test_none_text_raises(self) -> None:
        extractor = MagicMock()
        req = ExtractEntitiesRequest(document_id="d1", text=None, entity_types=[])
        with pytest.raises(ValueError):
            extract_entities_logic(extractor, req)

    def test_returns_entities_response(self) -> None:
        entities = [make_entity("Alice"), make_entity("Bob")]
        extractor = MagicMock()
        extractor.extract_entities.return_value = entities
        req = ExtractEntitiesRequest(
            document_id="d1",
            text="Alice and Bob are here",
            entity_types=["person"],
            resolve_coreferences=False,
        )
        resp = extract_entities_logic(extractor, req)
        assert resp.num_entities == 2
        assert len(resp.entities) == 2

    def test_entity_types_filter_passed(self) -> None:
        extractor = MagicMock()
        extractor.extract_entities.return_value = []
        req = ExtractEntitiesRequest(
            document_id="d1",
            text="some text",
            entity_types=["person"],
            resolve_coreferences=False,
        )
        extract_entities_logic(extractor, req)
        call_kwargs = extractor.extract_entities.call_args
        assert call_kwargs is not None

    def test_coreference_resolution_called(self) -> None:
        entity = make_entity()
        extractor = MagicMock()
        extractor.extract_entities.return_value = [entity]
        extractor.resolve_coreferences.return_value = [entity]
        req = ExtractEntitiesRequest(
            document_id="d1",
            text="He said he was here",
            entity_types=["person"],
            resolve_coreferences=True,
        )
        extract_entities_logic(extractor, req)
        extractor.resolve_coreferences.assert_called_once()

    def test_entity_counts_by_type(self) -> None:
        entities = [
            make_entity(etype=EntityType.PERSON),
            make_entity(etype=EntityType.ORGANIZATION),
        ]
        extractor = MagicMock()
        extractor.extract_entities.return_value = entities
        req = ExtractEntitiesRequest(
            document_id="d1", text="some text", entity_types=[], resolve_coreferences=False
        )
        resp = extract_entities_logic(extractor, req)
        assert EntityType.PERSON.value in resp.entity_counts_by_type


class TestExtractRelationsLogic:
    def test_empty_entities_raises(self) -> None:
        extractor = MagicMock()
        req = ExtractRelationsRequest(document_id="d1", text="text", entities=[])
        with pytest.raises(ValueError, match="entities is required"):
            extract_relations_logic(extractor, req)

    def test_empty_text_raises(self) -> None:
        extractor = MagicMock()
        entity_dicts = [{"id": "e1", "name": "Alice", "type": "person"}]
        req = ExtractRelationsRequest(document_id="d1", text="", entities=entity_dicts)
        with pytest.raises(ValueError, match="text is required"):
            extract_relations_logic(extractor, req)

    def test_returns_relations_response(self) -> None:
        src = str(uuid.uuid4())
        tgt = str(uuid.uuid4())
        rel = make_relation(src=src, tgt=tgt)
        extractor = MagicMock()
        extractor.extract_relations.return_value = [rel]
        entity_dicts = [{"id": src, "name": "Alice", "type": "person"}]
        req = ExtractRelationsRequest(
            document_id="d1",
            text="Alice knows Bob",
            entities=entity_dicts,
            infer_implicit=False,
        )
        resp = extract_relations_logic(extractor, req)
        assert resp.num_relations == 1

    def test_implicit_relations_inferred(self) -> None:
        rel = make_relation()
        implicit_rel = make_relation()
        extractor = MagicMock()
        extractor.extract_relations.return_value = [rel]
        extractor.infer_implicit_relations.return_value = [implicit_rel]
        entity_dicts = [{"id": "e1", "name": "Alice", "type": "person"}]
        req = ExtractRelationsRequest(
            document_id="d1",
            text="Alice knows Bob",
            entities=entity_dicts,
            infer_implicit=True,
        )
        resp = extract_relations_logic(extractor, req)
        extractor.infer_implicit_relations.assert_called_once()
        assert resp.num_relations == 2


# ─── Graph Operations ─────────────────────────────────────────────────────────


class TestGetGraph:
    def test_graph_found(self) -> None:
        g = nx.DiGraph()
        svc = make_service_stub({"g1": g})
        assert _get_graph(svc, "g1") is g

    def test_graph_not_found_raises(self) -> None:
        svc = make_service_stub({})
        with pytest.raises(ValueError, match="Graph not found"):
            _get_graph(svc, "missing")


class TestVisualizeGraph:
    @pytest.mark.asyncio
    async def test_returns_string(self) -> None:
        alice = make_entity("Alice")
        bob = make_entity("Bob")
        g = make_nx_graph([alice, bob])
        g.add_edge(alice.id, bob.id, type="knows")
        svc = make_service_stub({"g1": g})
        result = await visualize_graph(svc, "g1")
        assert isinstance(result, str)
        assert "g1" in result

    @pytest.mark.asyncio
    async def test_contains_nodes_and_edges_headers(self) -> None:
        g = nx.DiGraph()
        svc = make_service_stub({"g1": g})
        result = await visualize_graph(svc, "g1")
        assert "Nodes:" in result or "Entities:" in result

    @pytest.mark.asyncio
    async def test_not_found_raises(self) -> None:
        svc = make_service_stub({})
        with pytest.raises(ValueError):
            await visualize_graph(svc, "missing")

    @pytest.mark.asyncio
    async def test_many_nodes_truncated(self) -> None:
        g = nx.DiGraph()
        for i in range(25):
            g.add_node(str(i), name=f"Node{i}", type="PERSON")
        svc = make_service_stub({"g1": g})
        result = await visualize_graph(svc, "g1")
        assert "more" in result


class TestGetGraphStats:
    @pytest.mark.asyncio
    async def test_returns_dict_with_graph_id(self) -> None:
        g = nx.DiGraph()
        svc = make_service_stub({"g1": g})
        stats = await get_graph_stats(svc, "g1")
        assert isinstance(stats, dict)
        assert stats["graph_id"] == "g1"

    @pytest.mark.asyncio
    async def test_entity_type_counts(self) -> None:
        alice = make_entity("Alice", EntityType.PERSON)
        g = make_nx_graph([alice])
        svc = make_service_stub({"g1": g})
        stats = await get_graph_stats(svc, "g1")
        assert "entity_type_counts" in stats
        assert EntityType.PERSON.value in stats["entity_type_counts"]

    @pytest.mark.asyncio
    async def test_relation_type_counts(self) -> None:
        alice = make_entity("Alice")
        bob = make_entity("Bob")
        g = make_nx_graph([alice, bob])
        g.add_edge(alice.id, bob.id, type="knows")
        svc = make_service_stub({"g1": g})
        stats = await get_graph_stats(svc, "g1")
        assert "relation_type_counts" in stats
        assert "knows" in stats["relation_type_counts"]


class TestListGraphs:
    @pytest.mark.asyncio
    async def test_empty_returns_empty_list(self) -> None:
        svc = make_service_stub({})
        result = await list_graphs(svc)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_each_graph(self) -> None:
        g1, g2 = nx.DiGraph(), nx.DiGraph()
        svc = make_service_stub({"g1": g1, "g2": g2})
        result = await list_graphs(svc)
        assert len(result) == 2
        ids = [r["id"] for r in result]
        assert "g1" in ids
        assert "g2" in ids

    @pytest.mark.asyncio
    async def test_result_contains_node_edge_counts(self) -> None:
        g = nx.DiGraph()
        g.add_node("n1")
        svc = make_service_stub({"g1": g})
        result = await list_graphs(svc)
        assert result[0]["num_nodes"] == 1
        assert result[0]["num_edges"] == 0


class TestDeleteGraph:
    def test_delete_existing(self) -> None:
        g = nx.DiGraph()
        svc = make_service_stub({"g1": g})
        delete_graph(svc, "g1")
        assert "g1" not in svc._graphs

    def test_delete_nonexistent_no_error(self) -> None:
        svc = make_service_stub({})
        delete_graph(svc, "missing")  # should not raise

    def test_metadata_also_deleted(self) -> None:
        g = nx.DiGraph()
        svc = make_service_stub({"g1": g})
        svc._graph_metadata["g1"] = {"name": "Test"}
        delete_graph(svc, "g1")
        assert "g1" not in svc._graph_metadata


# ─── Graph Query ──────────────────────────────────────────────────────────────


class TestExecuteGraphQuery:
    def _make_query_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        alice = make_entity("Alice", EntityType.PERSON)
        bob = make_entity("Bob", EntityType.PERSON)
        apple = make_entity("Apple", EntityType.ORGANIZATION)
        g.add_node(alice.id, name="Alice", type=EntityType.PERSON.value)
        g.add_node(bob.id, name="Bob", type=EntityType.PERSON.value)
        g.add_node(apple.id, name="Apple", type=EntityType.ORGANIZATION.value)
        g.add_edge(alice.id, bob.id, type="knows")
        self._alice_id = alice.id
        self._bob_id = bob.id
        return g

    def test_find_entities_by_type(self) -> None:
        g = self._make_query_graph()
        req = QueryGraphRequest(
            graph_id="g1",
            query="find persons",
            query_type="find_entities_by_type",
            params={"entity_type": EntityType.PERSON.value},
        )
        resp = execute_graph_query(graph=g, neo4j_adapter=None, request=req)
        assert resp is not None

    def test_find_entities_by_name(self) -> None:
        g = self._make_query_graph()
        req = QueryGraphRequest(
            graph_id="g1",
            query="Alice",
            query_type="find_entities_by_name",
            params={"name": "Alice", "fuzzy": False},
        )
        resp = execute_graph_query(graph=g, neo4j_adapter=None, request=req)
        assert resp is not None

    def test_find_related_entities(self) -> None:
        g = self._make_query_graph()
        req = QueryGraphRequest(
            graph_id="g1",
            query="related to Alice",
            query_type="find_related_entities",
            params={"entity_id": self._alice_id, "max_hops": 1},
        )
        resp = execute_graph_query(graph=g, neo4j_adapter=None, request=req)
        assert resp is not None

    def test_find_shortest_path(self) -> None:
        g = self._make_query_graph()
        req = QueryGraphRequest(
            graph_id="g1",
            query="path",
            query_type="find_shortest_path",
            params={"source_id": self._alice_id, "target_id": self._bob_id},
        )
        resp = execute_graph_query(graph=g, neo4j_adapter=None, request=req)
        assert resp is not None

    def test_get_entity_details(self) -> None:
        g = self._make_query_graph()
        req = QueryGraphRequest(
            graph_id="g1",
            query="details",
            query_type="get_entity_details",
            params={"entity_id": self._alice_id},
        )
        resp = execute_graph_query(graph=g, neo4j_adapter=None, request=req)
        assert resp is not None

    def test_cypher_without_neo4j_raises(self) -> None:
        g = self._make_query_graph()
        req = QueryGraphRequest(
            graph_id="g1",
            query="MATCH (n) RETURN n",
            query_type="cypher",
        )
        with pytest.raises(ValueError, match="Neo4j"):
            execute_graph_query(graph=g, neo4j_adapter=None, request=req)

    def test_unknown_query_type_raises(self) -> None:
        g = self._make_query_graph()
        req = QueryGraphRequest(
            graph_id="g1",
            query="?",
            query_type="nonexistent_type",
        )
        with pytest.raises(ValueError):
            execute_graph_query(graph=g, neo4j_adapter=None, request=req)
