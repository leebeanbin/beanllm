"""
Knowledge Graph Domain 테스트 - 엔티티 모델, 도메인 타입
"""

import pytest

from beanllm.domain.knowledge_graph.entity_models import Entity, EntityType


class TestEntityType:
    def test_entity_type_values(self) -> None:
        assert EntityType.PERSON.value == "person"
        assert EntityType.ORGANIZATION.value == "organization"
        assert EntityType.LOCATION.value == "location"
        assert EntityType.CONCEPT.value == "concept"
        assert EntityType.EVENT.value == "event"
        assert EntityType.DATE.value == "date"
        assert EntityType.PRODUCT.value == "product"
        assert EntityType.TECHNOLOGY.value == "technology"
        assert EntityType.OTHER.value == "other"

    def test_entity_type_from_value(self) -> None:
        et = EntityType("person")
        assert et == EntityType.PERSON

    def test_entity_type_iteration(self) -> None:
        types = list(EntityType)
        assert len(types) >= 9


class TestEntity:
    def test_create_entity_minimal(self) -> None:
        entity = Entity(id="e1", name="Apple", type=EntityType.ORGANIZATION)
        assert entity.id == "e1"
        assert entity.name == "Apple"
        assert entity.type == EntityType.ORGANIZATION
        assert entity.description == ""
        assert entity.aliases == []
        assert entity.properties == {}

    def test_create_entity_full(self) -> None:
        entity = Entity(
            id="e1",
            name="Steve Jobs",
            type=EntityType.PERSON,
            description="Co-founder of Apple",
            aliases=["Jobs", "Steven Paul Jobs"],
            properties={"born": 1955, "nationality": "American"},
            confidence=0.95,
        )
        assert entity.name == "Steve Jobs"
        assert len(entity.aliases) == 2
        assert entity.properties["born"] == 1955
        assert entity.confidence == 0.95

    def test_entity_default_confidence(self) -> None:
        entity = Entity(id="e1", name="Apple", type=EntityType.ORGANIZATION)
        assert entity.confidence >= 0.0

    def test_entity_with_location_type(self) -> None:
        entity = Entity(
            id="loc1",
            name="Cupertino",
            type=EntityType.LOCATION,
            description="City in California",
        )
        assert entity.type == EntityType.LOCATION

    def test_entity_properties_mutable(self) -> None:
        entity = Entity(id="e1", name="Test", type=EntityType.OTHER)
        entity.properties["key"] = "value"
        assert entity.properties["key"] == "value"

    def test_entity_aliases_mutable(self) -> None:
        entity = Entity(id="e1", name="Steve Jobs", type=EntityType.PERSON)
        entity.aliases.append("Jobs")
        assert "Jobs" in entity.aliases

    def test_different_entities_have_independent_defaults(self) -> None:
        e1 = Entity(id="e1", name="A", type=EntityType.PERSON)
        e2 = Entity(id="e2", name="B", type=EntityType.PERSON)
        e1.aliases.append("alias")
        assert "alias" not in e2.aliases

    def test_entity_technology_type(self) -> None:
        entity = Entity(id="t1", name="Python", type=EntityType.TECHNOLOGY)
        assert entity.type == EntityType.TECHNOLOGY


class TestEntityNERModels:
    """NER models and patterns"""

    def test_ner_entity_import(self) -> None:
        from beanllm.domain.knowledge_graph.ner_models import NEREntity

        entity = NEREntity(text="Apple", label="ORG", start=0, end=5, confidence=0.99)
        assert entity.text == "Apple"
        assert entity.label == "ORG"
        assert entity.confidence == 0.99

    def test_ner_result_import(self) -> None:
        from beanllm.domain.knowledge_graph.ner_models import NEREntity, NERResult

        entity = NEREntity(text="Apple", label="ORG", start=0, end=5)
        result = NERResult(
            entities=[entity],
            engine_name="llm",
            latency_ms=120.5,
        )
        assert len(result.entities) == 1
        assert result.engine_name == "llm"

    def test_benchmark_sample(self) -> None:
        from beanllm.domain.knowledge_graph.ner_models import BenchmarkSample

        sample = BenchmarkSample(
            text="Apple was founded by Steve Jobs.",
            entities=[{"text": "Apple", "label": "ORG", "start": 0, "end": 5}],
        )
        assert sample.text == "Apple was founded by Steve Jobs."
        assert len(sample.entities) == 1

    def test_benchmark_result(self) -> None:
        from beanllm.domain.knowledge_graph.ner_models import BenchmarkResult

        result = BenchmarkResult(
            engine_name="test",
            precision=0.95,
            recall=0.88,
            f1_score=0.91,
            avg_latency_ms=50.0,
            total_samples=100,
        )
        assert result.precision == 0.95
        assert result.f1_score == 0.91
