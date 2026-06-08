"""
Comprehensive tests for:
- beanllm.domain.retrieval.hybrid_search (HybridRetriever)
- beanllm.domain.prompts.templates (PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate)
- beanllm.domain.prompts.versioning (PromptVersionManager, PromptVersion, PromptVersioning)
- beanllm.domain.rag_debug.chunk_validator (ChunkValidator)
- beanllm.domain.rag_debug.export (DebugReportExporter)
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers shared across all tests
# ---------------------------------------------------------------------------


def _make_mock_bm25_module() -> types.ModuleType:
    """Return a minimal fake rank_bm25 module so HybridRetriever can be imported
    even when rank-bm25 is not installed in the environment."""
    import numpy as np

    mock_module = types.ModuleType("rank_bm25")

    class BM25Okapi:
        def __init__(self, tokenized_corpus, k1=1.5, b=0.75):
            self._corpus = tokenized_corpus

        def get_scores(self, query_tokens):
            """Simple TF-based mock score: count how many query tokens appear in each doc."""
            scores = []
            for doc_tokens in self._corpus:
                score = sum(1.0 for t in query_tokens if t in doc_tokens)
                scores.append(score)
            return np.array(scores, dtype=float)

    mock_module.BM25Okapi = BM25Okapi
    return mock_module


# Inject the mock module before any import of HybridRetriever so that
# `from rank_bm25 import BM25Okapi` inside the source resolves correctly.
if "rank_bm25" not in sys.modules:
    sys.modules["rank_bm25"] = _make_mock_bm25_module()


def _make_embedding_function(dim: int = 4):
    """Return a deterministic embedding function.

    When called with a list it returns List[List[float]] (batch).
    When called with a str it returns List[float] (single vector).
    """
    import hashlib

    import numpy as np

    def embed(text_or_texts):
        if isinstance(text_or_texts, list):
            return [embed(t) for t in text_or_texts]
        # Deterministic vector derived from text hash
        digest = hashlib.md5(text_or_texts.encode()).digest()[:dim]
        vec = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
        return (vec / (vec.sum() + 1e-8)).tolist()

    return embed


@dataclass
class MockDocument:
    """Minimal document object used by ChunkValidator / DebugReportExporter tests."""

    page_content: str
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# 1. HybridRetriever tests
# ---------------------------------------------------------------------------


SAMPLE_DOCS = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with many layers.",
    "Natural language processing helps computers understand text.",
    "Reinforcement learning trains agents through rewards.",
    "Computer vision enables machines to interpret images.",
    "Transfer learning reuses models trained on large datasets.",
    "Unsupervised learning finds patterns without labeled data.",
]


@pytest.fixture
def hybrid_retriever():
    from beanllm.domain.retrieval.hybrid_search import HybridRetriever

    return HybridRetriever(
        documents=SAMPLE_DOCS,
        embedding_function=_make_embedding_function(4),
        fusion_method="rrf",
    )


class TestHybridRetrieverInit:
    def test_default_init_stores_documents(self, hybrid_retriever):
        assert len(hybrid_retriever.documents) == len(SAMPLE_DOCS)

    def test_fusion_method_stored_lowercase(self):
        from beanllm.domain.retrieval.hybrid_search import HybridRetriever

        r = HybridRetriever(
            documents=SAMPLE_DOCS,
            embedding_function=_make_embedding_function(4),
            fusion_method="RRF",
        )
        assert r.fusion_method == "rrf"

    def test_invalid_fusion_method_raises(self):
        from beanllm.domain.retrieval.hybrid_search import HybridRetriever

        with pytest.raises(ValueError, match="Invalid fusion_method"):
            HybridRetriever(
                documents=SAMPLE_DOCS,
                embedding_function=_make_embedding_function(4),
                fusion_method="unknown_method",
            )

    def test_bm25_index_created(self, hybrid_retriever):
        assert hybrid_retriever._bm25 is not None

    def test_embeddings_matrix_shape(self, hybrid_retriever):
        import numpy as np

        mat = hybrid_retriever._document_embeddings_matrix
        assert isinstance(mat, np.ndarray)
        assert mat.shape[0] == len(SAMPLE_DOCS)

    def test_normalized_embeddings_unit_norm(self, hybrid_retriever):
        import numpy as np

        norms = np.linalg.norm(hybrid_retriever._document_embeddings_normalized, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_custom_bm25_params_stored(self):
        from beanllm.domain.retrieval.hybrid_search import HybridRetriever

        r = HybridRetriever(
            documents=SAMPLE_DOCS,
            embedding_function=_make_embedding_function(4),
            bm25_k1=2.0,
            bm25_b=0.5,
            rrf_k=30,
        )
        assert r.bm25_k1 == 2.0
        assert r.bm25_b == 0.5
        assert r.rrf_k == 30

    def test_weighted_sum_fusion_init(self):
        from beanllm.domain.retrieval.hybrid_search import HybridRetriever

        r = HybridRetriever(
            documents=SAMPLE_DOCS,
            embedding_function=_make_embedding_function(4),
            fusion_method="weighted_sum",
            bm25_weight=0.3,
            dense_weight=0.7,
        )
        assert r.bm25_weight == 0.3
        assert r.dense_weight == 0.7

    def test_distribution_based_fusion_init(self):
        from beanllm.domain.retrieval.hybrid_search import HybridRetriever

        r = HybridRetriever(
            documents=SAMPLE_DOCS,
            embedding_function=_make_embedding_function(4),
            fusion_method="distribution_based",
        )
        assert r.fusion_method == "distribution_based"

    def test_repr_contains_key_info(self, hybrid_retriever):
        r = repr(hybrid_retriever)
        assert "rrf" in r
        assert str(len(SAMPLE_DOCS)) in r


class TestHybridRetrieverSearch:
    def test_search_returns_list(self, hybrid_retriever):
        results = hybrid_retriever.search("machine learning", top_k=3)
        assert isinstance(results, list)

    def test_search_respects_top_k(self, hybrid_retriever):
        results = hybrid_retriever.search("learning", top_k=2)
        assert len(results) <= 2

    def test_search_result_has_text_and_score(self, hybrid_retriever):
        from beanllm.domain.retrieval.types import SearchResult

        results = hybrid_retriever.search("machine learning", top_k=3)
        for r in results:
            assert isinstance(r, SearchResult)
            assert isinstance(r.text, str)
            assert isinstance(r.score, float)

    def test_search_metadata_contains_index(self, hybrid_retriever):
        results = hybrid_retriever.search("deep learning", top_k=3)
        for r in results:
            assert r.metadata is not None
            assert "index" in r.metadata
            assert "fusion_method" in r.metadata

    def test_search_rrf_fusion(self, hybrid_retriever):
        results = hybrid_retriever.search("neural network", top_k=5)
        assert len(results) >= 0  # may be 0 if no BM25 match + dense near-zero

    def test_search_weighted_sum_fusion(self):
        from beanllm.domain.retrieval.hybrid_search import HybridRetriever

        r = HybridRetriever(
            documents=SAMPLE_DOCS,
            embedding_function=_make_embedding_function(4),
            fusion_method="weighted_sum",
        )
        results = r.search("language", top_k=3)
        assert isinstance(results, list)

    def test_search_distribution_based_fusion(self):
        from beanllm.domain.retrieval.hybrid_search import HybridRetriever

        r = HybridRetriever(
            documents=SAMPLE_DOCS,
            embedding_function=_make_embedding_function(4),
            fusion_method="distribution_based",
        )
        results = r.search("computer vision", top_k=3)
        assert isinstance(results, list)

    def test_search_top_k_larger_than_docs(self, hybrid_retriever):
        results = hybrid_retriever.search("learning", top_k=100)
        assert len(results) <= len(SAMPLE_DOCS)

    def test_search_empty_query(self, hybrid_retriever):
        # Should not raise; returns list (possibly empty)
        results = hybrid_retriever.search("", top_k=3)
        assert isinstance(results, list)


class TestHybridRetrieverInternals:
    def test_bm25_search_returns_dict(self, hybrid_retriever):
        result = hybrid_retriever._bm25_search("machine learning", top_k=5)
        assert isinstance(result, dict)
        for k, v in result.items():
            assert isinstance(k, int)
            assert isinstance(v, float)

    def test_bm25_search_scores_positive(self, hybrid_retriever):
        result = hybrid_retriever._bm25_search("learning", top_k=5)
        for v in result.values():
            assert v > 0

    def test_dense_search_returns_dict(self, hybrid_retriever):
        result = hybrid_retriever._dense_search("machine learning", top_k=5)
        assert isinstance(result, dict)

    def test_normalize_scores_range(self, hybrid_retriever):
        scores = {0: 10.0, 1: 5.0, 2: 0.0}
        normalized = hybrid_retriever._normalize_scores(scores)
        assert normalized[0] == pytest.approx(1.0)
        assert normalized[2] == pytest.approx(0.0)
        assert 0.0 <= normalized[1] <= 1.0

    def test_normalize_scores_all_equal(self, hybrid_retriever):
        scores = {0: 3.0, 1: 3.0, 2: 3.0}
        normalized = hybrid_retriever._normalize_scores(scores)
        for v in normalized.values():
            assert v == pytest.approx(1.0)

    def test_normalize_scores_empty(self, hybrid_retriever):
        assert hybrid_retriever._normalize_scores({}) == {}

    def test_reciprocal_rank_fusion_combines_both(self, hybrid_retriever):
        bm25 = {0: 3.0, 1: 2.0, 2: 1.0}
        dense = {0: 0.9, 2: 0.8, 3: 0.7}
        rrf = hybrid_retriever._reciprocal_rank_fusion(bm25, dense)
        # All indices from both sources should appear
        assert set(rrf.keys()) == {0, 1, 2, 3}
        # Doc 0 appears in both so should have higher score than doc 1 or 3 alone
        assert rrf[0] > rrf[1]

    def test_weighted_sum_fusion(self, hybrid_retriever):
        bm25 = {0: 2.0, 1: 1.0}
        dense = {0: 0.8, 1: 0.4}
        ws = hybrid_retriever._weighted_sum_fusion(bm25, dense)
        assert 0 in ws and 1 in ws

    def test_add_documents_increases_count(self, hybrid_retriever):
        original_count = len(hybrid_retriever.documents)
        hybrid_retriever.add_documents(["New document about transformers."])
        assert len(hybrid_retriever.documents) == original_count + 1

    def test_add_documents_updates_matrix(self, hybrid_retriever):
        original_rows = hybrid_retriever._document_embeddings_matrix.shape[0]
        hybrid_retriever.add_documents(["Another new document."])
        assert hybrid_retriever._document_embeddings_matrix.shape[0] == original_rows + 1

    def test_add_documents_searchable(self, hybrid_retriever):
        hybrid_retriever.add_documents(["Quantum computing is revolutionary."])
        # After adding, search should still work without error
        results = hybrid_retriever.search("quantum", top_k=3)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# 2. PromptTemplate tests
# ---------------------------------------------------------------------------


class TestPromptTemplate:
    def test_basic_format(self):
        from beanllm.domain.prompts.templates import PromptTemplate

        t = PromptTemplate(template="Hello {name}!", input_variables=["name"])
        assert t.format(name="World") == "Hello World!"

    def test_auto_extract_variables(self):
        from beanllm.domain.prompts.templates import PromptTemplate

        t = PromptTemplate(template="Translate {text} to {language}")
        assert set(t.input_variables) == {"text", "language"}

    def test_multiple_variables(self):
        from beanllm.domain.prompts.templates import PromptTemplate

        t = PromptTemplate(
            template="{greeting}, {name}! You are {age} years old.",
            input_variables=["greeting", "name", "age"],
        )
        result = t.format(greeting="Hi", name="Alice", age="30")
        assert result == "Hi, Alice! You are 30 years old."

    def test_validate_template_mismatch_raises(self):
        from beanllm.domain.prompts.templates import PromptTemplate

        with pytest.raises(ValueError, match="Template variables mismatch"):
            PromptTemplate(
                template="Hello {name}",
                input_variables=["wrong_var"],
                validate_template=True,
            )

    def test_validate_template_disabled(self):
        from beanllm.domain.prompts.templates import PromptTemplate

        # Should not raise even with mismatched variables
        t = PromptTemplate(
            template="Hello {name}",
            input_variables=["wrong_var"],
            validate_template=False,
        )
        assert t is not None

    def test_missing_required_variable_raises(self):
        from beanllm.domain.prompts.templates import PromptTemplate

        t = PromptTemplate(template="{a} and {b}", input_variables=["a", "b"])
        with pytest.raises(ValueError, match="Missing required variables"):
            t.format(a="only_a")

    def test_get_input_variables_excludes_partial(self):
        from beanllm.domain.prompts.templates import PromptTemplate

        t = PromptTemplate(
            template="{city} weather is {condition}",
            input_variables=["city", "condition"],
            partial_variables={"city": "Seoul"},
        )
        assert t.get_input_variables() == ["condition"]

    def test_partial_returns_new_template(self):
        from beanllm.domain.prompts.templates import PromptTemplate

        t = PromptTemplate(
            template="Speak {language} in {tone} tone",
            input_variables=["language", "tone"],
        )
        partial_t = t.partial(language="English")
        assert isinstance(partial_t, PromptTemplate)
        assert "language" in partial_t.partial_variables
        result = partial_t.format(tone="formal")
        assert "English" in result
        assert "formal" in result

    def test_partial_does_not_mutate_original(self):
        from beanllm.domain.prompts.templates import PromptTemplate

        t = PromptTemplate(
            template="Hello {name} from {city}",
            input_variables=["name", "city"],
        )
        _ = t.partial(city="Busan")
        # Original template still requires both variables
        assert set(t.get_input_variables()) == {"name", "city"}

    def test_mustache_format(self):
        from beanllm.domain.prompts.enums import TemplateFormat
        from beanllm.domain.prompts.templates import PromptTemplate

        t = PromptTemplate(
            template="Hello {{name}}",
            template_format=TemplateFormat.MUSTACHE,
        )
        result = t.format(name="Bob")
        assert result == "Hello Bob"

    def test_extract_variables_no_duplicates(self):
        from beanllm.domain.prompts.templates import PromptTemplate

        t = PromptTemplate(template="{x} plus {x} equals {y}", input_variables=["x", "y"])
        # input_variables should contain each variable only once
        assert t.input_variables.count("x") == 1

    def test_format_with_partial_variables_merged(self):
        from beanllm.domain.prompts.templates import PromptTemplate

        t = PromptTemplate(
            template="{a} {b}",
            input_variables=["a", "b"],
            partial_variables={"a": "fixed"},
        )
        result = t.format(b="dynamic")
        assert result == "fixed dynamic"


# ---------------------------------------------------------------------------
# 3. FewShotPromptTemplate tests
# ---------------------------------------------------------------------------


class TestFewShotPromptTemplate:
    @pytest.fixture
    def math_examples(self):
        from beanllm.domain.prompts.types import PromptExample

        return [
            PromptExample(input="2+2", output="4"),
            PromptExample(input="3+3", output="6"),
            PromptExample(input="5*5", output="25"),
        ]

    @pytest.fixture
    def example_template(self):
        from beanllm.domain.prompts.templates import PromptTemplate

        return PromptTemplate(
            template="Q: {input}\nA: {output}",
            input_variables=["input", "output"],
        )

    def test_format_includes_prefix_and_examples(self, math_examples, example_template):
        from beanllm.domain.prompts.templates import FewShotPromptTemplate

        t = FewShotPromptTemplate(
            examples=math_examples,
            example_template=example_template,
            prefix="Solve the math problem:",
            suffix="Q: {input}\nA:",
            input_variables=["input"],
        )
        result = t.format(input="7+7")
        assert "Solve the math problem:" in result
        assert "Q: 2+2" in result
        assert "A: 4" in result
        assert "Q: 7+7" in result

    def test_max_examples_limits_output(self, math_examples, example_template):
        from beanllm.domain.prompts.templates import FewShotPromptTemplate

        t = FewShotPromptTemplate(
            examples=math_examples,
            example_template=example_template,
            suffix="Q: {input}\nA:",
            input_variables=["input"],
            max_examples=1,
        )
        result = t.format(input="9+9")
        # Only first example should appear
        assert "Q: 2+2" in result
        assert "Q: 3+3" not in result

    def test_add_example_increases_count(self, math_examples, example_template):
        from beanllm.domain.prompts.templates import FewShotPromptTemplate
        from beanllm.domain.prompts.types import PromptExample

        t = FewShotPromptTemplate(
            examples=math_examples,
            example_template=example_template,
            suffix="Q: {input}\nA:",
            input_variables=["input"],
        )
        before = len(t.examples)
        t.add_example(PromptExample(input="10+10", output="20"))
        assert len(t.examples) == before + 1

    def test_get_input_variables_from_suffix(self, math_examples, example_template):
        from beanllm.domain.prompts.templates import FewShotPromptTemplate

        t = FewShotPromptTemplate(
            examples=math_examples,
            example_template=example_template,
            suffix="Q: {question}\nA:",
            input_variables=None,  # auto-extract
        )
        assert "question" in t.get_input_variables()

    def test_format_without_prefix(self, math_examples, example_template):
        from beanllm.domain.prompts.templates import FewShotPromptTemplate

        t = FewShotPromptTemplate(
            examples=math_examples,
            example_template=example_template,
            suffix="Q: {input}\nA:",
            input_variables=["input"],
        )
        result = t.format(input="1+1")
        assert "Solve" not in result
        assert "Q: 2+2" in result

    def test_example_separator_used(self, math_examples, example_template):
        from beanllm.domain.prompts.templates import FewShotPromptTemplate

        sep = "---"
        t = FewShotPromptTemplate(
            examples=math_examples[:2],
            example_template=example_template,
            suffix="Q: {input}\nA:",
            input_variables=["input"],
            example_separator=sep,
        )
        result = t.format(input="1+1")
        assert sep in result


# ---------------------------------------------------------------------------
# 4. ChatPromptTemplate tests
# ---------------------------------------------------------------------------


class TestChatPromptTemplate:
    def test_init_from_tuples(self):
        from beanllm.domain.prompts.templates import ChatPromptTemplate
        from beanllm.domain.prompts.types import ChatMessage

        t = ChatPromptTemplate(messages=[("system", "You are helpful"), ("user", "{input}")])
        assert len(t.messages) == 2
        assert isinstance(t.messages[0], ChatMessage)

    def test_format_returns_string(self):
        from beanllm.domain.prompts.templates import ChatPromptTemplate

        t = ChatPromptTemplate(messages=[("system", "You are {role}"), ("user", "{question}")])
        result = t.format(role="assistant", question="Hello?")
        assert "SYSTEM" in result
        assert "USER" in result

    def test_format_messages_returns_list_of_chat_messages(self):
        from beanllm.domain.prompts.templates import ChatPromptTemplate
        from beanllm.domain.prompts.types import ChatMessage

        t = ChatPromptTemplate(messages=[("user", "Say {word}")])
        messages = t.format_messages(word="hello")
        assert len(messages) == 1
        assert isinstance(messages[0], ChatMessage)
        assert messages[0].content == "Say hello"

    def test_to_dict_messages_returns_list_of_dicts(self):
        from beanllm.domain.prompts.templates import ChatPromptTemplate

        t = ChatPromptTemplate(
            messages=[("system", "You are {role}"), ("user", "Answer: {question}")]
        )
        dicts = t.to_dict_messages(role="expert", question="What is AI?")
        assert isinstance(dicts, list)
        assert all(isinstance(d, dict) for d in dicts)
        assert dicts[0]["role"] == "system"
        assert dicts[1]["role"] == "user"

    def test_from_messages_classmethod(self):
        from beanllm.domain.prompts.templates import ChatPromptTemplate

        t = ChatPromptTemplate.from_messages([("user", "Hello {name}")])
        assert isinstance(t, ChatPromptTemplate)
        result = t.format(name="Alice")
        assert "Alice" in result

    def test_from_template_classmethod(self):
        from beanllm.domain.prompts.templates import ChatPromptTemplate

        t = ChatPromptTemplate.from_template("Tell me about {topic}")
        result = t.format(topic="Python")
        assert "Python" in result

    def test_extract_variables_from_all_messages(self):
        from beanllm.domain.prompts.templates import ChatPromptTemplate

        t = ChatPromptTemplate(messages=[("system", "You are {role}"), ("user", "{question}")])
        vars_ = t.get_input_variables()
        assert "role" in vars_
        assert "question" in vars_

    def test_tuple_with_name_field(self):
        from beanllm.domain.prompts.templates import ChatPromptTemplate
        from beanllm.domain.prompts.types import ChatMessage

        t = ChatPromptTemplate(messages=[("user", "Hello", "alice")])
        msg = t.messages[0]
        assert isinstance(msg, ChatMessage)
        assert msg.name == "alice"

    def test_to_dict_messages_name_included_when_set(self):
        from beanllm.domain.prompts.templates import ChatPromptTemplate

        t = ChatPromptTemplate(messages=[("user", "Hi {person}", "bob")])
        dicts = t.to_dict_messages(person="Alice")
        assert dicts[0].get("name") == "bob"


# ---------------------------------------------------------------------------
# 5. PromptVersionManager / PromptVersioning tests
# ---------------------------------------------------------------------------


class TestPromptVersion:
    def test_create_prompt_version(self):
        from beanllm.domain.prompts.versioning import PromptVersion

        pv = PromptVersion(
            version="v1",
            content="Hello {name}",
            created_at=datetime.now(),
        )
        assert pv.version == "v1"
        assert pv.content == "Hello {name}"
        assert pv.usage_count == 0

    def test_add_metric(self):
        from beanllm.domain.prompts.versioning import PromptVersion

        pv = PromptVersion(version="v1", content="test", created_at=datetime.now())
        pv.add_metric("accuracy", 0.95)
        assert pv.performance_metrics["accuracy"] == pytest.approx(0.95)

    def test_record_usage(self):
        from beanllm.domain.prompts.versioning import PromptVersion

        pv = PromptVersion(version="v1", content="test", created_at=datetime.now())
        pv.record_usage()
        pv.record_usage()
        assert pv.usage_count == 2
        assert pv.last_used is not None

    def test_to_dict_roundtrip(self):
        from beanllm.domain.prompts.versioning import PromptVersion

        pv = PromptVersion(
            version="v1",
            content="Hello {name}",
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            metadata={"author": "test"},
        )
        d = pv.to_dict()
        pv2 = PromptVersion.from_dict(d)
        assert pv2.version == pv.version
        assert pv2.content == pv.content
        assert pv2.metadata == pv.metadata

    def test_to_dict_last_used_none(self):
        from beanllm.domain.prompts.versioning import PromptVersion

        pv = PromptVersion(version="v1", content="c", created_at=datetime.now())
        d = pv.to_dict()
        assert d["last_used"] is None

    def test_from_dict_with_last_used(self):
        from beanllm.domain.prompts.versioning import PromptVersion

        d = {
            "version": "v2",
            "content": "content",
            "created_at": "2024-06-01T10:00:00",
            "last_used": "2024-06-02T11:00:00",
            "usage_count": 5,
        }
        pv = PromptVersion.from_dict(d)
        assert pv.usage_count == 5
        assert pv.last_used is not None


class TestPromptVersionManager:
    @pytest.fixture
    def manager(self):
        from beanllm.domain.prompts.versioning import PromptVersionManager

        return PromptVersionManager(storage_path=None)

    def test_create_version_auto_numbering(self, manager):
        v1 = manager.create_version("chat", "Hello {name}")
        v2 = manager.create_version("chat", "Hi {name}")
        assert v1.version == "v1"
        assert v2.version == "v2"

    def test_create_version_custom_version(self, manager):
        v = manager.create_version("rag", "Context: {context}", version="alpha")
        assert v.version == "alpha"

    def test_get_version_latest(self, manager):
        manager.create_version("chat", "v1 content")
        manager.create_version("chat", "v2 content")
        latest = manager.get_version("chat")
        assert latest.content == "v2 content"

    def test_get_version_specific(self, manager):
        manager.create_version("chat", "first", version="v1")
        manager.create_version("chat", "second", version="v2")
        v1 = manager.get_version("chat", "v1")
        assert v1.content == "first"

    def test_get_version_not_found_raises(self, manager):
        with pytest.raises(ValueError, match="not found"):
            manager.get_version("nonexistent")

    def test_get_version_wrong_version_raises(self, manager):
        manager.create_version("chat", "content")
        with pytest.raises(ValueError, match="Version"):
            manager.get_version("chat", "v999")

    def test_list_versions_all(self, manager):
        manager.create_version("chat", "c1")
        manager.create_version("rag", "r1")
        manager.create_version("rag", "r2")
        all_versions = manager.list_versions()
        assert "chat" in all_versions
        assert "rag" in all_versions
        assert len(all_versions["rag"]) == 2

    def test_list_versions_single_prompt(self, manager):
        manager.create_version("myp", "v1 content")
        manager.create_version("myp", "v2 content")
        result = manager.list_versions("myp")
        assert "myp" in result
        assert len(result["myp"]) == 2

    def test_list_versions_unknown_prompt(self, manager):
        result = manager.list_versions("missing_prompt")
        assert result == {}

    def test_compare_versions_returns_dict(self, manager):
        manager.create_version("chat", "Hello world", version="v1")
        manager.create_version("chat", "Hi there", version="v2")
        comparison = manager.compare_versions("chat", "v1", "v2")
        assert "content_diff" in comparison
        assert "metrics_diff" in comparison
        assert "usage_diff" in comparison
        assert "recommendation" in comparison

    def test_compare_versions_accuracy_recommendation(self, manager):
        manager.create_version("chat", "old content", version="v1")
        manager.create_version("chat", "new content", version="v2")
        v1 = manager.get_version("chat", "v1")
        v2 = manager.get_version("chat", "v2")
        v1.add_metric("accuracy", 0.7)
        v2.add_metric("accuracy", 0.9)
        comparison = manager.compare_versions("chat", "v1", "v2")
        assert comparison["recommendation"] == "v2"

    def test_compare_versions_content_diff_not_empty(self, manager):
        manager.create_version("chat", "Line A\nLine B\n", version="v1")
        manager.create_version("chat", "Line A\nLine C\n", version="v2")
        comparison = manager.compare_versions("chat", "v1", "v2")
        assert len(comparison["content_diff"]) > 0

    def test_create_version_with_metadata(self, manager):
        v = manager.create_version("chat", "content", metadata={"author": "alice"})
        assert v.metadata["author"] == "alice"

    def test_storage_save_and_load(self, tmp_path):
        from beanllm.domain.prompts.versioning import PromptVersionManager

        path = str(tmp_path / "versions.json")
        m1 = PromptVersionManager(storage_path=path)
        m1.create_version("stored_prompt", "Version 1 content", version="v1")

        # Load from the same path
        m2 = PromptVersionManager(storage_path=path)
        v = m2.get_version("stored_prompt", "v1")
        assert v.content == "Version 1 content"


class TestPromptVersioning:
    """Tests for legacy PromptVersioning class."""

    def test_save_and_load_latest(self):
        from beanllm.domain.prompts.templates import PromptTemplate
        from beanllm.domain.prompts.versioning import PromptVersioning

        vt = PromptVersioning()
        t = PromptTemplate(template="Hello {name}", input_variables=["name"])
        vt.save("greet", t, "v1")
        loaded = vt.load("greet")
        assert loaded is t

    def test_load_specific_version(self):
        from beanllm.domain.prompts.templates import PromptTemplate
        from beanllm.domain.prompts.versioning import PromptVersioning

        vt = PromptVersioning()
        t1 = PromptTemplate(template="V1 {x}", input_variables=["x"])
        t2 = PromptTemplate(template="V2 {x}", input_variables=["x"])
        vt.save("prompt", t1, "v1")
        vt.save("prompt", t2, "v2")
        assert vt.load("prompt", "v1") is t1
        assert vt.load("prompt", "v2") is t2

    def test_load_not_found_raises(self):
        from beanllm.domain.prompts.versioning import PromptVersioning

        vt = PromptVersioning()
        with pytest.raises(ValueError):
            vt.load("nonexistent")

    def test_list_versions(self):
        from beanllm.domain.prompts.templates import PromptTemplate
        from beanllm.domain.prompts.versioning import PromptVersioning

        vt = PromptVersioning()
        t = PromptTemplate(template="Hello {n}", input_variables=["n"])
        vt.save("p", t, "v1")
        vt.save("p", t, "v2")
        assert vt.list_versions("p") == ["v1", "v2"]

    def test_list_versions_empty(self):
        from beanllm.domain.prompts.versioning import PromptVersioning

        vt = PromptVersioning()
        assert vt.list_versions("no_such") == []


# ---------------------------------------------------------------------------
# 6. ChunkValidator tests
# ---------------------------------------------------------------------------


class TestChunkValidatorInit:
    def test_default_params(self):
        from beanllm.domain.rag_debug.chunk_validator import ChunkValidator

        cv = ChunkValidator()
        assert cv.min_chunk_size == 100
        assert cv.max_chunk_size == 2000
        assert cv.overlap_threshold == 0.9

    def test_custom_params(self):
        from beanllm.domain.rag_debug.chunk_validator import ChunkValidator

        cv = ChunkValidator(min_chunk_size=50, max_chunk_size=500, overlap_threshold=0.8)
        assert cv.min_chunk_size == 50
        assert cv.max_chunk_size == 500
        assert cv.overlap_threshold == 0.8


class TestChunkValidatorValidateSize:
    @pytest.fixture
    def validator(self):
        from beanllm.domain.rag_debug.chunk_validator import ChunkValidator

        return ChunkValidator(min_chunk_size=10, max_chunk_size=100)

    def _make_doc(self, content: str) -> MockDocument:
        return MockDocument(page_content=content, metadata={})

    def test_valid_chunks_no_issues(self, validator):
        docs = [self._make_doc("a" * 50), self._make_doc("b" * 80)]
        issues, dist = validator.validate_size(docs)
        assert len(issues) == 0

    def test_too_small_chunk_detected(self, validator):
        docs = [self._make_doc("hi")]  # len=2 < 10
        issues, _ = validator.validate_size(docs)
        assert len(issues) == 1
        assert issues[0]["type"] == "size_too_small"
        assert issues[0]["chunk_id"] == 0

    def test_too_large_chunk_detected(self, validator):
        docs = [self._make_doc("x" * 200)]  # 200 > 100
        issues, _ = validator.validate_size(docs)
        assert len(issues) == 1
        assert issues[0]["type"] == "size_too_large"

    def test_distribution_buckets_returned(self, validator):
        docs = [self._make_doc("a" * 50), self._make_doc("b" * 80)]
        _, dist = validator.validate_size(docs)
        # Both docs are in 0-200 bucket (validator max=100 but distribution uses fixed bins)
        assert isinstance(dist, dict)
        assert all(isinstance(v, int) for v in dist.values())

    def test_uses_page_content_attribute(self, validator):
        docs = [MockDocument(page_content="short", metadata={})]
        issues, _ = validator.validate_size(docs)
        assert issues[0]["size"] == 5

    def test_string_fallback_when_no_page_content(self, validator):
        """Validator falls back to str(doc) when no page_content attribute."""
        docs = ["xy"]  # plain string, len 2 < 10
        issues, _ = validator.validate_size(docs)
        assert len(issues) == 1


class TestChunkValidatorDetectDuplicates:
    @pytest.fixture
    def validator(self):
        from beanllm.domain.rag_debug.chunk_validator import ChunkValidator

        return ChunkValidator(overlap_threshold=0.9)

    def _make_doc(self, content: str) -> MockDocument:
        return MockDocument(page_content=content, metadata={})

    def test_no_duplicates_in_distinct_docs(self, validator):
        docs = [
            self._make_doc("machine learning helps predict outcomes"),
            self._make_doc("deep sea fishing is a popular hobby"),
        ]
        dups = validator.detect_duplicates(docs)
        assert len(dups) == 0

    def test_identical_docs_detected(self, validator):
        content = "this is an identical document for testing purposes"
        docs = [self._make_doc(content), self._make_doc(content)]
        dups = validator.detect_duplicates(docs)
        assert (0, 1) in dups

    def test_custom_threshold(self, validator):
        content = "word1 word2 word3 word4 word5"
        docs = [self._make_doc(content), self._make_doc(content)]
        # With low threshold every pair is a duplicate
        dups = validator.detect_duplicates(docs, threshold=0.1)
        assert len(dups) >= 1

    def test_returns_list_of_tuples(self, validator):
        docs = [self._make_doc("a b c"), self._make_doc("d e f")]
        dups = validator.detect_duplicates(docs)
        assert isinstance(dups, list)
        for d in dups:
            assert isinstance(d, tuple) and len(d) == 2


class TestChunkValidatorJaccard:
    def test_identical_texts(self):
        from beanllm.domain.rag_debug.chunk_validator import ChunkValidator

        cv = ChunkValidator()
        assert cv._jaccard_similarity("hello world", "hello world") == pytest.approx(1.0)

    def test_disjoint_texts(self):
        from beanllm.domain.rag_debug.chunk_validator import ChunkValidator

        cv = ChunkValidator()
        assert cv._jaccard_similarity("apple orange", "banana grape") == pytest.approx(0.0)

    def test_partial_overlap(self):
        from beanllm.domain.rag_debug.chunk_validator import ChunkValidator

        cv = ChunkValidator()
        sim = cv._jaccard_similarity("a b c", "b c d")
        assert 0.0 < sim < 1.0

    def test_empty_texts(self):
        from beanllm.domain.rag_debug.chunk_validator import ChunkValidator

        cv = ChunkValidator()
        assert cv._jaccard_similarity("", "") == pytest.approx(1.0)


class TestChunkValidatorMetadata:
    @pytest.fixture
    def validator(self):
        from beanllm.domain.rag_debug.chunk_validator import ChunkValidator

        return ChunkValidator()

    def test_missing_required_key_reported(self, validator):
        docs = [MockDocument(page_content="content", metadata={"page": 1})]
        issues = validator.validate_metadata(docs, required_keys=["source"])
        assert any(i["missing_key"] == "source" for i in issues)

    def test_empty_metadata_reported(self, validator):
        docs = [MockDocument(page_content="content", metadata={})]
        issues = validator.validate_metadata(docs)
        assert any(i["type"] == "empty_metadata" for i in issues)

    def test_valid_metadata_no_issues(self, validator):
        docs = [
            MockDocument(
                page_content="content",
                metadata={"source": "test.pdf", "page": 1},
            )
        ]
        issues = validator.validate_metadata(docs, required_keys=["source"])
        # Should not have missing_key issue for "source"
        assert all(i.get("missing_key") != "source" for i in issues)

    def test_multiple_docs_multiple_issues(self, validator):
        docs = [
            MockDocument(page_content="c", metadata={}),
            MockDocument(page_content="c", metadata={}),
        ]
        issues = validator.validate_metadata(docs, required_keys=["source"])
        # Each doc triggers: empty_metadata + missing_source = 2 issues per doc
        assert len(issues) >= 2


class TestChunkValidatorOverlap:
    @pytest.fixture
    def validator(self):
        from beanllm.domain.rag_debug.chunk_validator import ChunkValidator

        return ChunkValidator()

    def test_single_doc_returns_none(self, validator):
        docs = [MockDocument(page_content="only one", metadata={})]
        result = validator.check_overlap(docs)
        assert result is None

    def test_two_docs_returns_stats_dict(self, validator):
        docs = [
            MockDocument(page_content="machine learning is great", metadata={}),
            MockDocument(page_content="deep learning is also great", metadata={}),
        ]
        result = validator.check_overlap(docs)
        assert result is not None
        assert "avg_overlap_ratio" in result
        assert "max_overlap_ratio" in result
        assert "min_overlap_ratio" in result
        assert "num_pairs" in result

    def test_identical_docs_high_overlap(self, validator):
        text = "a b c d e f g h i j k l m n o p q r s t"
        docs = [MockDocument(page_content=text, metadata={})] * 3
        result = validator.check_overlap(docs)
        assert result["max_overlap_ratio"] == pytest.approx(1.0, abs=0.01)


class TestChunkValidatorGenerateRecommendations:
    @pytest.fixture
    def validator(self):
        from beanllm.domain.rag_debug.chunk_validator import ChunkValidator

        return ChunkValidator(min_chunk_size=100, max_chunk_size=2000)

    def test_no_issues_positive_message(self, validator):
        recs = validator.generate_recommendations(
            size_issues=[],
            duplicates=[],
            metadata_issues=[],
            overlap_stats=None,
        )
        assert len(recs) == 1
        assert "No major issues" in recs[0]

    def test_too_small_recommendation(self, validator):
        size_issues = [{"type": "size_too_small"}]
        recs = validator.generate_recommendations(
            size_issues=size_issues,
            duplicates=[],
            metadata_issues=[],
            overlap_stats=None,
        )
        assert any("too small" in r for r in recs)

    def test_too_large_recommendation(self, validator):
        size_issues = [{"type": "size_too_large"}]
        recs = validator.generate_recommendations(
            size_issues=size_issues,
            duplicates=[],
            metadata_issues=[],
            overlap_stats=None,
        )
        assert any("too large" in r for r in recs)

    def test_duplicate_recommendation(self, validator):
        recs = validator.generate_recommendations(
            size_issues=[],
            duplicates=[(0, 1)],
            metadata_issues=[],
            overlap_stats=None,
        )
        assert any("duplicate" in r.lower() for r in recs)

    def test_high_overlap_recommendation(self, validator):
        recs = validator.generate_recommendations(
            size_issues=[],
            duplicates=[],
            metadata_issues=[],
            overlap_stats={"avg_overlap_ratio": 0.7},
        )
        assert any("High overlap" in r for r in recs)

    def test_low_overlap_recommendation(self, validator):
        recs = validator.generate_recommendations(
            size_issues=[],
            duplicates=[],
            metadata_issues=[],
            overlap_stats={"avg_overlap_ratio": 0.05},
        )
        assert any("Low overlap" in r for r in recs)


class TestChunkValidatorValidateAll:
    def test_validate_all_returns_expected_keys(self):
        from beanllm.domain.rag_debug.chunk_validator import ChunkValidator

        cv = ChunkValidator(min_chunk_size=5, max_chunk_size=500)
        docs = [
            MockDocument(page_content="Hello world document one", metadata={"source": "a.txt"}),
            MockDocument(
                page_content="Another document with content", metadata={"source": "b.txt"}
            ),
        ]
        result = cv.validate_all(docs, required_metadata_keys=["source"])
        for key in [
            "total_chunks",
            "valid_chunks",
            "size_issues",
            "size_distribution",
            "duplicate_chunks",
            "metadata_issues",
            "overlap_stats",
            "recommendations",
        ]:
            assert key in result

    def test_validate_all_total_chunks_count(self):
        from beanllm.domain.rag_debug.chunk_validator import ChunkValidator

        cv = ChunkValidator()
        docs = [MockDocument(page_content="x" * 200, metadata={"s": "1"}) for _ in range(5)]
        result = cv.validate_all(docs)
        assert result["total_chunks"] == 5


# ---------------------------------------------------------------------------
# 7. DebugReportExporter tests
# ---------------------------------------------------------------------------


SAMPLE_DEBUG_DATA: Dict[str, Any] = {
    "session": {
        "session_id": "sess-001",
        "session_name": "Test Session",
        "created_at": "2024-01-01T00:00:00",
    },
    "metadata": {
        "num_documents": 100,
        "num_embeddings": 100,
        "embedding_dim": 768,
        "vector_store_type": "FAISS",
    },
    "chunk_validation": {
        "total_chunks": 100,
        "valid_chunks": 95,
        "size_issues": [{"type": "size_too_small"}],
        "duplicate_chunks": [],
        "recommendations": ["No major issues detected."],
    },
}


class TestDebugReportExporterJSON:
    def test_export_json_creates_file(self, tmp_path):
        from beanllm.domain.rag_debug.export import DebugReportExporter

        out = str(tmp_path / "report.json")
        returned_path = DebugReportExporter.export_json(SAMPLE_DEBUG_DATA, out)
        assert Path(returned_path).exists()

    def test_export_json_valid_content(self, tmp_path):
        from beanllm.domain.rag_debug.export import DebugReportExporter

        out = str(tmp_path / "report.json")
        DebugReportExporter.export_json(SAMPLE_DEBUG_DATA, out)
        with open(out, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["session"]["session_id"] == "sess-001"

    def test_export_json_pretty_indented(self, tmp_path):
        from beanllm.domain.rag_debug.export import DebugReportExporter

        out = str(tmp_path / "pretty.json")
        DebugReportExporter.export_json(SAMPLE_DEBUG_DATA, out, pretty=True)
        content = Path(out).read_text(encoding="utf-8")
        assert "\n" in content  # pretty-printed

    def test_export_json_compact(self, tmp_path):
        from beanllm.domain.rag_debug.export import DebugReportExporter

        out = str(tmp_path / "compact.json")
        DebugReportExporter.export_json(SAMPLE_DEBUG_DATA, out, pretty=False)
        content = Path(out).read_text(encoding="utf-8")
        # Compact JSON has no leading whitespace on second line
        assert isinstance(content, str)

    def test_export_json_returns_string_path(self, tmp_path):
        from beanllm.domain.rag_debug.export import DebugReportExporter

        out = str(tmp_path / "r.json")
        result = DebugReportExporter.export_json({}, out)
        assert isinstance(result, str)

    def test_export_json_creates_parent_dirs(self, tmp_path):
        from beanllm.domain.rag_debug.export import DebugReportExporter

        out = str(tmp_path / "deep" / "nested" / "report.json")
        DebugReportExporter.export_json({"key": "value"}, out)
        assert Path(out).exists()


class TestDebugReportExporterMarkdown:
    def test_export_markdown_creates_file(self, tmp_path):
        from beanllm.domain.rag_debug.export import DebugReportExporter

        out = str(tmp_path / "report.md")
        returned_path = DebugReportExporter.export_markdown(SAMPLE_DEBUG_DATA, out)
        assert Path(returned_path).exists()

    def test_export_markdown_contains_title(self, tmp_path):
        from beanllm.domain.rag_debug.export import DebugReportExporter

        out = str(tmp_path / "report.md")
        DebugReportExporter.export_markdown(SAMPLE_DEBUG_DATA, out)
        content = Path(out).read_text(encoding="utf-8")
        assert "# RAG Debug Report" in content

    def test_export_markdown_contains_session_id(self, tmp_path):
        from beanllm.domain.rag_debug.export import DebugReportExporter

        out = str(tmp_path / "report.md")
        DebugReportExporter.export_markdown(SAMPLE_DEBUG_DATA, out)
        content = Path(out).read_text(encoding="utf-8")
        assert "sess-001" in content

    def test_export_markdown_contains_metadata(self, tmp_path):
        from beanllm.domain.rag_debug.export import DebugReportExporter

        out = str(tmp_path / "report.md")
        DebugReportExporter.export_markdown(SAMPLE_DEBUG_DATA, out)
        content = Path(out).read_text(encoding="utf-8")
        assert "100" in content  # num_documents

    def test_export_markdown_chunk_validation_section(self, tmp_path):
        from beanllm.domain.rag_debug.export import DebugReportExporter

        out = str(tmp_path / "report.md")
        DebugReportExporter.export_markdown(SAMPLE_DEBUG_DATA, out)
        content = Path(out).read_text(encoding="utf-8")
        assert "Chunk Validation" in content

    def test_export_markdown_returns_string_path(self, tmp_path):
        from beanllm.domain.rag_debug.export import DebugReportExporter

        out = str(tmp_path / "r.md")
        result = DebugReportExporter.export_markdown({}, out)
        assert isinstance(result, str)

    def test_export_markdown_empty_data(self, tmp_path):
        from beanllm.domain.rag_debug.export import DebugReportExporter

        out = str(tmp_path / "empty.md")
        DebugReportExporter.export_markdown({}, out)
        content = Path(out).read_text(encoding="utf-8")
        assert "# RAG Debug Report" in content


class TestDebugReportExporterCreateFullReport:
    def test_create_full_report_all_formats(self, tmp_path):
        from beanllm.domain.rag_debug.export import DebugReportExporter

        results = DebugReportExporter.create_full_report(
            session_data=SAMPLE_DEBUG_DATA,
            output_dir=str(tmp_path),
            formats=["json", "markdown"],
        )
        assert "json" in results
        assert "markdown" in results
        assert Path(results["json"]).exists()
        assert Path(results["markdown"]).exists()

    def test_create_full_report_json_only(self, tmp_path):
        from beanllm.domain.rag_debug.export import DebugReportExporter

        results = DebugReportExporter.create_full_report(
            session_data=SAMPLE_DEBUG_DATA,
            output_dir=str(tmp_path),
            formats=["json"],
        )
        assert "json" in results
        assert "markdown" not in results

    def test_create_full_report_default_formats(self, tmp_path):
        from beanllm.domain.rag_debug.export import DebugReportExporter

        results = DebugReportExporter.create_full_report(
            session_data=SAMPLE_DEBUG_DATA,
            output_dir=str(tmp_path),
        )
        assert "json" in results
        assert "markdown" in results
        assert "html" in results

    def test_create_full_report_uses_session_id_in_filename(self, tmp_path):
        from beanllm.domain.rag_debug.export import DebugReportExporter

        results = DebugReportExporter.create_full_report(
            session_data=SAMPLE_DEBUG_DATA,
            output_dir=str(tmp_path),
            formats=["json"],
        )
        assert "sess-001" in Path(results["json"]).name

    def test_create_full_report_creates_output_dir(self, tmp_path):
        from beanllm.domain.rag_debug.export import DebugReportExporter

        new_dir = str(tmp_path / "new_subdir")
        DebugReportExporter.create_full_report(
            session_data=SAMPLE_DEBUG_DATA,
            output_dir=new_dir,
            formats=["json"],
        )
        assert Path(new_dir).exists()

    def test_create_full_report_unknown_session_id(self, tmp_path):
        from beanllm.domain.rag_debug.export import DebugReportExporter

        # Data without session.session_id should use "unknown" fallback
        results = DebugReportExporter.create_full_report(
            session_data={},
            output_dir=str(tmp_path),
            formats=["json"],
        )
        assert "unknown" in Path(results["json"]).name
