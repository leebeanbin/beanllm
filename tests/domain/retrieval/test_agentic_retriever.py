"""Tests for AgenticRetriever."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from beanllm.domain.retrieval.agentic_retriever import AgenticRetriever

# ---------------------------------------------------------------------------
# helpers / fakes
# ---------------------------------------------------------------------------


class FakeVectorStore:
    def __init__(self, docs: Optional[List[Any]] = None):
        self._docs = docs or [MagicMock(page_content="doc1"), MagicMock(page_content="doc2")]

    def similarity_search(self, query: str, k: int = 4) -> List[Any]:
        return self._docs[:k]


class FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLMProvider:
    def __init__(self, needs: bool = True):
        self._needs = needs
        self.call_count = 0

    async def chat(self, messages, model, max_tokens=None, **kwargs) -> FakeLLMResponse:
        self.call_count += 1
        payload = json.dumps({"needs_retrieval": self._needs, "reason": "test"})
        return FakeLLMResponse(payload)


# ---------------------------------------------------------------------------
# init / repr
# ---------------------------------------------------------------------------


class TestAgenticRetrieverInit:
    def test_default_params(self):
        store = FakeVectorStore()
        llm = FakeLLMProvider()
        r = AgenticRetriever(vector_store=store, llm_provider=llm)
        assert r._model == "gpt-4o-mini"
        assert r._k == 4
        assert r._decision_max_tokens == 64

    def test_custom_params(self):
        store = FakeVectorStore()
        llm = FakeLLMProvider()
        r = AgenticRetriever(vector_store=store, llm_provider=llm, model="gpt-4o", k=8)
        assert r._model == "gpt-4o"
        assert r._k == 8

    def test_repr(self):
        store = FakeVectorStore()
        llm = FakeLLMProvider()
        r = AgenticRetriever(vector_store=store, llm_provider=llm)
        text = repr(r)
        assert "AgenticRetriever" in text
        assert "gpt-4o-mini" in text
        assert "FakeVectorStore" in text


# ---------------------------------------------------------------------------
# maybe_retrieve
# ---------------------------------------------------------------------------


class TestMaybeRetrieve:
    async def test_returns_docs_when_llm_says_yes(self):
        store = FakeVectorStore()
        llm = FakeLLMProvider(needs=True)
        r = AgenticRetriever(vector_store=store, llm_provider=llm, k=2)

        docs = await r.maybe_retrieve("what is AI?")
        assert docs is not None
        assert len(docs) == 2

    async def test_returns_none_when_llm_says_no(self):
        store = FakeVectorStore()
        llm = FakeLLMProvider(needs=False)
        r = AgenticRetriever(vector_store=store, llm_provider=llm)

        docs = await r.maybe_retrieve("what is 2+2?")
        assert docs is None

    async def test_llm_called_once(self):
        store = FakeVectorStore()
        llm = FakeLLMProvider(needs=True)
        r = AgenticRetriever(vector_store=store, llm_provider=llm)

        await r.maybe_retrieve("query")
        assert llm.call_count == 1

    async def test_falls_back_to_retrieve_on_llm_error(self):
        store = FakeVectorStore()

        class ErrorLLM:
            async def chat(self, *args, **kwargs):
                raise RuntimeError("LLM down")

        r = AgenticRetriever(vector_store=store, llm_provider=ErrorLLM())
        # Fallback: should still retrieve
        docs = await r.maybe_retrieve("query")
        assert docs is not None

    async def test_maybe_retrieve_passes_context_to_llm(self):
        store = FakeVectorStore()
        captured: Dict[str, Any] = {}

        class CaptureLLM:
            async def chat(self, messages, model, **kwargs) -> FakeLLMResponse:
                captured["messages"] = messages
                return FakeLLMResponse('{"needs_retrieval": false, "reason": "ok"}')

        r = AgenticRetriever(vector_store=store, llm_provider=CaptureLLM())
        await r.maybe_retrieve("my query", context="existing context")
        prompt = captured["messages"][0]["content"]
        assert "existing context" in prompt
        assert "my query" in prompt


# ---------------------------------------------------------------------------
# retrieve (sync)
# ---------------------------------------------------------------------------


class TestRetrieve:
    def test_retrieve_returns_docs(self):
        store = FakeVectorStore()
        llm = FakeLLMProvider()
        r = AgenticRetriever(vector_store=store, llm_provider=llm, k=2)
        docs = r.retrieve("any query")
        assert len(docs) == 2

    def test_retrieve_passes_k(self):
        calls: List[int] = []

        class TrackingStore:
            def similarity_search(self, query: str, k: int = 4) -> List[Any]:
                calls.append(k)
                return []

        r = AgenticRetriever(vector_store=TrackingStore(), llm_provider=FakeLLMProvider(), k=7)
        r.retrieve("q")
        assert calls == [7]


# ---------------------------------------------------------------------------
# _decide — edge cases
# ---------------------------------------------------------------------------


class TestDecide:
    async def test_decide_parses_true(self):
        store = FakeVectorStore()
        llm = FakeLLMProvider(needs=True)
        r = AgenticRetriever(vector_store=store, llm_provider=llm)
        assert await r._decide("q", "") is True

    async def test_decide_parses_false(self):
        store = FakeVectorStore()
        llm = FakeLLMProvider(needs=False)
        r = AgenticRetriever(vector_store=store, llm_provider=llm)
        assert await r._decide("q", "") is False

    async def test_decide_with_surrounding_text(self):
        store = FakeVectorStore()

        class NoisyLLM:
            async def chat(self, *args, **kwargs) -> FakeLLMResponse:
                return FakeLLMResponse('Sure! {"needs_retrieval": false, "reason": "x"} done')

        r = AgenticRetriever(vector_store=store, llm_provider=NoisyLLM())
        assert await r._decide("q", "") is False

    async def test_decide_truncates_long_context(self):
        store = FakeVectorStore()
        captured: Dict[str, Any] = {}

        class CaptureLLM:
            async def chat(self, messages, model, **kwargs) -> FakeLLMResponse:
                captured["prompt"] = messages[0]["content"]
                return FakeLLMResponse('{"needs_retrieval": true, "reason": "x"}')

        r = AgenticRetriever(vector_store=store, llm_provider=CaptureLLM())
        long_context = "X" * 2000
        await r._decide("q", long_context)
        # Context should be truncated to 800 chars
        assert len(captured["prompt"]) < 2500
