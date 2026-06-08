"""Tests for domain/retrieval/query_expansion.py."""

from unittest.mock import MagicMock

import pytest

from beanllm.domain.retrieval.query_expansion import (
    BaseQueryExpander,
    HyDEExpander,
    MultiQueryExpander,
    StepBackExpander,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm(return_value: str) -> MagicMock:
    mock = MagicMock(return_value=return_value)
    return mock


# ---------------------------------------------------------------------------
# HyDEExpander
# ---------------------------------------------------------------------------


class TestHyDEExpander:
    def test_init_defaults(self):
        llm = _make_llm("answer")
        exp = HyDEExpander(llm_function=llm)
        assert exp.num_documents == 1
        assert exp.max_tokens == 512
        assert exp.temperature > 0
        assert "{query}" in exp.prompt_template

    def test_custom_prompt_template(self):
        llm = _make_llm("answer")
        exp = HyDEExpander(llm_function=llm, prompt_template="Q: {query}\nA:")
        assert exp.prompt_template == "Q: {query}\nA:"

    def test_expand_single_document(self):
        llm = _make_llm("This is the hypothetical answer.")
        exp = HyDEExpander(llm_function=llm, num_documents=1)
        result = exp.expand("What is gravity?")
        assert result == "This is the hypothetical answer."
        llm.assert_called_once()

    def test_expand_calls_llm_with_formatted_prompt(self):
        llm = _make_llm("answer")
        exp = HyDEExpander(llm_function=llm, prompt_template="Answer: {query}")
        exp.expand("test query")
        call_args = llm.call_args[0][0]
        assert "test query" in call_args

    def test_expand_multiple_documents(self):
        llm = _make_llm("doc answer")
        exp = HyDEExpander(llm_function=llm, num_documents=3)
        result = exp.expand("What is AI?")
        assert isinstance(result, list)
        assert len(result) == 3
        assert llm.call_count == 3

    def test_expand_multiple_all_same_content(self):
        llm = _make_llm("generated doc")
        exp = HyDEExpander(llm_function=llm, num_documents=2)
        result = exp.expand("query")
        assert all(r == "generated doc" for r in result)

    def test_repr(self):
        llm = _make_llm("")
        exp = HyDEExpander(llm_function=llm, num_documents=2, temperature=0.5)
        r = repr(exp)
        assert "HyDEExpander" in r
        assert "2" in r

    def test_default_prompt_contains_question_label(self):
        llm = _make_llm("")
        exp = HyDEExpander(llm_function=llm)
        assert "Question" in exp.prompt_template or "question" in exp.prompt_template.lower()


# ---------------------------------------------------------------------------
# MultiQueryExpander
# ---------------------------------------------------------------------------


class TestMultiQueryExpander:
    def test_init_defaults(self):
        llm = _make_llm("1. query one\n2. query two\n3. query three")
        exp = MultiQueryExpander(llm_function=llm)
        assert exp.num_queries == 3
        assert "{query}" in exp.prompt_template
        assert "{num_queries}" in exp.prompt_template

    def test_custom_num_queries(self):
        llm = _make_llm("line one\nline two")
        exp = MultiQueryExpander(llm_function=llm, num_queries=5)
        assert exp.num_queries == 5

    def test_expand_returns_list(self):
        response = (
            "1. What is machine learning?\n2. Explain AI algorithms.\n3. How do neural nets work?"
        )
        llm = _make_llm(response)
        exp = MultiQueryExpander(llm_function=llm, num_queries=3)
        result = exp.expand("How does AI work?")
        assert isinstance(result, list)

    def test_expand_strips_numbering(self):
        response = "1. First query here\n2. Second query here\n3. Third query here"
        llm = _make_llm(response)
        exp = MultiQueryExpander(llm_function=llm, num_queries=3)
        result = exp.expand("query")
        assert all(not q[0].isdigit() for q in result)

    def test_expand_limits_to_num_queries(self):
        response = "\n".join(f"{i}. Query number {i} is long enough" for i in range(1, 10))
        llm = _make_llm(response)
        exp = MultiQueryExpander(llm_function=llm, num_queries=3)
        result = exp.expand("question")
        assert len(result) <= 3

    def test_expand_filters_short_lines(self):
        response = "1. ok\n2. This is a sufficiently long query variant\n3. hi"
        llm = _make_llm(response)
        exp = MultiQueryExpander(llm_function=llm, num_queries=3)
        result = exp.expand("query")
        assert all(len(q) > 10 for q in result)

    def test_expand_calls_llm_with_formatted_prompt(self):
        llm = _make_llm("1. alt query that is long enough to pass filter")
        exp = MultiQueryExpander(llm_function=llm, num_queries=2)
        exp.expand("my original question")
        call_args = llm.call_args[0][0]
        assert "my original question" in call_args

    def test_custom_prompt_template(self):
        llm = _make_llm("1. variant that is long enough here")
        exp = MultiQueryExpander(
            llm_function=llm,
            prompt_template="Rephrase: {query} (x{num_queries})",
            num_queries=1,
        )
        exp.expand("test")
        assert "test" in llm.call_args[0][0]

    def test_repr(self):
        llm = _make_llm("")
        exp = MultiQueryExpander(llm_function=llm, num_queries=5)
        assert "MultiQueryExpander" in repr(exp)
        assert "5" in repr(exp)

    def test_expand_bullet_stripping(self):
        response = "- First variant long enough\n* Second variant long enough\n- Third variant"
        llm = _make_llm(response)
        exp = MultiQueryExpander(llm_function=llm, num_queries=3)
        result = exp.expand("question")
        assert all(not q.startswith("-") for q in result)
        assert all(not q.startswith("*") for q in result)


# ---------------------------------------------------------------------------
# StepBackExpander
# ---------------------------------------------------------------------------


class TestStepBackExpander:
    def test_init_defaults(self):
        llm = _make_llm("general question")
        exp = StepBackExpander(llm_function=llm)
        assert "{query}" in exp.prompt_template

    def test_custom_prompt_template(self):
        llm = _make_llm("general")
        exp = StepBackExpander(
            llm_function=llm,
            prompt_template="Generalize this: {query}",
        )
        assert exp.prompt_template == "Generalize this: {query}"

    def test_expand_returns_string(self):
        llm = _make_llm("What is the general concept?")
        exp = StepBackExpander(llm_function=llm)
        result = exp.expand("What was the sales of Company X in Q3 2023?")
        assert isinstance(result, str)
        assert "general" in result.lower()

    def test_expand_strips_whitespace(self):
        llm = _make_llm("  some general question  ")
        exp = StepBackExpander(llm_function=llm)
        result = exp.expand("specific question")
        assert result == "some general question"

    def test_expand_calls_llm_once(self):
        llm = _make_llm("general")
        exp = StepBackExpander(llm_function=llm)
        exp.expand("specific query")
        assert llm.call_count == 1

    def test_expand_formats_query_in_prompt(self):
        llm = _make_llm("general")
        exp = StepBackExpander(llm_function=llm)
        exp.expand("my specific question")
        assert "my specific question" in llm.call_args[0][0]

    def test_repr(self):
        llm = _make_llm("")
        exp = StepBackExpander(llm_function=llm)
        assert "StepBackExpander" in repr(exp)

    def test_default_prompt_mentions_general(self):
        llm = _make_llm("")
        exp = StepBackExpander(llm_function=llm)
        assert "general" in exp.prompt_template.lower()


# ---------------------------------------------------------------------------
# BaseQueryExpander abstract
# ---------------------------------------------------------------------------


class TestBaseQueryExpanderAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseQueryExpander()  # type: ignore[abstract]

    def test_concrete_subclass_works(self):
        class _Concrete(BaseQueryExpander):
            def expand(self, query):
                return query + " expanded"

        exp = _Concrete()
        assert exp.expand("hello") == "hello expanded"
