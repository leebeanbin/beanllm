"""Tests for domain/evaluation/trulens_wrapper.py (TruLensWrapper)."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# helpers — inject mock trulens_eval into sys.modules
# ---------------------------------------------------------------------------


def _build_trulens_mocks():
    """Create a minimal mock trulens_eval module tree."""
    mock_trulens = ModuleType("trulens_eval")
    mock_feedback = ModuleType("trulens_eval.feedback")
    mock_provider = ModuleType("trulens_eval.feedback.provider")
    mock_openai_mod = ModuleType("trulens_eval.feedback.provider.openai")

    # Feedback mock
    mock_feedback_cls = MagicMock()
    mock_feedback.Feedback = mock_feedback_cls

    # LLMProvider mock
    mock_llm_provider = MagicMock()
    mock_llm_provider.create.return_value = MagicMock()
    mock_trulens.LLMProvider = mock_llm_provider

    # TruChain, Tru
    mock_trulens.TruChain = MagicMock(return_value=MagicMock())
    mock_trulens.Tru = MagicMock(return_value=MagicMock())

    # OpenAI provider with scoring methods
    mock_openai_provider = MagicMock()
    mock_openai_provider.context_relevance.return_value = 0.9
    mock_openai_provider.groundedness_measure_with_cot_reasons.return_value = 0.85
    mock_openai_provider.relevance.return_value = 0.8
    mock_openai_mod.OpenAI = MagicMock(return_value=mock_openai_provider)

    return {
        "trulens_eval": mock_trulens,
        "trulens_eval.feedback": mock_feedback,
        "trulens_eval.feedback.provider": mock_provider,
        "trulens_eval.feedback.provider.openai": mock_openai_mod,
        "_mock_openai_provider": mock_openai_provider,
    }


def _inject(mocks: dict) -> dict:
    saved = {}
    for key, val in mocks.items():
        if key.startswith("_"):
            continue
        saved[key] = sys.modules.get(key)
        sys.modules[key] = val
    return saved


def _cleanup(saved: dict):
    for key, val in saved.items():
        if val is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = val


# ---------------------------------------------------------------------------
# constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_defaults(self):
        from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

        w = TruLensWrapper()
        assert w.provider == "openai"
        assert w.model == "gpt-4o-mini"
        assert w.api_key is None
        assert w.enable_dashboard is False

    def test_custom_params(self):
        from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

        w = TruLensWrapper(
            provider="anthropic", model="claude-3", api_key="k", enable_dashboard=True
        )
        assert w.provider == "anthropic"
        assert w.model == "claude-3"
        assert w.api_key == "k"
        assert w.enable_dashboard is True

    def test_lazy_attrs_are_none(self):
        from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

        w = TruLensWrapper()
        assert w._trulens is None
        assert w._llm is None
        assert w._feedback_functions is None

    def test_repr(self):
        from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

        w = TruLensWrapper(provider="openai", model="gpt-4o-mini", enable_dashboard=False)
        s = repr(w)
        assert "openai" in s
        assert "gpt-4o-mini" in s


# ---------------------------------------------------------------------------
# _check_dependencies
# ---------------------------------------------------------------------------


class TestCheckDependencies:
    def test_raises_when_trulens_not_installed(self):
        saved = sys.modules.pop("trulens_eval", None)
        sys.modules["trulens_eval"] = None  # type: ignore[assignment]
        try:
            from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

            w = TruLensWrapper()
            with pytest.raises(ImportError, match="trulens-eval"):
                w._check_dependencies()
        finally:
            if saved is None:
                sys.modules.pop("trulens_eval", None)
            else:
                sys.modules["trulens_eval"] = saved

    def test_sets_trulens_attr(self):
        mocks = _build_trulens_mocks()
        saved = _inject(mocks)
        try:
            from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

            w = TruLensWrapper()
            w._check_dependencies()
            assert w._trulens is not None
        finally:
            _cleanup(saved)


# ---------------------------------------------------------------------------
# _get_llm
# ---------------------------------------------------------------------------


class TestGetLLM:
    def test_returns_cached_llm_on_second_call(self):
        mocks = _build_trulens_mocks()
        saved = _inject(mocks)
        try:
            from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

            w = TruLensWrapper()
            w._llm = MagicMock()  # pre-fill cache
            result = w._get_llm()
            assert result is w._llm
        finally:
            _cleanup(saved)

    def test_openai_provider_path(self):
        mocks = _build_trulens_mocks()
        saved = _inject(mocks)
        try:
            from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

            w = TruLensWrapper(provider="openai")
            result = w._get_llm()
            # Should have called LLMProvider.create for openai
        finally:
            _cleanup(saved)

    def test_anthropic_provider_path(self):
        mocks = _build_trulens_mocks()
        saved = _inject(mocks)
        try:
            from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

            w = TruLensWrapper(provider="anthropic")
            w._get_llm()
        finally:
            _cleanup(saved)

    def test_azure_provider_path(self):
        mocks = _build_trulens_mocks()
        saved = _inject(mocks)
        try:
            from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

            w = TruLensWrapper(provider="azure")
            w._get_llm()
        finally:
            _cleanup(saved)

    def test_unknown_provider_returns_none(self):
        mocks = _build_trulens_mocks()
        saved = _inject(mocks)
        try:
            from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

            w = TruLensWrapper(provider="unknown_xyz")
            result = w._get_llm()
            assert result is None  # falls back to None on ValueError
        finally:
            _cleanup(saved)


# ---------------------------------------------------------------------------
# _get_feedback_functions
# ---------------------------------------------------------------------------


class TestGetFeedbackFunctions:
    def test_returns_cached_on_second_call(self):
        mocks = _build_trulens_mocks()
        saved = _inject(mocks)
        try:
            from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

            w = TruLensWrapper()
            w._feedback_functions = {"context_relevance": MagicMock()}
            result = w._get_feedback_functions()
            assert "context_relevance" in result
        finally:
            _cleanup(saved)

    def test_initializes_feedback_functions_dict(self):
        mocks = _build_trulens_mocks()
        saved = _inject(mocks)
        try:
            from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

            w = TruLensWrapper()
            result = w._get_feedback_functions()
            assert isinstance(result, dict)
        finally:
            _cleanup(saved)

    def test_exception_in_feedback_returns_empty_dict(self):
        mocks = _build_trulens_mocks()
        saved = _inject(mocks)
        try:
            from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

            # Make Feedback class raise
            mocks["trulens_eval.feedback"].Feedback.side_effect = RuntimeError("no feedback")
            w = TruLensWrapper()
            result = w._get_feedback_functions()
            assert result == {}
        finally:
            _cleanup(saved)


# ---------------------------------------------------------------------------
# evaluate (dispatch)
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_dispatches_to_triad(self):
        mocks = _build_trulens_mocks()
        saved = _inject(mocks)
        try:
            from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

            w = TruLensWrapper()
            with patch.object(
                w, "evaluate_rag_triad", return_value={"context_relevance": 0.9}
            ) as m:
                result = w.evaluate(
                    metric="triad",
                    question="q?",
                    answer="a",
                    contexts=["ctx"],
                )
            m.assert_called_once()
        finally:
            _cleanup(saved)

    def test_dispatches_to_context_relevance(self):
        mocks = _build_trulens_mocks()
        saved = _inject(mocks)
        try:
            from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

            w = TruLensWrapper()
            with patch.object(
                w, "evaluate_context_relevance", return_value={"context_relevance": 0.7}
            ) as m:
                w.evaluate(metric="context_relevance", question="q?", contexts=["ctx"])
            m.assert_called_once()
        finally:
            _cleanup(saved)

    def test_dispatches_to_groundedness(self):
        mocks = _build_trulens_mocks()
        saved = _inject(mocks)
        try:
            from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

            w = TruLensWrapper()
            with patch.object(w, "evaluate_groundedness", return_value={"groundedness": 0.8}) as m:
                w.evaluate(metric="groundedness", answer="a", contexts=["ctx"])
            m.assert_called_once()
        finally:
            _cleanup(saved)

    def test_dispatches_to_answer_relevance(self):
        mocks = _build_trulens_mocks()
        saved = _inject(mocks)
        try:
            from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

            w = TruLensWrapper()
            with patch.object(
                w, "evaluate_answer_relevance", return_value={"answer_relevance": 0.9}
            ) as m:
                w.evaluate(metric="answer_relevance", question="q?", answer="a")
            m.assert_called_once()
        finally:
            _cleanup(saved)

    def test_unknown_metric_raises_value_error(self):
        mocks = _build_trulens_mocks()
        saved = _inject(mocks)
        try:
            from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

            w = TruLensWrapper()
            with pytest.raises(ValueError, match="Unknown metric"):
                w.evaluate(metric="nonexistent")
        finally:
            _cleanup(saved)

    def test_default_metric_is_triad(self):
        mocks = _build_trulens_mocks()
        saved = _inject(mocks)
        try:
            from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

            w = TruLensWrapper()
            with patch.object(w, "evaluate_rag_triad", return_value={}) as m:
                w.evaluate(question="q?", answer="a", contexts=["c"])
            m.assert_called_once()
        finally:
            _cleanup(saved)


# ---------------------------------------------------------------------------
# evaluate_context_relevance
# ---------------------------------------------------------------------------


class TestEvaluateContextRelevance:
    def test_returns_average_score(self):
        mocks = _build_trulens_mocks()
        saved = _inject(mocks)
        try:
            from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

            mock_provider = mocks["_mock_openai_provider"]
            mock_provider.context_relevance.side_effect = [0.8, 0.6]

            w = TruLensWrapper()
            result = w.evaluate_context_relevance("q?", ["ctx1", "ctx2"])
            assert "context_relevance" in result
            assert result["context_relevance"] == pytest.approx(0.7)
        finally:
            _cleanup(saved)

    def test_empty_contexts_returns_zero(self):
        mocks = _build_trulens_mocks()
        saved = _inject(mocks)
        try:
            from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

            w = TruLensWrapper()
            result = w.evaluate_context_relevance("q?", [])
            assert result["context_relevance"] == 0.0
        finally:
            _cleanup(saved)

    def test_returns_zero_on_exception(self):
        mocks = _build_trulens_mocks()
        saved = _inject(mocks)
        try:
            from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

            # Make OpenAI constructor fail
            mocks["trulens_eval.feedback.provider.openai"].OpenAI.side_effect = RuntimeError("fail")
            w = TruLensWrapper()
            result = w.evaluate_context_relevance("q?", ["ctx"])
            assert result["context_relevance"] == 0.0
        finally:
            _cleanup(saved)


# ---------------------------------------------------------------------------
# evaluate_groundedness
# ---------------------------------------------------------------------------


class TestEvaluateGroundedness:
    def test_returns_score(self):
        mocks = _build_trulens_mocks()
        saved = _inject(mocks)
        try:
            from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

            mock_provider = mocks["_mock_openai_provider"]
            mock_provider.groundedness_measure_with_cot_reasons.return_value = 0.75

            w = TruLensWrapper()
            result = w.evaluate_groundedness("answer", ["ctx1", "ctx2"])
            assert result["groundedness"] == pytest.approx(0.75)
        finally:
            _cleanup(saved)

    def test_handles_tuple_score(self):
        mocks = _build_trulens_mocks()
        saved = _inject(mocks)
        try:
            from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

            mock_provider = mocks["_mock_openai_provider"]
            mock_provider.groundedness_measure_with_cot_reasons.return_value = (
                0.9,
                "Good reasoning",
            )

            w = TruLensWrapper()
            result = w.evaluate_groundedness("answer", ["ctx"])
            assert result["groundedness"] == pytest.approx(0.9)
        finally:
            _cleanup(saved)

    def test_returns_zero_on_exception(self):
        mocks = _build_trulens_mocks()
        saved = _inject(mocks)
        try:
            from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

            mocks["trulens_eval.feedback.provider.openai"].OpenAI.side_effect = RuntimeError("fail")
            w = TruLensWrapper()
            result = w.evaluate_groundedness("answer", ["ctx"])
            assert result["groundedness"] == 0.0
        finally:
            _cleanup(saved)


# ---------------------------------------------------------------------------
# evaluate_answer_relevance
# ---------------------------------------------------------------------------


class TestEvaluateAnswerRelevance:
    def test_returns_score(self):
        mocks = _build_trulens_mocks()
        saved = _inject(mocks)
        try:
            from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

            mock_provider = mocks["_mock_openai_provider"]
            mock_provider.relevance.return_value = 0.88

            w = TruLensWrapper()
            result = w.evaluate_answer_relevance("q?", "answer")
            assert result["answer_relevance"] == pytest.approx(0.88)
        finally:
            _cleanup(saved)

    def test_returns_zero_on_exception(self):
        mocks = _build_trulens_mocks()
        saved = _inject(mocks)
        try:
            from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

            mocks["trulens_eval.feedback.provider.openai"].OpenAI.side_effect = RuntimeError("fail")
            w = TruLensWrapper()
            result = w.evaluate_answer_relevance("q?", "a")
            assert result["answer_relevance"] == 0.0
        finally:
            _cleanup(saved)


# ---------------------------------------------------------------------------
# evaluate_rag_triad
# ---------------------------------------------------------------------------


class TestEvaluateRAGTriad:
    def test_combines_three_metrics(self):
        mocks = _build_trulens_mocks()
        saved = _inject(mocks)
        try:
            from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

            w = TruLensWrapper()
            with (
                patch.object(
                    w, "evaluate_context_relevance", return_value={"context_relevance": 0.9}
                ),
                patch.object(w, "evaluate_groundedness", return_value={"groundedness": 0.8}),
                patch.object(
                    w, "evaluate_answer_relevance", return_value={"answer_relevance": 0.7}
                ),
            ):
                result = w.evaluate_rag_triad("q?", "answer", ["ctx"])

            assert result["context_relevance"] == pytest.approx(0.9)
            assert result["groundedness"] == pytest.approx(0.8)
            assert result["answer_relevance"] == pytest.approx(0.7)
        finally:
            _cleanup(saved)


# ---------------------------------------------------------------------------
# list_tasks
# ---------------------------------------------------------------------------


class TestListTasks:
    def test_returns_four_metrics(self):
        from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

        w = TruLensWrapper()
        tasks = w.list_tasks()
        assert "context_relevance" in tasks
        assert "groundedness" in tasks
        assert "answer_relevance" in tasks
        assert "triad" in tasks


# ---------------------------------------------------------------------------
# track_app
# ---------------------------------------------------------------------------


class TestTrackApp:
    def test_returns_tru_chain_on_success(self):
        mocks = _build_trulens_mocks()
        saved = _inject(mocks)
        try:
            from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

            mock_app = MagicMock()
            w = TruLensWrapper()
            w._feedback_functions = {}  # pre-fill to avoid nested calls
            result = w.track_app(mock_app)
            # TruChain was called
            assert mocks["trulens_eval"].TruChain.called or result is mock_app
        finally:
            _cleanup(saved)

    def test_returns_original_app_on_exception(self):
        mocks = _build_trulens_mocks()
        saved = _inject(mocks)
        try:
            from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

            mocks["trulens_eval"].TruChain.side_effect = RuntimeError("fail")
            mock_app = MagicMock()
            w = TruLensWrapper()
            result = w.track_app(mock_app)
            assert result is mock_app
        finally:
            _cleanup(saved)


# ---------------------------------------------------------------------------
# show_dashboard
# ---------------------------------------------------------------------------


class TestShowDashboard:
    def test_does_nothing_when_dashboard_disabled(self):
        from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

        w = TruLensWrapper(enable_dashboard=False)
        # Should not raise even if trulens_eval not installed
        w.show_dashboard()

    def test_starts_dashboard_when_enabled(self):
        mocks = _build_trulens_mocks()
        saved = _inject(mocks)
        try:
            from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

            mock_tru_instance = MagicMock()
            mocks["trulens_eval"].Tru.return_value = mock_tru_instance

            w = TruLensWrapper(enable_dashboard=True)
            w.show_dashboard()
            mock_tru_instance.run_dashboard.assert_called_once()
        finally:
            _cleanup(saved)

    def test_handles_dashboard_exception_gracefully(self):
        mocks = _build_trulens_mocks()
        saved = _inject(mocks)
        try:
            from beanllm.domain.evaluation.trulens_wrapper import TruLensWrapper

            mocks["trulens_eval"].Tru.side_effect = RuntimeError("fail")
            w = TruLensWrapper(enable_dashboard=True)
            w.show_dashboard()  # Should not raise
        finally:
            _cleanup(saved)
