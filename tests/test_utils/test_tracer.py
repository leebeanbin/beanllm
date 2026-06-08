"""Tests for utils/tracer.py."""

import json
from datetime import datetime, timedelta

import pytest

from beanllm.utils.tracer import (
    Trace,
    Tracer,
    TraceSpan,
    enable_tracing,
    get_tracer,
)

# ---------------------------------------------------------------------------
# TraceSpan
# ---------------------------------------------------------------------------


class TestTraceSpan:
    def _span(self, **kwargs) -> TraceSpan:
        defaults = dict(
            span_id="s1",
            parent_id=None,
            name="test",
            start_time=datetime.now(),
        )
        defaults.update(kwargs)
        return TraceSpan(**defaults)

    def test_duration_ms_no_end_time(self):
        span = self._span()
        assert span.duration_ms == 0.0

    def test_duration_ms_with_end_time(self):
        start = datetime.now()
        end = start + timedelta(milliseconds=123)
        span = self._span(start_time=start, end_time=end)
        assert span.duration_ms == pytest.approx(123.0, abs=1.0)

    def test_to_dict_includes_fields(self):
        span = self._span(provider="openai", model="gpt-4o")
        d = span.to_dict()
        assert d["name"] == "test"
        assert d["provider"] == "openai"
        assert d["model"] == "gpt-4o"
        assert "duration_ms" in d
        assert "start_time" in d

    def test_to_dict_end_time_iso(self):
        start = datetime.now()
        end = start + timedelta(seconds=1)
        span = self._span(start_time=start, end_time=end)
        d = span.to_dict()
        assert "T" in d["end_time"]  # ISO format

    def test_default_status_running(self):
        span = self._span()
        assert span.status == "running"

    def test_metadata_default_empty(self):
        span = self._span()
        assert span.metadata == {}

    def test_tags_default_empty(self):
        span = self._span()
        assert span.tags == []


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------


class TestTrace:
    def _trace(self, **kwargs) -> Trace:
        defaults = dict(
            trace_id="t1",
            project_name="test-project",
            start_time=datetime.now(),
        )
        defaults.update(kwargs)
        return Trace(**defaults)

    def test_total_duration_no_end(self):
        trace = self._trace()
        assert trace.total_duration_ms == 0.0

    def test_total_duration_with_end(self):
        start = datetime.now()
        end = start + timedelta(milliseconds=500)
        trace = self._trace(start_time=start, end_time=end)
        assert trace.total_duration_ms == pytest.approx(500.0, abs=5.0)

    def test_total_tokens_no_spans(self):
        trace = self._trace()
        assert trace.total_tokens == 0

    def test_total_tokens_with_spans(self):
        span = TraceSpan(
            span_id="s1",
            parent_id=None,
            name="s",
            start_time=datetime.now(),
            input_tokens=10,
            output_tokens=20,
        )
        trace = self._trace(spans=[span])
        assert trace.total_tokens == 30

    def test_total_tokens_none_tokens(self):
        span = TraceSpan(
            span_id="s1",
            parent_id=None,
            name="s",
            start_time=datetime.now(),
            input_tokens=None,
            output_tokens=None,
        )
        trace = self._trace(spans=[span])
        assert trace.total_tokens == 0

    def test_to_dict_basic(self):
        trace = self._trace()
        d = trace.to_dict()
        assert d["trace_id"] == "t1"
        assert d["project_name"] == "test-project"
        assert "total_duration_ms" in d
        assert isinstance(d["spans"], list)

    def test_to_dict_no_end_time_is_none(self):
        trace = self._trace()
        d = trace.to_dict()
        assert d["end_time"] is None


# ---------------------------------------------------------------------------
# Tracer — start_trace / end_trace
# ---------------------------------------------------------------------------


class TestTracerStartEnd:
    def setup_method(self):
        self.tracer = Tracer(project_name="test")

    def test_start_trace_returns_trace(self):
        trace = self.tracer.start_trace()
        assert isinstance(trace, Trace)
        assert trace.project_name == "test"

    def test_start_trace_sets_current_trace_id(self):
        trace = self.tracer.start_trace()
        assert self.tracer.current_trace_id == trace.trace_id

    def test_start_trace_stores_in_traces(self):
        trace = self.tracer.start_trace()
        assert trace.trace_id in self.tracer.traces

    def test_start_trace_with_metadata(self):
        trace = self.tracer.start_trace(metadata={"env": "test"})
        assert trace.metadata["env"] == "test"

    def test_end_trace_sets_end_time(self):
        trace = self.tracer.start_trace()
        self.tracer.end_trace()
        assert self.tracer.traces[trace.trace_id].end_time is not None

    def test_end_trace_with_explicit_id(self):
        trace = self.tracer.start_trace()
        self.tracer.end_trace(trace.trace_id)
        assert self.tracer.traces[trace.trace_id].end_time is not None

    def test_end_trace_no_active_trace_no_crash(self):
        self.tracer.end_trace()  # should not raise

    def test_end_trace_unknown_id_no_crash(self):
        self.tracer.end_trace("nonexistent-id")  # should not raise

    def test_get_trace_returns_trace(self):
        trace = self.tracer.start_trace()
        found = self.tracer.get_trace(trace.trace_id)
        assert found is trace

    def test_get_trace_unknown_returns_none(self):
        assert self.tracer.get_trace("nope") is None

    def test_clear_removes_all(self):
        self.tracer.start_trace()
        self.tracer.clear()
        assert self.tracer.traces == {}
        assert self.tracer.current_trace_id is None


# ---------------------------------------------------------------------------
# Tracer — span management
# ---------------------------------------------------------------------------


class TestTracerSpans:
    def setup_method(self):
        self.tracer = Tracer(project_name="test")
        self.trace = self.tracer.start_trace()

    def test_start_span_returns_span(self):
        span = self.tracer.start_span("my-span")
        assert isinstance(span, TraceSpan)
        assert span.name == "my-span"

    def test_start_span_added_to_trace(self):
        self.tracer.start_span("my-span")
        assert len(self.trace.spans) == 1

    def test_start_span_provider_model(self):
        span = self.tracer.start_span("llm", provider="openai", model="gpt-4o")
        assert span.provider == "openai"
        assert span.model == "gpt-4o"

    def test_start_span_no_trace_creates_one(self):
        tracer = Tracer(project_name="fresh")
        span = tracer.start_span("auto-span")
        assert span is not None
        assert tracer.current_trace_id is not None

    def test_nested_spans_set_parent(self):
        parent = self.tracer.start_span("parent")
        child = self.tracer.start_span("child")
        assert child.parent_id == parent.span_id

    def test_end_span_sets_status_success(self):
        self.tracer.start_span("s1")
        self.tracer.end_span(status="success")
        assert self.trace.spans[0].status == "success"

    def test_end_span_sets_error(self):
        self.tracer.start_span("s1")
        self.tracer.end_span(status="error", error="boom")
        assert self.trace.spans[0].error == "boom"

    def test_end_span_sets_tokens(self):
        self.tracer.start_span("s1")
        self.tracer.end_span(input_tokens=10, output_tokens=20)
        span = self.trace.spans[0]
        assert span.input_tokens == 10
        assert span.output_tokens == 20

    def test_end_span_no_active_span_no_crash(self):
        self.tracer.end_span()  # nothing on stack — should not raise

    def test_end_span_sets_end_time(self):
        self.tracer.start_span("s1")
        self.tracer.end_span()
        assert self.trace.spans[0].end_time is not None


# ---------------------------------------------------------------------------
# Tracer — span() context manager
# ---------------------------------------------------------------------------


class TestSpanContextManager:
    def setup_method(self):
        self.tracer = Tracer(project_name="test")
        self.tracer.start_trace()

    def test_context_manager_success(self):
        with self.tracer.span("cm-span") as span:
            assert span.name == "cm-span"
        assert span.status == "success"

    def test_context_manager_error_on_exception(self):
        with pytest.raises(ValueError):
            with self.tracer.span("err-span") as span:
                raise ValueError("test error")
        assert span.status == "error"
        assert "test error" in span.error

    def test_context_manager_provider_passed(self):
        with self.tracer.span("s", provider="anthropic", model="claude-3") as span:
            pass
        assert span.provider == "anthropic"
        assert span.model == "claude-3"


# ---------------------------------------------------------------------------
# Tracer — stats
# ---------------------------------------------------------------------------


class TestTracerStats:
    def setup_method(self):
        self.tracer = Tracer(project_name="test")

    def test_stats_no_trace_returns_empty(self):
        assert self.tracer.get_stats() == {}

    def test_stats_basic(self):
        self.tracer.start_trace()
        self.tracer.start_span("s1")
        self.tracer.end_span(status="success")
        self.tracer.end_trace()
        stats = self.tracer.get_stats()
        assert stats["total_spans"] == 1
        assert stats["success_spans"] == 1
        assert stats["error_spans"] == 0

    def test_stats_counts_errors(self):
        self.tracer.start_trace()
        self.tracer.start_span("s1")
        self.tracer.end_span(status="error", error="boom")
        stats = self.tracer.get_stats()
        assert stats["error_spans"] == 1

    def test_stats_unknown_trace_id(self):
        assert self.tracer.get_stats("nope") == {}

    def test_stats_total_tokens(self):
        self.tracer.start_trace()
        self.tracer.start_span("s1")
        self.tracer.end_span(input_tokens=5, output_tokens=10)
        stats = self.tracer.get_stats()
        assert stats["total_tokens"] == 15


# ---------------------------------------------------------------------------
# Tracer — save_trace
# ---------------------------------------------------------------------------


class TestTracerSave:
    def test_save_trace_creates_file(self, tmp_path):
        tracer = Tracer(project_name="test", save_dir=str(tmp_path))
        trace = tracer.start_trace()
        tracer.end_trace()
        tracer.save_trace(trace.trace_id, filename="out.json")
        output = tmp_path / "out.json"
        assert output.exists()
        data = json.loads(output.read_text())
        assert data["trace_id"] == trace.trace_id

    def test_save_trace_auto_filename(self, tmp_path):
        tracer = Tracer(project_name="test", save_dir=str(tmp_path))
        trace = tracer.start_trace()
        tracer.end_trace()
        tracer.save_trace(trace.trace_id)
        files = list(tmp_path.glob("trace_*.json"))
        assert len(files) == 1

    def test_save_trace_no_trace_no_crash(self, tmp_path):
        tracer = Tracer(project_name="test", save_dir=str(tmp_path))
        tracer.save_trace()  # should not raise

    def test_save_trace_unknown_id_no_crash(self, tmp_path):
        tracer = Tracer(project_name="test", save_dir=str(tmp_path))
        tracer.save_trace("nonexistent")  # should not raise

    def test_auto_save_on_end_trace(self, tmp_path):
        tracer = Tracer(project_name="test", auto_save=True, save_dir=str(tmp_path))
        tracer.start_trace()
        tracer.end_trace()
        files = list(tmp_path.glob("trace_*.json"))
        assert len(files) == 1


# ---------------------------------------------------------------------------
# Global tracer helpers
# ---------------------------------------------------------------------------


class TestGlobalTracer:
    def test_get_tracer_returns_tracer(self):
        t = get_tracer("my-project")
        assert isinstance(t, Tracer)
        assert t.project_name == "my-project"

    def test_get_tracer_same_project_reuses(self):
        t1 = get_tracer("proj-reuse")
        t2 = get_tracer("proj-reuse")
        assert t1 is t2

    def test_get_tracer_different_project_creates_new(self):
        t1 = get_tracer("proj-x1")
        t2 = get_tracer("proj-y1")
        assert t1 is not t2

    def test_enable_tracing_sets_global(self, tmp_path):
        enable_tracing(project_name="enabled-proj", auto_save=False, save_dir=str(tmp_path))
        t = get_tracer("enabled-proj")
        assert t.project_name == "enabled-proj"
