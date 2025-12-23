"""
Tracer 테스트 - 추적 유틸리티 테스트
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
import json
from pathlib import Path

from llmkit.utils.tracer import (
    Tracer,
    get_tracer,
    enable_tracing,
    TraceSpan,
    Trace,
)


class TestTracer:
    """Tracer 테스트"""

    @pytest.fixture
    def tracer(self):
        """Tracer 인스턴스"""
        return Tracer(project_name="test")

    def test_tracer_start_trace(self, tracer):
        """추적 시작 테스트"""
        trace = tracer.start_trace()

        assert trace is not None
        assert trace.trace_id is not None
        assert trace.project_name == "test"

    def test_tracer_start_span(self, tracer):
        """Span 시작 테스트"""
        trace = tracer.start_trace()
        span = tracer.start_span("test_span", provider="openai", model="gpt-4o-mini")

        assert span is not None
        assert span.name == "test_span"

    def test_tracer_end_trace(self, tracer):
        """추적 종료 테스트"""
        trace = tracer.start_trace()
        tracer.end_trace(trace.trace_id)

        retrieved_trace = tracer.get_trace(trace.trace_id)
        assert retrieved_trace is not None
        assert retrieved_trace.end_time is not None

    def test_tracer_get_trace(self, tracer):
        """추적 정보 조회 테스트"""
        trace = tracer.start_trace()
        tracer.end_trace(trace.trace_id)

        retrieved_trace = tracer.get_trace(trace.trace_id)

        assert retrieved_trace is not None
        assert retrieved_trace.trace_id == trace.trace_id

    def test_tracer_span_context_manager(self, tracer):
        """Span 컨텍스트 매니저 테스트"""
        trace = tracer.start_trace()

        with tracer.span("test_span", provider="openai"):
            pass

        assert len(trace.spans) == 1
        assert trace.spans[0].name == "test_span"

    def test_tracer_get_stats(self, tracer):
        """통계 정보 조회 테스트"""
        trace = tracer.start_trace()
        tracer.start_span("test_span")
        tracer.end_span()
        tracer.end_trace(trace.trace_id)

        stats = tracer.get_stats(trace.trace_id)

        assert isinstance(stats, dict)
        assert "total_spans" in stats

    def test_trace_span_to_dict(self, tracer):
        """TraceSpan to_dict 테스트"""
        trace = tracer.start_trace()
        span = tracer.start_span("test_span", provider="openai", model="gpt-4o-mini")
        span.input_tokens = 100
        span.output_tokens = 50
        tracer.end_span()

        span_dict = span.to_dict()
        assert isinstance(span_dict, dict)
        assert "span_id" in span_dict
        assert "name" in span_dict
        assert "duration_ms" in span_dict
        assert "start_time" in span_dict

    def test_trace_to_dict(self, tracer):
        """Trace to_dict 테스트"""
        trace = tracer.start_trace()
        tracer.start_span("test_span")
        tracer.end_span()
        tracer.end_trace(trace.trace_id)

        trace_dict = trace.to_dict()
        assert isinstance(trace_dict, dict)
        assert "trace_id" in trace_dict
        assert "project_name" in trace_dict
        assert "total_duration_ms" in trace_dict
        assert "total_tokens" in trace_dict
        assert "spans" in trace_dict

    def test_tracer_save_trace(self, tracer, tmp_path):
        """추적 저장 테스트"""
        tracer.save_dir = tmp_path
        trace = tracer.start_trace()
        tracer.end_trace(trace.trace_id)

        tracer.save_trace(trace.trace_id, "test_trace.json")

        filepath = tmp_path / "test_trace.json"
        assert filepath.exists()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["trace_id"] == trace.trace_id

    def test_tracer_end_span_with_tokens(self, tracer):
        """토큰 수 포함 스팬 종료 테스트"""
        trace = tracer.start_trace()
        tracer.start_span("test_span")
        tracer.end_span(input_tokens=100, output_tokens=50)

        span = trace.spans[0]
        assert span.input_tokens == 100
        assert span.output_tokens == 50

    def test_tracer_end_span_with_error(self, tracer):
        """에러 포함 스팬 종료 테스트"""
        trace = tracer.start_trace()
        tracer.start_span("test_span")
        tracer.end_span(status="error", error="Test error")

        span = trace.spans[0]
        assert span.status == "error"
        assert span.error == "Test error"

    def test_tracer_nested_spans(self, tracer):
        """중첩 스팬 테스트"""
        trace = tracer.start_trace()
        tracer.start_span("parent_span")
        tracer.start_span("child_span")
        tracer.end_span()
        tracer.end_span()

        assert len(trace.spans) == 2
        assert trace.spans[1].parent_id == trace.spans[0].span_id

    def test_tracer_clear(self, tracer):
        """추적 초기화 테스트"""
        trace = tracer.start_trace()
        tracer.start_span("test_span")
        tracer.clear()

        assert len(tracer.traces) == 0
        assert tracer.current_trace_id is None
        assert len(tracer.span_stack) == 0

    def test_tracer_span_context_manager_with_error(self, tracer):
        """에러 발생 시 스팬 컨텍스트 매니저 테스트"""
        trace = tracer.start_trace()

        try:
            with tracer.span("test_span"):
                raise ValueError("Test error")
        except ValueError:
            pass

        assert len(trace.spans) == 1
        assert trace.spans[0].status == "error"
        assert "Test error" in (trace.spans[0].error or "")

    def test_tracer_auto_save(self, tmp_path):
        """자동 저장 테스트"""
        tracer = Tracer(project_name="test", auto_save=True, save_dir=str(tmp_path))
        trace = tracer.start_trace()
        tracer.end_trace(trace.trace_id)

        # 자동 저장 확인
        files = list(tmp_path.glob("trace_*.json"))
        assert len(files) > 0


class TestTracerFunctions:
    """Tracer 편의 함수 테스트"""

    def test_get_tracer(self):
        """get_tracer 함수 테스트"""
        tracer = get_tracer("test-project")

        assert isinstance(tracer, Tracer)
        assert tracer.project_name == "test-project"

    def test_enable_tracing(self):
        """enable_tracing 함수 테스트"""
        enable_tracing(project_name="test-project", auto_save=False)

        tracer = get_tracer("test-project")
        assert isinstance(tracer, Tracer)

        assert "name" in span_dict
        assert "duration_ms" in span_dict
        assert "start_time" in span_dict

    def test_trace_to_dict(self, tracer):
        """Trace to_dict 테스트"""
        trace = tracer.start_trace()
        tracer.start_span("test_span")
        tracer.end_span()
        tracer.end_trace(trace.trace_id)

        trace_dict = trace.to_dict()
        assert isinstance(trace_dict, dict)
        assert "trace_id" in trace_dict
        assert "project_name" in trace_dict
        assert "total_duration_ms" in trace_dict
        assert "total_tokens" in trace_dict
        assert "spans" in trace_dict

    def test_tracer_save_trace(self, tracer, tmp_path):
        """추적 저장 테스트"""
        tracer.save_dir = tmp_path
        trace = tracer.start_trace()
        tracer.end_trace(trace.trace_id)

        tracer.save_trace(trace.trace_id, "test_trace.json")

        filepath = tmp_path / "test_trace.json"
        assert filepath.exists()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["trace_id"] == trace.trace_id

    def test_tracer_end_span_with_tokens(self, tracer):
        """토큰 수 포함 스팬 종료 테스트"""
        trace = tracer.start_trace()
        tracer.start_span("test_span")
        tracer.end_span(input_tokens=100, output_tokens=50)

        span = trace.spans[0]
        assert span.input_tokens == 100
        assert span.output_tokens == 50

    def test_tracer_end_span_with_error(self, tracer):
        """에러 포함 스팬 종료 테스트"""
        trace = tracer.start_trace()
        tracer.start_span("test_span")
        tracer.end_span(status="error", error="Test error")

        span = trace.spans[0]
        assert span.status == "error"
        assert span.error == "Test error"

    def test_tracer_nested_spans(self, tracer):
        """중첩 스팬 테스트"""
        trace = tracer.start_trace()
        tracer.start_span("parent_span")
        tracer.start_span("child_span")
        tracer.end_span()
        tracer.end_span()

        assert len(trace.spans) == 2
        assert trace.spans[1].parent_id == trace.spans[0].span_id

    def test_tracer_clear(self, tracer):
        """추적 초기화 테스트"""
        trace = tracer.start_trace()
        tracer.start_span("test_span")
        tracer.clear()

        assert len(tracer.traces) == 0
        assert tracer.current_trace_id is None
        assert len(tracer.span_stack) == 0

    def test_tracer_span_context_manager_with_error(self, tracer):
        """에러 발생 시 스팬 컨텍스트 매니저 테스트"""
        trace = tracer.start_trace()

        try:
            with tracer.span("test_span"):
                raise ValueError("Test error")
        except ValueError:
            pass

        assert len(trace.spans) == 1
        assert trace.spans[0].status == "error"
        assert "Test error" in (trace.spans[0].error or "")

    def test_tracer_auto_save(self, tmp_path):
        """자동 저장 테스트"""
        tracer = Tracer(project_name="test", auto_save=True, save_dir=str(tmp_path))
        trace = tracer.start_trace()
        tracer.end_trace(trace.trace_id)

        # 자동 저장 확인
        files = list(tmp_path.glob("trace_*.json"))
        assert len(files) > 0


class TestTracerFunctions:
    """Tracer 편의 함수 테스트"""

    def test_get_tracer(self):
        """get_tracer 함수 테스트"""
        tracer = get_tracer("test-project")

        assert isinstance(tracer, Tracer)
        assert tracer.project_name == "test-project"

    def test_enable_tracing(self):
        """enable_tracing 함수 테스트"""
        enable_tracing(project_name="test-project", auto_save=False)

        tracer = get_tracer("test-project")
        assert isinstance(tracer, Tracer)

        assert "name" in span_dict
        assert "duration_ms" in span_dict
        assert "start_time" in span_dict

    def test_trace_to_dict(self, tracer):
        """Trace to_dict 테스트"""
        trace = tracer.start_trace()
        tracer.start_span("test_span")
        tracer.end_span()
        tracer.end_trace(trace.trace_id)

        trace_dict = trace.to_dict()
        assert isinstance(trace_dict, dict)
        assert "trace_id" in trace_dict
        assert "project_name" in trace_dict
        assert "total_duration_ms" in trace_dict
        assert "total_tokens" in trace_dict
        assert "spans" in trace_dict

    def test_tracer_save_trace(self, tracer, tmp_path):
        """추적 저장 테스트"""
        tracer.save_dir = tmp_path
        trace = tracer.start_trace()
        tracer.end_trace(trace.trace_id)

        tracer.save_trace(trace.trace_id, "test_trace.json")

        filepath = tmp_path / "test_trace.json"
        assert filepath.exists()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["trace_id"] == trace.trace_id

    def test_tracer_end_span_with_tokens(self, tracer):
        """토큰 수 포함 스팬 종료 테스트"""
        trace = tracer.start_trace()
        tracer.start_span("test_span")
        tracer.end_span(input_tokens=100, output_tokens=50)

        span = trace.spans[0]
        assert span.input_tokens == 100
        assert span.output_tokens == 50

    def test_tracer_end_span_with_error(self, tracer):
        """에러 포함 스팬 종료 테스트"""
        trace = tracer.start_trace()
        tracer.start_span("test_span")
        tracer.end_span(status="error", error="Test error")

        span = trace.spans[0]
        assert span.status == "error"
        assert span.error == "Test error"

    def test_tracer_nested_spans(self, tracer):
        """중첩 스팬 테스트"""
        trace = tracer.start_trace()
        tracer.start_span("parent_span")
        tracer.start_span("child_span")
        tracer.end_span()
        tracer.end_span()

        assert len(trace.spans) == 2
        assert trace.spans[1].parent_id == trace.spans[0].span_id

    def test_tracer_clear(self, tracer):
        """추적 초기화 테스트"""
        trace = tracer.start_trace()
        tracer.start_span("test_span")
        tracer.clear()

        assert len(tracer.traces) == 0
        assert tracer.current_trace_id is None
        assert len(tracer.span_stack) == 0

    def test_tracer_span_context_manager_with_error(self, tracer):
        """에러 발생 시 스팬 컨텍스트 매니저 테스트"""
        trace = tracer.start_trace()

        try:
            with tracer.span("test_span"):
                raise ValueError("Test error")
        except ValueError:
            pass

        assert len(trace.spans) == 1
        assert trace.spans[0].status == "error"
        assert "Test error" in (trace.spans[0].error or "")

    def test_tracer_auto_save(self, tmp_path):
        """자동 저장 테스트"""
        tracer = Tracer(project_name="test", auto_save=True, save_dir=str(tmp_path))
        trace = tracer.start_trace()
        tracer.end_trace(trace.trace_id)

        # 자동 저장 확인
        files = list(tmp_path.glob("trace_*.json"))
        assert len(files) > 0


class TestTracerFunctions:
    """Tracer 편의 함수 테스트"""

    def test_get_tracer(self):
        """get_tracer 함수 테스트"""
        tracer = get_tracer("test-project")

        assert isinstance(tracer, Tracer)
        assert tracer.project_name == "test-project"

    def test_enable_tracing(self):
        """enable_tracing 함수 테스트"""
        enable_tracing(project_name="test-project", auto_save=False)

        tracer = get_tracer("test-project")
        assert isinstance(tracer, Tracer)
