"""
Tracer - Request Tracking System
LangSmith 스타일의 요청 추적 시스템
"""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


@dataclass
class TraceSpan:
    """추적 스팬 (단일 요청)"""

    span_id: str
    parent_id: Optional[str]
    name: str
    start_time: datetime
    end_time: Optional[datetime] = None

    # 요청 정보
    provider: Optional[str] = None
    model: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    # 메타데이터
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # 결과
    status: str = "running"  # running, success, error
    error: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        """소요 시간 (밀리초)"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0.0

    def to_dict(self) -> Dict:
        """딕셔너리 변환"""
        d = asdict(self)
        d["start_time"] = self.start_time.isoformat()
        if self.end_time:
            d["end_time"] = self.end_time.isoformat()
        d["duration_ms"] = self.duration_ms
        return d


@dataclass
class Trace:
    """추적 (여러 스팬의 집합)"""

    trace_id: str
    project_name: str
    start_time: datetime
    end_time: Optional[datetime] = None

    spans: List[TraceSpan] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_duration_ms(self) -> float:
        """전체 소요 시간"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0.0

    @property
    def total_tokens(self) -> int:
        """전체 토큰 수"""
        return sum((span.input_tokens or 0) + (span.output_tokens or 0) for span in self.spans)

    def to_dict(self) -> Dict:
        """딕셔너리 변환"""
        return {
            "trace_id": self.trace_id,
            "project_name": self.project_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration_ms": self.total_duration_ms,
            "total_tokens": self.total_tokens,
            "spans": [span.to_dict() for span in self.spans],
            "metadata": self.metadata,
        }


class Tracer:
    """
    요청 추적 시스템

    LangSmith 스타일의 추적 기능:
    - 프로젝트별 추적
    - 계층적 스팬 (nested spans)
    - 토큰 사용량 추적
    - JSON/파일 저장
    - 통계 분석

    Example:
        ```python
        from beanllm import Client
        from beanllm.tracer import Tracer

        # Tracer 초기화
        tracer = Tracer(project_name="my-app")

        # 추적 시작
        trace = tracer.start_trace()

        # 스팬 생성
        with tracer.span("llm-call", provider="openai", model="gpt-4o-mini"):
            client = Client(model="gpt-4o-mini")
            response = await client.chat(messages)

        # 추적 종료
        tracer.end_trace(trace.trace_id)

        # 결과 저장
        tracer.save_trace(trace.trace_id, "trace.json")
        ```
    """

    def __init__(
        self, project_name: str = "default", auto_save: bool = False, save_dir: Optional[str] = None
    ):
        """
        Args:
            project_name: 프로젝트 이름
            auto_save: 자동 저장 여부
            save_dir: 저장 디렉토리
        """
        self.project_name = project_name
        self.auto_save = auto_save
        self.save_dir = Path(save_dir) if save_dir else Path.home() / ".beanllm" / "traces"

        if self.auto_save:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.traces: Dict[str, Trace] = {}
        self.current_trace_id: Optional[str] = None
        self.span_stack: List[str] = []  # 스팬 스택 (nested spans)

    def start_trace(self, metadata: Optional[Dict[str, Any]] = None) -> Trace:
        """새 추적 시작"""
        trace_id = str(uuid.uuid4())
        trace = Trace(
            trace_id=trace_id,
            project_name=self.project_name,
            start_time=datetime.now(),
            metadata=metadata or {},
        )

        self.traces[trace_id] = trace
        self.current_trace_id = trace_id

        logger.debug(f"Started trace: {trace_id}")
        return trace

    def end_trace(self, trace_id: Optional[str] = None):
        """추적 종료"""
        tid = trace_id or self.current_trace_id
        if not tid:
            logger.warning("No active trace to end")
            return

        trace = self.traces.get(tid)
        if not trace:
            logger.warning(f"Trace not found: {tid}")
            return

        trace.end_time = datetime.now()

        if self.auto_save:
            self.save_trace(tid)

        logger.debug(f"Ended trace: {tid} ({trace.total_duration_ms:.2f}ms)")

    def start_span(
        self,
        name: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> TraceSpan:
        """스팬 시작"""
        if not self.current_trace_id:
            logger.warning("No active trace, starting a new one")
            self.start_trace()

        trace = self.traces[self.current_trace_id]

        span_id = str(uuid.uuid4())
        parent_id = self.span_stack[-1] if self.span_stack else None

        span = TraceSpan(
            span_id=span_id,
            parent_id=parent_id,
            name=name,
            start_time=datetime.now(),
            provider=provider,
            model=model,
            metadata=metadata or {},
            tags=tags or [],
        )

        trace.spans.append(span)
        self.span_stack.append(span_id)

        logger.debug(f"Started span: {name} (id: {span_id})")
        return span

    def end_span(
        self,
        status: str = "success",
        error: Optional[str] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ):
        """스팬 종료"""
        if not self.span_stack:
            logger.warning("No active span to end")
            return

        span_id = self.span_stack.pop()
        trace = self.traces[self.current_trace_id]

        # 스팬 찾기
        span = next((s for s in trace.spans if s.span_id == span_id), None)
        if not span:
            logger.warning(f"Span not found: {span_id}")
            return

        span.end_time = datetime.now()
        span.status = status
        span.error = error
        span.input_tokens = input_tokens
        span.output_tokens = output_tokens

        logger.debug(
            f"Ended span: {span.name} ({span.duration_ms:.2f}ms, "
            f"tokens: {(input_tokens or 0) + (output_tokens or 0)})"
        )

    def span(
        self, name: str, provider: Optional[str] = None, model: Optional[str] = None, **kwargs
    ):
        """
        컨텍스트 매니저로 스팬 사용

        Example:
            ```python
            with tracer.span("llm-call", provider="openai"):
                response = await client.chat(messages)
            ```
        """
        return _SpanContext(self, name, provider, model, **kwargs)

    def save_trace(self, trace_id: Optional[str] = None, filename: Optional[str] = None):
        """추적 저장"""
        tid = trace_id or self.current_trace_id
        if not tid:
            logger.warning("No trace to save")
            return

        trace = self.traces.get(tid)
        if not trace:
            logger.warning(f"Trace not found: {tid}")
            return

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trace_{timestamp}_{tid[:8]}.json"

        filepath = self.save_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(trace.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Saved trace to: {filepath}")

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """추적 가져오기"""
        return self.traces.get(trace_id)

    def get_stats(self, trace_id: Optional[str] = None) -> Dict:
        """통계 정보"""
        tid = trace_id or self.current_trace_id
        if not tid:
            return {}

        trace = self.traces.get(tid)
        if not trace:
            return {}

        return {
            "trace_id": trace.trace_id,
            "project_name": trace.project_name,
            "total_duration_ms": trace.total_duration_ms,
            "total_spans": len(trace.spans),
            "total_tokens": trace.total_tokens,
            "success_spans": sum(1 for s in trace.spans if s.status == "success"),
            "error_spans": sum(1 for s in trace.spans if s.status == "error"),
        }

    def clear(self):
        """모든 추적 삭제"""
        self.traces.clear()
        self.current_trace_id = None
        self.span_stack.clear()


class _SpanContext:
    """스팬 컨텍스트 매니저"""

    def __init__(
        self, tracer: Tracer, name: str, provider: Optional[str], model: Optional[str], **kwargs
    ):
        self.tracer = tracer
        self.name = name
        self.provider = provider
        self.model = model
        self.kwargs = kwargs
        self.span = None

    def __enter__(self):
        self.span = self.tracer.start_span(
            self.name, provider=self.provider, model=self.model, **self.kwargs
        )
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.tracer.end_span(status="error", error=str(exc_val))
        else:
            self.tracer.end_span(status="success")


# 전역 Tracer (편의)
_global_tracer: Optional[Tracer] = None


def get_tracer(project_name: str = "default") -> Tracer:
    """전역 Tracer 가져오기"""
    global _global_tracer
    if _global_tracer is None or _global_tracer.project_name != project_name:
        _global_tracer = Tracer(project_name=project_name)
    return _global_tracer


def enable_tracing(
    project_name: str = "default", auto_save: bool = True, save_dir: Optional[str] = None
):
    """
    추적 활성화

    Example:
        ```python
        from beanllm.tracer import enable_tracing

        # 추적 활성화
        enable_tracing(project_name="my-app", auto_save=True)

        # 이제 Client 사용 시 자동 추적
        client = Client(model="gpt-4o-mini")
        response = await client.chat(messages)  # 자동 추적됨
        ```
    """
    global _global_tracer
    _global_tracer = Tracer(project_name=project_name, auto_save=auto_save, save_dir=save_dir)
    logger.info(f"Tracing enabled for project: {project_name}")
