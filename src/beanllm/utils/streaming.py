"""
Streaming Helpers
실시간 스트리밍 출력 헬퍼
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Dict, List, Optional

if TYPE_CHECKING:
    pass

try:
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None
    Live = None
    Markdown = None
    Panel = None
    Text = None

try:
    from .logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)

if RICH_AVAILABLE:
    console = Console()
else:
    console = None


@dataclass
class StreamStats:
    """스트리밍 통계"""

    total_tokens: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    chunks: int = 0

    @property
    def duration(self) -> float:
        """소요 시간 (초)"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    @property
    def tokens_per_second(self) -> float:
        """초당 토큰 수"""
        if self.duration > 0:
            return self.total_tokens / self.duration
        return 0.0


@dataclass
class StreamResponse:
    """스트리밍 응답 결과"""

    content: str
    stats: StreamStats
    metadata: dict = field(default_factory=dict)


async def stream_response(
    stream: AsyncIterator[str],
    return_output: bool = True,
    display: bool = True,
    use_rich: bool = True,
    markdown: bool = False,
    show_stats: bool = False,
    panel_title: Optional[str] = None,
    on_chunk: Optional[Callable[[str], Any]] = None,
    enable_buffer: bool = False,
    buffer: Optional["StreamBuffer"] = None,
    stream_id: Optional[str] = None,
) -> Optional[StreamResponse]:
    """
    스트리밍 응답 출력 헬퍼

    참고: LangChain과 TeddyNote의 stream_response에서 영감을 받았습니다.
    beanllm의 개선된 기능:
    - Rich 기반 아름다운 출력
    - 마크다운 렌더링
    - 통계 정보 (토큰 수, 속도)
    - 커스텀 콜백
    - Panel 래핑
    - 버퍼링 지원 (일시정지/재개/재생)

    Args:
        stream: AsyncIterator[str] - 스트림 소스
        return_output: 출력 내용 반환 여부
        display: 화면 출력 여부
        use_rich: rich 라이브러리 사용 여부
        markdown: 마크다운 렌더링 여부
        show_stats: 통계 정보 표시
        panel_title: Panel 제목
        on_chunk: 청크마다 호출할 콜백
        enable_buffer: 버퍼링 활성화
        buffer: StreamingBuffer 인스턴스 (None이면 자동 생성)
        stream_id: 스트림 ID

    Returns:
        StreamResponse | None: 응답 결과 (return_output=True인 경우)

    Example:
        ```python
        from beanllm import Client, stream_response

        client = Client(model="gpt-4o-mini")
        stream = client.stream_chat(messages, temperature=0.7)

        # 기본 출력
        await stream_response(stream)

        # 마크다운 + 통계
        result = await stream_response(
            stream,
            markdown=True,
            show_stats=True,
            panel_title="GPT-4o-mini"
        )
        print(f"Tokens: {result.stats.total_tokens}")
        print(f"Speed: {result.stats.tokens_per_second:.2f} tok/s")
        ```
    """
    stats = StreamStats(start_time=datetime.now())
    collected = []

    # Rich 사용 가능 여부 확인
    if use_rich and not RICH_AVAILABLE:
        logger.warning("Rich library not available. Falling back to plain output.")
        use_rich = False

    try:
        if display and use_rich and panel_title and console:
            # Rich Panel + Live 업데이트
            with Live(console=console, refresh_per_second=10) as live:
                current_text = ""
                async for chunk in stream:
                    current_text += chunk
                    collected.append(chunk)
                    stats.chunks += 1

                    if on_chunk:
                        on_chunk(chunk)

                    # Live 업데이트
                    if markdown:
                        content = Markdown(current_text)
                    else:
                        content = Text(current_text)

                    live.update(
                        Panel(
                            content,
                            title=f"[bold cyan]{panel_title}[/bold cyan]",
                            border_style="cyan",
                        )
                    )

        elif display and use_rich and console:
            # Rich 출력 (Panel 없음)
            current_text = ""
            async for chunk in stream:
                current_text += chunk
                collected.append(chunk)
                stats.chunks += 1

                if on_chunk:
                    on_chunk(chunk)

                # 점진적 출력
                console.print(chunk, end="", markup=False)

            console.print()  # 줄바꿈

        elif display:
            # 일반 print 출력
            async for chunk in stream:
                collected.append(chunk)
                stats.chunks += 1

                if on_chunk:
                    on_chunk(chunk)

                print(chunk, end="", flush=True)

            print()  # 줄바꿈

        else:
            # 출력 없음, 수집만
            async for chunk in stream:
                collected.append(chunk)
                stats.chunks += 1

                if on_chunk:
                    on_chunk(chunk)

        stats.end_time = datetime.now()

        # 버퍼링된 경우 버퍼에서도 가져오기
        if enable_buffer and buffer:
            buffered_content = buffer.get_content(stream_id)
            final_content = buffered_content if buffered_content else "".join(collected)
        else:
            final_content = "".join(collected)

        # 토큰 수 추정 (공백 기준)
        stats.total_tokens = len(final_content.split())

        # 통계 표시
        if show_stats and display:
            _display_stats(stats)

        if return_output:
            return StreamResponse(content=final_content, stats=stats, metadata={})

        return None

    except Exception as e:
        logger.error(f"Stream error: {e}")
        raise


def _display_stats(stats: StreamStats):
    """통계 정보 표시"""
    if not RICH_AVAILABLE or not console:
        # Plain text fallback
        print(f"\nDuration: {stats.duration:.2f}s")
        print(f"Tokens: {stats.total_tokens}")
        print(f"Speed: {stats.tokens_per_second:.2f} tok/s")
        print(f"Chunks: {stats.chunks}")
        return

    stats_panel = Panel(
        f"""[bold cyan]Duration:[/bold cyan] {stats.duration:.2f}s
[bold cyan]Tokens:[/bold cyan] {stats.total_tokens}
[bold cyan]Speed:[/bold cyan] {stats.tokens_per_second:.2f} tok/s
[bold cyan]Chunks:[/bold cyan] {stats.chunks}""",
        title="[bold yellow]📊 Statistics[/bold yellow]",
        border_style="yellow",
        expand=False,
    )
    console.print()
    console.print(stats_panel)


async def stream_print(
    stream: AsyncIterator[str], markdown: bool = False, panel_title: Optional[str] = None
) -> str:
    """
    간단한 스트리밍 출력 (짧은 버전)

    Example:
        ```python
        content = await stream_print(stream, markdown=True)
        ```
    """
    result = await stream_response(
        stream,
        return_output=True,
        display=True,
        use_rich=True,
        markdown=markdown,
        panel_title=panel_title,
    )
    return result.content if result else ""


async def stream_collect(stream: AsyncIterator[str]) -> str:
    """
    스트리밍 수집만 (출력 없음)

    Example:
        ```python
        content = await stream_collect(stream)
        ```
    """
    result = await stream_response(stream, return_output=True, display=False)
    return result.content if result else ""


class StreamBuffer:
    """
    스트리밍 버퍼
    여러 스트림을 동시에 처리
    일시정지, 재개, 재생 기능 지원
    """

    def __init__(self, max_size: int = 10000):
        self.buffers: Dict[str, List[str]] = {}
        self.max_size = max_size
        self.is_paused: Dict[str, bool] = {}  # 스트림별 일시정지 상태
        self._lock = asyncio.Lock()

    async def add_chunk(self, stream_id: str, chunk: str):
        """청크 추가"""
        async with self._lock:
            if stream_id not in self.buffers:
                self.buffers[stream_id] = []
                self.is_paused[stream_id] = False

            # 일시정지 중이어도 버퍼에는 저장
            self.buffers[stream_id].append(chunk)

            # 최대 크기 제한
            if len(self.buffers[stream_id]) > self.max_size:
                # 오래된 항목 제거 (FIFO)
                self.buffers[stream_id] = self.buffers[stream_id][-self.max_size :]

    def pause(self, stream_id: str):
        """일시정지"""
        if stream_id in self.is_paused:
            self.is_paused[stream_id] = True

    def resume(self, stream_id: str):
        """재개"""
        if stream_id in self.is_paused:
            self.is_paused[stream_id] = False

    def is_stream_paused(self, stream_id: str) -> bool:
        """일시정지 상태 확인"""
        return self.is_paused.get(stream_id, False)

    async def replay(
        self,
        stream_id: str,
        delay: float = 0.0,  # 청크 간 지연 시간 (초)
    ) -> AsyncIterator[str]:
        """
        재생 (버퍼된 내용을 다시 스트리밍)

        Args:
            stream_id: 스트림 ID
            delay: 청크 간 지연 시간 (원본 속도 재현)

        Yields:
            str: 청크
        """
        async with self._lock:
            chunks = self.buffers.get(stream_id, []).copy()

        for chunk in chunks:
            yield chunk
            if delay > 0:
                await asyncio.sleep(delay)

    def get_content(self, stream_id: str) -> str:
        """전체 내용 가져오기"""
        return "".join(self.buffers.get(stream_id, []))

    def clear(self, stream_id: str):
        """버퍼 초기화"""
        if stream_id in self.buffers:
            del self.buffers[stream_id]
        if stream_id in self.is_paused:
            del self.is_paused[stream_id]

    def get_all(self) -> dict:
        """모든 버퍼 내용"""
        return {stream_id: "".join(chunks) for stream_id, chunks in self.buffers.items()}


# 편의 함수
async def pretty_stream(stream: AsyncIterator[str], title: str = "Response") -> StreamResponse:
    """
    예쁜 스트리밍 출력 (모든 기능 활성화)

    Example:
        ```python
        from beanllm import Client
        from beanllm.streaming import pretty_stream

        client = Client(model="gpt-4o-mini")
        stream = client.stream_chat(messages)
        result = await pretty_stream(stream, title="GPT-4o-mini")
        ```
    """
    return await stream_response(
        stream,
        return_output=True,
        display=True,
        use_rich=True,
        markdown=True,
        show_stats=True,
        panel_title=title,
    )
