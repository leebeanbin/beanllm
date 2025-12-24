"""
Streaming Wrapper - 버퍼링된 스트리밍 래퍼
"""

from typing import AsyncIterator

from .streaming import StreamingBuffer


class BufferedStreamWrapper:
    """
    버퍼링된 스트리밍 래퍼

    스트림을 버퍼에 저장하면서 동시에 yield
    """

    def __init__(
        self,
        stream: AsyncIterator[str],
        buffer: StreamingBuffer,
        stream_id: str = "default",
    ):
        self.stream = stream
        self.buffer = buffer
        self.stream_id = stream_id

    async def __aiter__(self):
        """스트리밍 반복"""
        async for chunk in self.stream:
            # 버퍼에 저장 (일시정지 중이어도 저장)
            await self.buffer.add_chunk(self.stream_id, chunk)

            # 일시정지 중이면 yield하지 않음
            if not self.buffer.is_stream_paused(self.stream_id):
                yield chunk
            # 일시정지 중이면 버퍼에만 저장하고 yield하지 않음


class PausableStream:
    """
    일시정지 가능한 스트림

    사용자가 일시정지/재개를 제어할 수 있음
    """

    def __init__(
        self,
        stream: AsyncIterator[str],
        buffer: StreamingBuffer,
        stream_id: str = "default",
    ):
        self._wrapper = BufferedStreamWrapper(stream, buffer, stream_id)
        self.buffer = buffer
        self.stream_id = stream_id

    async def __aiter__(self):
        """스트리밍 반복"""
        async for chunk in self._wrapper:
            yield chunk

    def pause(self):
        """일시정지"""
        self.buffer.pause(self.stream_id)

    def resume(self):
        """재개"""
        self.buffer.resume(self.stream_id)

    def is_paused(self) -> bool:
        """일시정지 상태 확인"""
        return self.buffer.is_stream_paused(self.stream_id)

    def replay(self, delay: float = 0.0) -> AsyncIterator[str]:
        """
        재생

        Args:
            delay: 청크 간 지연 시간 (초)

        Returns:
            AsyncIterator[str]: 재생 스트림
        """
        return self.buffer.replay(self.stream_id, delay)

    def get_content(self) -> str:
        """전체 내용 가져오기"""
        return self.buffer.get_content(self.stream_id)

    def clear(self):
        """버퍼 초기화"""
        self.buffer.clear(self.stream_id)
