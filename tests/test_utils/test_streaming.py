"""
Streaming 테스트 - 스트리밍 유틸리티 테스트
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from beanllm.utils.streaming import (
    StreamStats,
    StreamResponse,
    stream_response,
    stream_collect,
    stream_print,
    StreamBuffer,
    pretty_stream,
)


class TestStreamStats:
    """StreamStats 테스트"""

    @pytest.fixture
    def stream_stats(self):
        """StreamStats 인스턴스"""
        return StreamStats()

    def test_stream_stats_initialization(self, stream_stats):
        """StreamStats 초기화 테스트"""
        assert stream_stats.chunks == 0
        assert stream_stats.total_tokens == 0
        assert stream_stats.start_time is None or isinstance(stream_stats.start_time, datetime)

    def test_stream_stats_duration(self, stream_stats):
        """StreamStats duration 계산 테스트"""
        stream_stats.start_time = datetime.now()
        stream_stats.end_time = datetime.now()

        assert isinstance(stream_stats.duration, float)
        assert stream_stats.duration >= 0

    def test_stream_stats_tokens_per_second(self, stream_stats):
        """StreamStats tokens_per_second 계산 테스트"""
        stream_stats.start_time = datetime.now()
        stream_stats.end_time = datetime.now()
        stream_stats.total_tokens = 100

        assert isinstance(stream_stats.tokens_per_second, float)


class TestStreamResponse:
    """stream_response 테스트"""

    @pytest.mark.asyncio
    async def test_stream_response(self):
        """스트리밍 응답 테스트"""

        async def mock_stream():
            yield "chunk1"
            yield "chunk2"

        result = await stream_response(
            mock_stream(),
            return_output=True,
            display=False,
        )

        assert result is not None
        assert result.content == "chunk1chunk2"
        assert isinstance(result.stats, StreamStats)

    @pytest.mark.asyncio
    async def test_stream_collect(self):
        """스트리밍 수집 테스트"""

        async def mock_stream():
            yield "chunk1"
            yield "chunk2"

        content = await stream_collect(mock_stream())

        assert content == "chunk1chunk2"

    @pytest.mark.asyncio
    async def test_stream_response_with_on_chunk(self):
        """on_chunk 콜백 테스트"""
        callback_calls = []

        async def mock_stream():
            yield "chunk1"
            yield "chunk2"

        def on_chunk(chunk):
            callback_calls.append(chunk)

        result = await stream_response(
            mock_stream(),
            return_output=True,
            display=False,
            on_chunk=on_chunk,
        )

        assert result is not None
        assert len(callback_calls) == 2
        assert callback_calls == ["chunk1", "chunk2"]

    @pytest.mark.asyncio
    async def test_stream_response_with_stats(self):
        """통계 표시 테스트"""

        async def mock_stream():
            yield "chunk1 "
            yield "chunk2 "

        result = await stream_response(
            mock_stream(),
            return_output=True,
            display=False,
            show_stats=True,
        )

        assert result is not None
        assert result.stats.chunks == 2
        assert result.stats.total_tokens > 0

    @pytest.mark.asyncio
    async def test_stream_response_no_return(self):
        """출력 반환 없이 테스트"""

        async def mock_stream():
            yield "chunk1"
            yield "chunk2"

        result = await stream_response(
            mock_stream(),
            return_output=False,
            display=False,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_stream_response_display_plain(self):
        """일반 print 출력 테스트"""

        async def mock_stream():
            yield "chunk1"
            yield "chunk2"

        result = await stream_response(
            mock_stream(),
            return_output=True,
            display=True,
            use_rich=False,
        )

        assert result is not None
        assert result.content == "chunk1chunk2"

    @pytest.mark.asyncio
    async def test_stream_print(self):
        """stream_print 테스트"""

        async def mock_stream():
            yield "chunk1"
            yield "chunk2"

        content = await stream_print(mock_stream(), markdown=False)

        assert content == "chunk1chunk2"

    @pytest.mark.asyncio
    async def test_pretty_stream(self):
        """pretty_stream 테스트"""

        async def mock_stream():
            yield "chunk1"
            yield "chunk2"

        result = await pretty_stream(mock_stream(), title="Test")

        assert result is not None
        assert result.content == "chunk1chunk2"
        assert isinstance(result.stats, StreamStats)


class TestStreamBuffer:
    """StreamBuffer 테스트"""

    @pytest.fixture
    def stream_buffer(self):
        """StreamBuffer 인스턴스"""
        return StreamBuffer()

    @pytest.mark.asyncio
    async def test_stream_buffer_add_chunk(self, stream_buffer):
        """버퍼에 청크 추가 테스트"""
        await stream_buffer.add_chunk("stream1", "chunk1")
        await stream_buffer.add_chunk("stream1", "chunk2")

        content = stream_buffer.get_content("stream1")
        assert content == "chunk1chunk2"

    @pytest.mark.asyncio
    async def test_stream_buffer_multiple_streams(self, stream_buffer):
        """여러 스트림 처리 테스트"""
        await stream_buffer.add_chunk("stream1", "chunk1")
        await stream_buffer.add_chunk("stream2", "chunk2")

        assert stream_buffer.get_content("stream1") == "chunk1"
        assert stream_buffer.get_content("stream2") == "chunk2"

    def test_stream_buffer_clear(self, stream_buffer):
        """버퍼 초기화 테스트"""
        import asyncio

        async def setup():
            await stream_buffer.add_chunk("stream1", "chunk1")
            stream_buffer.clear("stream1")
            return stream_buffer.get_content("stream1")

        content = asyncio.run(setup())
        assert content == ""

    def test_stream_buffer_get_all(self, stream_buffer):
        """모든 버퍼 내용 가져오기 테스트"""
        import asyncio

        async def setup():
            await stream_buffer.add_chunk("stream1", "chunk1")
            await stream_buffer.add_chunk("stream2", "chunk2")
            return stream_buffer.get_all()

        all_buffers = asyncio.run(setup())
        assert "stream1" in all_buffers
        assert "stream2" in all_buffers
        assert all_buffers["stream1"] == "chunk1"
        assert all_buffers["stream2"] == "chunk2"

    @pytest.mark.asyncio
    async def test_stream_response_with_on_chunk(self):
        """on_chunk 콜백 테스트"""
        callback_calls = []

        async def mock_stream():
            yield "chunk1"
            yield "chunk2"

        def on_chunk(chunk):
            callback_calls.append(chunk)

        result = await stream_response(
            mock_stream(),
            return_output=True,
            display=False,
            on_chunk=on_chunk,
        )

        assert result is not None
        assert len(callback_calls) == 2
        assert callback_calls == ["chunk1", "chunk2"]

    @pytest.mark.asyncio
    async def test_stream_response_with_stats(self):
        """통계 표시 테스트"""

        async def mock_stream():
            yield "chunk1 "
            yield "chunk2 "

        result = await stream_response(
            mock_stream(),
            return_output=True,
            display=False,
            show_stats=True,
        )

        assert result is not None
        assert result.stats.chunks == 2
        assert result.stats.total_tokens > 0

    @pytest.mark.asyncio
    async def test_stream_response_no_return(self):
        """출력 반환 없이 테스트"""

        async def mock_stream():
            yield "chunk1"
            yield "chunk2"

        result = await stream_response(
            mock_stream(),
            return_output=False,
            display=False,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_stream_response_display_plain(self):
        """일반 print 출력 테스트"""

        async def mock_stream():
            yield "chunk1"
            yield "chunk2"

        result = await stream_response(
            mock_stream(),
            return_output=True,
            display=True,
            use_rich=False,
        )

        assert result is not None
        assert result.content == "chunk1chunk2"

    @pytest.mark.asyncio
    async def test_stream_print(self):
        """stream_print 테스트"""

        async def mock_stream():
            yield "chunk1"
            yield "chunk2"

        content = await stream_print(mock_stream(), markdown=False)

        assert content == "chunk1chunk2"

    @pytest.mark.asyncio
    async def test_pretty_stream(self):
        """pretty_stream 테스트"""

        async def mock_stream():
            yield "chunk1"
            yield "chunk2"

        result = await pretty_stream(mock_stream(), title="Test")

        assert result is not None
        assert result.content == "chunk1chunk2"
        assert isinstance(result.stats, StreamStats)


class TestStreamBuffer:
    """StreamBuffer 테스트"""

    @pytest.fixture
    def stream_buffer(self):
        """StreamBuffer 인스턴스"""
        return StreamBuffer()

    @pytest.mark.asyncio
    async def test_stream_buffer_add_chunk(self, stream_buffer):
        """버퍼에 청크 추가 테스트"""
        await stream_buffer.add_chunk("stream1", "chunk1")
        await stream_buffer.add_chunk("stream1", "chunk2")

        content = stream_buffer.get_content("stream1")
        assert content == "chunk1chunk2"

    @pytest.mark.asyncio
    async def test_stream_buffer_multiple_streams(self, stream_buffer):
        """여러 스트림 처리 테스트"""
        await stream_buffer.add_chunk("stream1", "chunk1")
        await stream_buffer.add_chunk("stream2", "chunk2")

        assert stream_buffer.get_content("stream1") == "chunk1"
        assert stream_buffer.get_content("stream2") == "chunk2"

    def test_stream_buffer_clear(self, stream_buffer):
        """버퍼 초기화 테스트"""
        import asyncio

        async def setup():
            await stream_buffer.add_chunk("stream1", "chunk1")
            stream_buffer.clear("stream1")
            return stream_buffer.get_content("stream1")

        content = asyncio.run(setup())
        assert content == ""

    def test_stream_buffer_get_all(self, stream_buffer):
        """모든 버퍼 내용 가져오기 테스트"""
        import asyncio

        async def setup():
            await stream_buffer.add_chunk("stream1", "chunk1")
            await stream_buffer.add_chunk("stream2", "chunk2")
            return stream_buffer.get_all()

        all_buffers = asyncio.run(setup())
        assert "stream1" in all_buffers
        assert "stream2" in all_buffers
        assert all_buffers["stream1"] == "chunk1"
        assert all_buffers["stream2"] == "chunk2"

    @pytest.mark.asyncio
    async def test_stream_response_with_on_chunk(self):
        """on_chunk 콜백 테스트"""
        callback_calls = []

        async def mock_stream():
            yield "chunk1"
            yield "chunk2"

        def on_chunk(chunk):
            callback_calls.append(chunk)

        result = await stream_response(
            mock_stream(),
            return_output=True,
            display=False,
            on_chunk=on_chunk,
        )

        assert result is not None
        assert len(callback_calls) == 2
        assert callback_calls == ["chunk1", "chunk2"]

    @pytest.mark.asyncio
    async def test_stream_response_with_stats(self):
        """통계 표시 테스트"""

        async def mock_stream():
            yield "chunk1 "
            yield "chunk2 "

        result = await stream_response(
            mock_stream(),
            return_output=True,
            display=False,
            show_stats=True,
        )

        assert result is not None
        assert result.stats.chunks == 2
        assert result.stats.total_tokens > 0

    @pytest.mark.asyncio
    async def test_stream_response_no_return(self):
        """출력 반환 없이 테스트"""

        async def mock_stream():
            yield "chunk1"
            yield "chunk2"

        result = await stream_response(
            mock_stream(),
            return_output=False,
            display=False,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_stream_response_display_plain(self):
        """일반 print 출력 테스트"""

        async def mock_stream():
            yield "chunk1"
            yield "chunk2"

        result = await stream_response(
            mock_stream(),
            return_output=True,
            display=True,
            use_rich=False,
        )

        assert result is not None
        assert result.content == "chunk1chunk2"

    @pytest.mark.asyncio
    async def test_stream_print(self):
        """stream_print 테스트"""

        async def mock_stream():
            yield "chunk1"
            yield "chunk2"

        content = await stream_print(mock_stream(), markdown=False)

        assert content == "chunk1chunk2"

    @pytest.mark.asyncio
    async def test_pretty_stream(self):
        """pretty_stream 테스트"""

        async def mock_stream():
            yield "chunk1"
            yield "chunk2"

        result = await pretty_stream(mock_stream(), title="Test")

        assert result is not None
        assert result.content == "chunk1chunk2"
        assert isinstance(result.stats, StreamStats)


class TestStreamBuffer:
    """StreamBuffer 테스트"""

    @pytest.fixture
    def stream_buffer(self):
        """StreamBuffer 인스턴스"""
        return StreamBuffer()

    @pytest.mark.asyncio
    async def test_stream_buffer_add_chunk(self, stream_buffer):
        """버퍼에 청크 추가 테스트"""
        await stream_buffer.add_chunk("stream1", "chunk1")
        await stream_buffer.add_chunk("stream1", "chunk2")

        content = stream_buffer.get_content("stream1")
        assert content == "chunk1chunk2"

    @pytest.mark.asyncio
    async def test_stream_buffer_multiple_streams(self, stream_buffer):
        """여러 스트림 처리 테스트"""
        await stream_buffer.add_chunk("stream1", "chunk1")
        await stream_buffer.add_chunk("stream2", "chunk2")

        assert stream_buffer.get_content("stream1") == "chunk1"
        assert stream_buffer.get_content("stream2") == "chunk2"

    def test_stream_buffer_clear(self, stream_buffer):
        """버퍼 초기화 테스트"""
        import asyncio

        async def setup():
            await stream_buffer.add_chunk("stream1", "chunk1")
            stream_buffer.clear("stream1")
            return stream_buffer.get_content("stream1")

        content = asyncio.run(setup())
        assert content == ""

    def test_stream_buffer_get_all(self, stream_buffer):
        """모든 버퍼 내용 가져오기 테스트"""
        import asyncio

        async def setup():
            await stream_buffer.add_chunk("stream1", "chunk1")
            await stream_buffer.add_chunk("stream2", "chunk2")
            return stream_buffer.get_all()

        all_buffers = asyncio.run(setup())
        assert "stream1" in all_buffers
        assert "stream2" in all_buffers
        assert all_buffers["stream1"] == "chunk1"
        assert all_buffers["stream2"] == "chunk2"
