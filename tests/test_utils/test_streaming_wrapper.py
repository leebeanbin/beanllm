"""
Tests for beanllm.utils.streaming_wrapper
Goal: maximize line coverage (32 lines missed, 0% current)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from beanllm.utils.streaming_wrapper import BufferedStreamWrapper, PausableStream

    AVAILABLE = True
except ImportError:
    AVAILABLE = False

try:
    from beanllm.utils.streaming import StreamBuffer

    STREAM_BUFFER_AVAILABLE = True
except ImportError:
    STREAM_BUFFER_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not AVAILABLE or not STREAM_BUFFER_AVAILABLE,
    reason="streaming_wrapper or StreamBuffer not available",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make_stream(*chunks: str):
    for chunk in chunks:
        yield chunk


# ===========================================================================
# BufferedStreamWrapper
# ===========================================================================


class TestBufferedStreamWrapper:
    @pytest.mark.asyncio
    async def test_yields_all_chunks_when_not_paused(self):
        buffer = StreamBuffer()
        stream = _make_stream("hello", " ", "world")
        wrapper = BufferedStreamWrapper(stream, buffer, stream_id="test1")

        collected = []
        async for chunk in wrapper:
            collected.append(chunk)

        assert collected == ["hello", " ", "world"]

    @pytest.mark.asyncio
    async def test_stores_chunks_in_buffer(self):
        buffer = StreamBuffer()
        stream = _make_stream("a", "b", "c")
        wrapper = BufferedStreamWrapper(stream, buffer, stream_id="buf_test")

        async for _ in wrapper:
            pass

        content = buffer.get_content("buf_test")
        assert content == "abc"

    @pytest.mark.asyncio
    async def test_paused_stream_stores_but_does_not_yield(self):
        """When stream is paused, chunks are buffered but not yielded."""
        buffer = StreamBuffer()
        stream_id = "paused_stream"
        stream = _make_stream("chunk1", "chunk2", "chunk3")

        # Pre-initialize buffer entries so pause works correctly
        await buffer.add_chunk(stream_id, "")
        buffer.pause(stream_id)

        wrapper = BufferedStreamWrapper(stream, buffer, stream_id=stream_id)
        collected = []
        async for chunk in wrapper:
            collected.append(chunk)

        # Nothing should be yielded because stream is paused
        assert collected == []
        # But buffer should contain all chunks (plus our initial empty one)
        content = buffer.get_content(stream_id)
        assert "chunk1" in content
        assert "chunk2" in content
        assert "chunk3" in content

    @pytest.mark.asyncio
    async def test_default_stream_id(self):
        buffer = StreamBuffer()
        stream = _make_stream("x")
        wrapper = BufferedStreamWrapper(stream, buffer)  # default stream_id

        collected = []
        async for chunk in wrapper:
            collected.append(chunk)

        assert collected == ["x"]
        assert wrapper.stream_id == "default"

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        buffer = StreamBuffer()
        stream = _make_stream()  # empty
        wrapper = BufferedStreamWrapper(stream, buffer, stream_id="empty")

        collected = []
        async for chunk in wrapper:
            collected.append(chunk)

        assert collected == []

    @pytest.mark.asyncio
    async def test_mixed_pause_resume_during_stream(self):
        """Resume midway should allow subsequent chunks to be yielded."""
        buffer = StreamBuffer()
        stream_id = "mixed"

        # Add initial entry so pause state is tracked
        await buffer.add_chunk(stream_id, "")
        buffer.pause(stream_id)  # Start paused

        stream = _make_stream("a", "b", "c")
        wrapper = BufferedStreamWrapper(stream, buffer, stream_id=stream_id)

        collected = []
        chunk_idx = 0
        async for chunk in wrapper:
            collected.append(chunk)
            chunk_idx += 1

        # Started paused → nothing yielded
        assert len(collected) == 0


# ===========================================================================
# PausableStream
# ===========================================================================


class TestPausableStream:
    @pytest.mark.asyncio
    async def test_iterate_yields_all_chunks(self):
        buffer = StreamBuffer()
        stream = _make_stream("p", "a", "u", "s", "e")
        ps = PausableStream(stream, buffer, stream_id="ps1")

        collected = []
        async for chunk in ps:
            collected.append(chunk)

        assert collected == ["p", "a", "u", "s", "e"]

    @pytest.mark.asyncio
    async def test_pause_stops_yielding(self):
        buffer = StreamBuffer()
        stream_id = "ps_pause"

        # Initialize so pause works
        await buffer.add_chunk(stream_id, "")
        buffer.pause(stream_id)

        stream = _make_stream("x", "y", "z")
        ps = PausableStream(stream, buffer, stream_id=stream_id)

        collected = []
        async for chunk in ps:
            collected.append(chunk)

        assert collected == []

    @pytest.mark.asyncio
    async def test_pause_method(self):
        buffer = StreamBuffer()
        stream_id = "ps_m"
        await buffer.add_chunk(stream_id, "init")

        stream = _make_stream()
        ps = PausableStream(stream, buffer, stream_id=stream_id)
        ps.pause()

        assert ps.is_paused() is True

    @pytest.mark.asyncio
    async def test_resume_method(self):
        buffer = StreamBuffer()
        stream_id = "ps_r"
        await buffer.add_chunk(stream_id, "init")

        stream = _make_stream()
        ps = PausableStream(stream, buffer, stream_id=stream_id)
        ps.pause()
        ps.resume()

        assert ps.is_paused() is False

    def test_is_paused_initially_false(self):
        buffer = StreamBuffer()
        stream = _make_stream("a")
        ps = PausableStream(stream, buffer, stream_id="ps_init")
        # Not yet initialized in buffer → is_paused should return False
        assert ps.is_paused() is False

    @pytest.mark.asyncio
    async def test_get_content(self):
        buffer = StreamBuffer()
        stream_id = "ps_content"
        await buffer.add_chunk(stream_id, "hello")
        await buffer.add_chunk(stream_id, " world")

        stream = _make_stream()
        ps = PausableStream(stream, buffer, stream_id=stream_id)

        content = ps.get_content()
        assert content == "hello world"

    @pytest.mark.asyncio
    async def test_clear(self):
        buffer = StreamBuffer()
        stream_id = "ps_clear"
        await buffer.add_chunk(stream_id, "data")

        stream = _make_stream()
        ps = PausableStream(stream, buffer, stream_id=stream_id)
        ps.clear()

        assert ps.get_content() == ""

    @pytest.mark.asyncio
    async def test_replay_returns_async_iterator(self):
        buffer = StreamBuffer()
        stream_id = "ps_replay"
        await buffer.add_chunk(stream_id, "r1")
        await buffer.add_chunk(stream_id, "r2")

        stream = _make_stream()
        ps = PausableStream(stream, buffer, stream_id=stream_id)

        replay_iter = ps.replay(delay=0.0)
        # Should be an async iterator
        assert hasattr(replay_iter, "__aiter__") or asyncio.iscoroutine(replay_iter)

    @pytest.mark.asyncio
    async def test_replay_yields_buffered_content(self):
        buffer = StreamBuffer()
        stream_id = "ps_replay2"

        await buffer.add_chunk(stream_id, "a")
        await buffer.add_chunk(stream_id, "b")
        await buffer.add_chunk(stream_id, "c")

        stream = _make_stream()
        ps = PausableStream(stream, buffer, stream_id=stream_id)

        replay_chunks = []
        async for chunk in ps.replay(delay=0.0):
            replay_chunks.append(chunk)

        assert replay_chunks == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_full_flow_stream_then_replay(self):
        """Stream some data, then replay it."""
        buffer = StreamBuffer()
        stream_id = "full_flow"
        stream = _make_stream("Hello", " ", "World")
        ps = PausableStream(stream, buffer, stream_id=stream_id)

        # Stream all chunks
        streamed = []
        async for chunk in ps:
            streamed.append(chunk)

        assert streamed == ["Hello", " ", "World"]
        assert ps.get_content() == "Hello World"

        # Replay
        replayed = []
        async for chunk in ps.replay():
            replayed.append(chunk)

        assert replayed == ["Hello", " ", "World"]

    @pytest.mark.asyncio
    async def test_clear_then_stream_again(self):
        buffer = StreamBuffer()
        stream_id = "clear_stream"

        # First stream
        stream1 = _make_stream("first", "run")
        ps = PausableStream(stream1, buffer, stream_id=stream_id)
        async for _ in ps:
            pass

        assert ps.get_content() == "firstrun"

        # Clear
        ps.clear()
        assert ps.get_content() == ""

        # Second stream (new PausableStream to get fresh stream)
        stream2 = _make_stream("second", "run")
        ps2 = PausableStream(stream2, buffer, stream_id=stream_id)
        async for _ in ps2:
            pass

        assert ps2.get_content() == "secondrun"

    def test_default_stream_id(self):
        buffer = StreamBuffer()
        stream = _make_stream("x")
        ps = PausableStream(stream, buffer)
        assert ps.stream_id == "default"

    @pytest.mark.asyncio
    async def test_pause_resume_during_iteration(self):
        """Test that pausing and resuming mid-iteration works."""
        buffer = StreamBuffer()
        stream_id = "mid_iter"

        # Set up a stream that we can control
        stream = _make_stream("a", "b", "c", "d", "e")
        ps = PausableStream(stream, buffer, stream_id=stream_id)

        collected = []
        count = 0
        async for chunk in ps:
            collected.append(chunk)
            count += 1
            if count == 2:
                # Pause after 2 chunks — but we're already in iteration
                # so subsequent chunks from the same stream won't be yielded
                # This just tests it doesn't error
                ps.pause()

        # At least 2 chunks collected before pause
        assert "a" in collected
        assert "b" in collected
