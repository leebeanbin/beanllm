"""Tests for filter_thinking_stream and strip_thinking_tokens."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, List

import pytest


async def _collect(stream: AsyncIterator[str]) -> str:
    return "".join([c async for c in stream])


async def _gen(*chunks: str) -> AsyncIterator[str]:
    for c in chunks:
        yield c


class TestStripThinkingTokens:
    def test_strips_single_block(self):
        from beanllm.utils.streaming.streaming import strip_thinking_tokens

        assert strip_thinking_tokens("Hello <thinking>step</thinking> world") == "Hello  world"

    def test_strips_multiline_block(self):
        from beanllm.utils.streaming.streaming import strip_thinking_tokens

        assert strip_thinking_tokens("<thinking>\nline1\nline2\n</thinking>done") == "done"

    def test_strips_multiple_blocks(self):
        from beanllm.utils.streaming.streaming import strip_thinking_tokens

        result = strip_thinking_tokens("A <thinking>x</thinking> B <thinking>y</thinking> C")
        assert result == "A  B  C"

    def test_no_thinking_unchanged(self):
        from beanllm.utils.streaming.streaming import strip_thinking_tokens

        assert strip_thinking_tokens("plain text") == "plain text"

    def test_empty_string(self):
        from beanllm.utils.streaming.streaming import strip_thinking_tokens

        assert strip_thinking_tokens("") == ""

    def test_only_thinking_block(self):
        from beanllm.utils.streaming.streaming import strip_thinking_tokens

        assert strip_thinking_tokens("<thinking>everything</thinking>") == ""


class TestFilterThinkingStream:
    def test_complete_block_in_one_chunk(self):
        from beanllm.utils.streaming.streaming import filter_thinking_stream

        result = asyncio.run(_collect(filter_thinking_stream(_gen("<thinking>r</thinking>Answer"))))
        assert result == "Answer"

    def test_tag_split_across_chunks(self):
        from beanllm.utils.streaming.streaming import filter_thinking_stream

        result = asyncio.run(
            _collect(filter_thinking_stream(_gen("Hello <think", "ing>step</thinking>", " world")))
        )
        assert result == "Hello  world"

    def test_close_tag_split_across_chunks(self):
        from beanllm.utils.streaming.streaming import filter_thinking_stream

        result = asyncio.run(
            _collect(filter_thinking_stream(_gen("<thinking>step</think", "ing> done")))
        )
        assert result == " done"

    def test_no_thinking_passes_through(self):
        from beanllm.utils.streaming.streaming import filter_thinking_stream

        result = asyncio.run(_collect(filter_thinking_stream(_gen("plain ", "text"))))
        assert result == "plain text"

    def test_unclosed_thinking_block_dropped(self):
        from beanllm.utils.streaming.streaming import filter_thinking_stream

        result = asyncio.run(_collect(filter_thinking_stream(_gen("text <thinking>half"))))
        assert result == "text "

    def test_multiple_complete_blocks(self):
        from beanllm.utils.streaming.streaming import filter_thinking_stream

        result = asyncio.run(
            _collect(
                filter_thinking_stream(_gen("A <thinking>x</thinking> B <thinking>y</thinking> C"))
            )
        )
        assert result == "A  B  C"

    def test_empty_stream(self):
        from beanllm.utils.streaming.streaming import filter_thinking_stream

        async def empty() -> AsyncIterator[str]:
            return
            yield  # make it an async generator

        result = asyncio.run(_collect(filter_thinking_stream(empty())))
        assert result == ""
