"""Tests for domain/loaders/core/text.py — TextLoader."""

from pathlib import Path
from unittest.mock import patch

import pytest

from beanllm.domain.loaders.core.text import TextLoader
from beanllm.domain.loaders.types import Document


def _make_file(tmp_path, name="test.txt", content="Hello world"):
    f = tmp_path / name
    f.write_text(content, encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestTextLoaderInit:
    def test_stores_file_path(self, tmp_path):
        f = _make_file(tmp_path)
        loader = TextLoader(f, validate_path=False)
        assert loader.file_path == Path(f)

    def test_string_path_converted_to_path(self, tmp_path):
        f = _make_file(tmp_path)
        loader = TextLoader(str(f), validate_path=False)
        assert isinstance(loader.file_path, Path)

    def test_default_encoding_is_utf8(self, tmp_path):
        f = _make_file(tmp_path)
        loader = TextLoader(f, validate_path=False)
        assert loader.encoding == "utf-8"

    def test_custom_encoding(self, tmp_path):
        f = _make_file(tmp_path)
        loader = TextLoader(f, encoding="latin-1", validate_path=False)
        assert loader.encoding == "latin-1"

    def test_autodetect_encoding_default_true(self, tmp_path):
        f = _make_file(tmp_path)
        loader = TextLoader(f, validate_path=False)
        assert loader.autodetect_encoding is True

    def test_use_mmap_default_none(self, tmp_path):
        f = _make_file(tmp_path)
        loader = TextLoader(f, validate_path=False)
        assert loader.use_mmap is None

    def test_explicit_use_mmap(self, tmp_path):
        f = _make_file(tmp_path)
        loader = TextLoader(f, use_mmap=True, validate_path=False)
        assert loader.use_mmap is True


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


class TestTextLoaderLoad:
    def test_load_returns_list_with_one_document(self, tmp_path):
        f = _make_file(tmp_path, content="Sample text")
        loader = TextLoader(f, validate_path=False)
        docs = loader.load()
        assert len(docs) == 1
        assert isinstance(docs[0], Document)

    def test_load_content_matches_file(self, tmp_path):
        f = _make_file(tmp_path, content="Line1\nLine2\nLine3")
        loader = TextLoader(f, validate_path=False)
        docs = loader.load()
        assert "Line1" in docs[0].content
        assert "Line2" in docs[0].content

    def test_load_metadata_has_source(self, tmp_path):
        f = _make_file(tmp_path)
        loader = TextLoader(f, validate_path=False)
        docs = loader.load()
        assert "source" in docs[0].metadata
        assert str(f) in docs[0].metadata["source"]

    def test_load_metadata_has_encoding(self, tmp_path):
        f = _make_file(tmp_path)
        loader = TextLoader(f, validate_path=False)
        docs = loader.load()
        assert "encoding" in docs[0].metadata

    def test_load_empty_file(self, tmp_path):
        f = _make_file(tmp_path, content="")
        loader = TextLoader(f, validate_path=False)
        docs = loader.load()
        assert len(docs) == 1
        assert docs[0].content == ""

    def test_load_raises_on_missing_file(self, tmp_path):
        f = tmp_path / "nonexistent.txt"
        loader = TextLoader(f, validate_path=False)
        with pytest.raises(Exception):
            loader.load()


# ---------------------------------------------------------------------------
# lazy_load()
# ---------------------------------------------------------------------------


class TestTextLoaderLazyLoad:
    def test_lazy_load_yields_same_as_load(self, tmp_path):
        f = _make_file(tmp_path, content="Hello lazy")
        loader = TextLoader(f, validate_path=False)
        docs = list(loader.lazy_load())
        assert len(docs) == 1
        assert "Hello lazy" in docs[0].content


# ---------------------------------------------------------------------------
# _should_use_mmap
# ---------------------------------------------------------------------------


class TestShouldUseMmap:
    def test_explicit_true(self, tmp_path):
        f = _make_file(tmp_path)
        loader = TextLoader(f, use_mmap=True, validate_path=False)
        assert loader._should_use_mmap() is True

    def test_explicit_false(self, tmp_path):
        f = _make_file(tmp_path)
        loader = TextLoader(f, use_mmap=False, validate_path=False)
        assert loader._should_use_mmap() is False

    def test_auto_false_for_small_file(self, tmp_path):
        f = _make_file(tmp_path, content="tiny")
        loader = TextLoader(f, validate_path=False)
        assert loader._should_use_mmap() is False

    def test_auto_true_for_large_file(self, tmp_path):
        f = _make_file(tmp_path)
        loader = TextLoader(f, validate_path=False)
        # Simulate large file by patching os.path.getsize
        with patch("os.path.getsize", return_value=20 * 1024 * 1024):
            result = loader._should_use_mmap()
        assert result is True

    def test_auto_returns_false_on_error(self, tmp_path):
        f = _make_file(tmp_path)
        loader = TextLoader(f, validate_path=False)
        with patch("os.path.getsize", side_effect=OSError("no such file")):
            result = loader._should_use_mmap()
        assert result is False


# ---------------------------------------------------------------------------
# _read_normal
# ---------------------------------------------------------------------------


class TestReadNormal:
    def test_reads_utf8_content(self, tmp_path):
        f = _make_file(tmp_path, content="UTF-8 content: 안녕")
        loader = TextLoader(f, validate_path=False)
        content = loader._read_normal()
        assert "안녕" in content

    def test_reads_without_autodetect(self, tmp_path):
        f = _make_file(tmp_path, content="simple text")
        loader = TextLoader(f, autodetect_encoding=False, validate_path=False)
        content = loader._read_normal()
        assert content == "simple text"

    def test_falls_back_to_latin1_on_unicode_error(self, tmp_path):
        # Write latin-1 encoded content
        f = tmp_path / "latin.txt"
        f.write_bytes(b"caf\xe9 au lait")  # 'é' in latin-1
        loader = TextLoader(f, encoding="utf-8", autodetect_encoding=True, validate_path=False)
        content = loader._read_normal()
        assert "caf" in content


# ---------------------------------------------------------------------------
# _decode_with_encoding_detection
# ---------------------------------------------------------------------------


class TestDecodeWithEncodingDetection:
    def test_decodes_utf8(self, tmp_path):
        f = _make_file(tmp_path)
        loader = TextLoader(f, validate_path=False)
        data = "Hello world".encode("utf-8")
        result = loader._decode_with_encoding_detection(data)
        assert result == "Hello world"

    def test_falls_back_to_latin1(self, tmp_path):
        f = _make_file(tmp_path)
        loader = TextLoader(f, encoding="utf-8", validate_path=False)
        data = b"caf\xe9"  # latin-1 'é'
        result = loader._decode_with_encoding_detection(data)
        assert "caf" in result


# ---------------------------------------------------------------------------
# load_streaming() — normal mode
# ---------------------------------------------------------------------------


class TestLoadStreamingNormal:
    def test_streaming_yields_documents(self, tmp_path):
        content = "A" * 100
        f = _make_file(tmp_path, content=content)
        loader = TextLoader(f, use_mmap=False, validate_path=False, chunk_size=50)
        docs = list(loader.load_streaming())
        assert len(docs) >= 1
        assert all(isinstance(d, Document) for d in docs)

    def test_streaming_chunks_correctly(self, tmp_path):
        content = "X" * 200
        f = _make_file(tmp_path, content=content)
        loader = TextLoader(f, use_mmap=False, validate_path=False, chunk_size=100)
        docs = list(loader.load_streaming())
        assert len(docs) == 2
        assert docs[0].metadata["chunk_index"] == 0
        assert docs[1].metadata["chunk_index"] == 1

    def test_streaming_metadata_has_source(self, tmp_path):
        f = _make_file(tmp_path, content="content")
        loader = TextLoader(f, use_mmap=False, validate_path=False)
        docs = list(loader.load_streaming())
        assert "source" in docs[0].metadata

    def test_streaming_small_file_produces_one_chunk(self, tmp_path):
        f = _make_file(tmp_path, content="small content")
        loader = TextLoader(f, use_mmap=False, validate_path=False, chunk_size=1024 * 1024)
        docs = list(loader.load_streaming())
        assert len(docs) == 1


# ---------------------------------------------------------------------------
# load_streaming() — mmap mode
# ---------------------------------------------------------------------------


class TestLoadStreamingMmap:
    def test_mmap_streaming_yields_documents(self, tmp_path):
        content = "B" * 100
        f = _make_file(tmp_path, content=content)
        loader = TextLoader(f, use_mmap=True, validate_path=False, chunk_size=50)
        docs = list(loader.load_streaming())
        # mmap may fail and fall back to normal on small files, but should yield docs
        assert len(docs) >= 1

    def test_mmap_streaming_has_file_size_metadata(self, tmp_path):
        content = "C" * 50
        f = _make_file(tmp_path, content=content)
        loader = TextLoader(f, use_mmap=True, validate_path=False, chunk_size=100)
        docs = list(loader.load_streaming())
        assert len(docs) >= 1
        # Either mmap or fallback should produce valid docs
        assert "source" in docs[0].metadata


# ---------------------------------------------------------------------------
# _read_with_mmap
# ---------------------------------------------------------------------------


class TestReadWithMmap:
    def test_mmap_read_returns_content(self, tmp_path):
        f = _make_file(tmp_path, content="mmap content")
        loader = TextLoader(f, use_mmap=True, validate_path=False)
        content = loader._read_with_mmap()
        assert "mmap content" in content

    def test_mmap_falls_back_on_error(self, tmp_path):
        f = _make_file(tmp_path, content="fallback")
        loader = TextLoader(f, use_mmap=True, validate_path=False)
        with patch("mmap.mmap", side_effect=OSError("mmap failed")):
            content = loader._read_with_mmap()
        assert "fallback" in content
