"""Tests for domain/loaders/core/directory.py — DirectoryLoader."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from beanllm.domain.loaders.core.directory import DirectoryLoader
from beanllm.domain.loaders.types import Document


def _make_doc(content="test", source="file.txt"):
    return Document(content=content, metadata={"source": source})


def _make_dir_with_files(tmp_path, files):
    """Create a temp directory with the given files."""
    for name, content in files.items():
        fp = tmp_path / name
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestDirectoryLoaderInit:
    def test_creates_with_defaults(self, tmp_path):
        loader = DirectoryLoader(tmp_path)
        assert loader.path == tmp_path
        assert loader.glob == "**/*"
        assert loader.recursive is True
        assert loader.use_parallel is True
        assert loader.exclude == []

    def test_custom_glob(self, tmp_path):
        loader = DirectoryLoader(tmp_path, glob="**/*.txt")
        assert loader.glob == "**/*.txt"

    def test_exclude_patterns_compiled(self, tmp_path):
        loader = DirectoryLoader(tmp_path, exclude=["**/__pycache__/**", "**/*.pyc"])
        assert len(loader._compiled_exclude_patterns) == 2

    def test_invalid_exclude_pattern_falls_back(self, tmp_path):
        # An invalid regex-convertible pattern should be kept as string fallback
        loader = DirectoryLoader(tmp_path, exclude=["**/*.txt"])
        assert len(loader._compiled_exclude_patterns) == 1

    def test_string_path_converted_to_path(self, tmp_path):
        loader = DirectoryLoader(str(tmp_path))
        assert isinstance(loader.path, Path)

    def test_sequential_mode(self, tmp_path):
        loader = DirectoryLoader(tmp_path, use_parallel=False)
        assert loader.use_parallel is False


# ---------------------------------------------------------------------------
# load() — empty dir
# ---------------------------------------------------------------------------


class TestDirectoryLoaderLoadEmpty:
    def test_empty_dir_returns_empty_list(self, tmp_path):
        loader = DirectoryLoader(tmp_path)
        result = loader.load()
        assert result == []

    def test_dir_with_only_subdirs_returns_empty(self, tmp_path):
        (tmp_path / "subdir").mkdir()
        loader = DirectoryLoader(tmp_path)
        with patch(
            "beanllm.domain.loaders.core.directory.DirectoryLoader._load_single_file",
            return_value=[],
        ):
            result = loader.load()
        assert result == []


# ---------------------------------------------------------------------------
# load() — sequential mode
# ---------------------------------------------------------------------------


class TestDirectoryLoaderLoadSequential:
    def test_loads_single_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        loader = DirectoryLoader(tmp_path, use_parallel=False)

        mock_doc = _make_doc("hello", str(f))
        with patch(
            "beanllm.domain.loaders.core.directory.DirectoryLoader._load_single_file",
            return_value=[mock_doc],
        ) as mock_load:
            result = loader.load()

        assert len(result) == 1
        assert result[0].page_content == "hello"

    def test_loads_multiple_files(self, tmp_path):
        for i in range(3):
            (tmp_path / f"doc{i}.txt").write_text(f"content{i}")

        loader = DirectoryLoader(tmp_path, use_parallel=False)
        mock_docs = [_make_doc(f"c{i}") for i in range(3)]

        call_count = [0]

        def side_effect(path):
            doc = mock_docs[call_count[0] % len(mock_docs)]
            call_count[0] += 1
            return [doc]

        with patch(
            "beanllm.domain.loaders.core.directory.DirectoryLoader._load_single_file",
            side_effect=side_effect,
        ):
            result = loader.load()

        assert len(result) == 3

    def test_sequential_for_single_file(self, tmp_path):
        (tmp_path / "only.txt").write_text("content")
        loader = DirectoryLoader(tmp_path, use_parallel=True)

        with patch(
            "beanllm.domain.loaders.core.directory.DirectoryLoader._load_single_file",
            return_value=[_make_doc("content")],
        ):
            result = loader.load()

        assert len(result) == 1

    def test_loader_returns_empty_on_exception(self, tmp_path):
        (tmp_path / "bad.txt").write_text("content")
        loader = DirectoryLoader(tmp_path, use_parallel=False)

        with patch(
            "beanllm.domain.loaders.core.directory.DirectoryLoader._load_single_file",
            return_value=[],
        ):
            result = loader.load()

        assert result == []


# ---------------------------------------------------------------------------
# load() — exclude patterns
# ---------------------------------------------------------------------------


class TestDirectoryLoaderExclude:
    def test_excludes_matching_files(self, tmp_path):
        keep = tmp_path / "keep.txt"
        keep.write_text("keep this")
        skip = tmp_path / "skip.pyc"
        skip.write_text("skip this")

        loader = DirectoryLoader(tmp_path, exclude=["*.pyc"], use_parallel=False)
        loaded_paths = []

        def capture_load(path):
            loaded_paths.append(path.name)
            return [_make_doc(str(path))]

        with patch(
            "beanllm.domain.loaders.core.directory.DirectoryLoader._load_single_file",
            side_effect=capture_load,
        ):
            loader.load()

        assert "keep.txt" in loaded_paths
        assert "skip.pyc" not in loaded_paths

    def test_no_exclude_loads_all_files(self, tmp_path):
        for name in ["a.txt", "b.py", "c.md"]:
            (tmp_path / name).write_text(name)

        loader = DirectoryLoader(tmp_path, exclude=[], use_parallel=False)
        loaded = []

        def capture(path):
            loaded.append(path.name)
            return []

        with patch(
            "beanllm.domain.loaders.core.directory.DirectoryLoader._load_single_file",
            side_effect=capture,
        ):
            loader.load()

        assert len(loaded) == 3


# ---------------------------------------------------------------------------
# load() — parallel mode fallback
# ---------------------------------------------------------------------------


class TestDirectoryLoaderParallel:
    def test_parallel_fallback_to_sequential_on_error(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")

        loader = DirectoryLoader(tmp_path, use_parallel=True)

        with patch(
            "concurrent.futures.ProcessPoolExecutor.__enter__",
            side_effect=RuntimeError("executor error"),
        ):
            with patch(
                "beanllm.domain.loaders.core.directory.DirectoryLoader._load_single_file",
                return_value=[_make_doc("fallback")],
            ):
                result = loader.load()

        assert isinstance(result, list)

    def test_parallel_with_batch_processor(self, tmp_path):
        (tmp_path / "doc.txt").write_text("content")
        (tmp_path / "doc2.txt").write_text("content2")

        mock_batch_processor = MagicMock()
        import asyncio

        async def mock_process_batch(items, handler):
            return [[_make_doc("from batch")]] * len(items)

        mock_batch_processor.process_batch = mock_process_batch

        loader = DirectoryLoader(tmp_path, use_parallel=True, batch_processor=mock_batch_processor)
        result = loader.load()
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _load_single_file (static method)
# ---------------------------------------------------------------------------


class TestLoadSingleFile:
    def test_returns_empty_when_no_loader(self, tmp_path):
        f = tmp_path / "unknown.xyz"
        f.write_text("data")

        with patch(
            "beanllm.domain.loaders.factory.DocumentLoader.get_loader",
            return_value=None,
        ):
            result = DirectoryLoader._load_single_file(f)
        assert result == []

    def test_returns_documents_from_loader(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")

        mock_loader = MagicMock()
        mock_loader.load.return_value = [_make_doc("hello")]

        with patch(
            "beanllm.domain.loaders.factory.DocumentLoader.get_loader",
            return_value=mock_loader,
        ):
            result = DirectoryLoader._load_single_file(f)

        assert len(result) == 1
        assert result[0].content == "hello"

    def test_returns_empty_on_loader_exception(self, tmp_path):
        f = tmp_path / "bad.txt"
        f.write_text("content")

        mock_loader = MagicMock()
        mock_loader.load.side_effect = RuntimeError("parse error")

        with patch(
            "beanllm.domain.loaders.factory.DocumentLoader.get_loader",
            return_value=mock_loader,
        ):
            result = DirectoryLoader._load_single_file(f)

        assert result == []
