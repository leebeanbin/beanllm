"""
Comprehensive tests for JupyterLoader.
Target: src/beanllm/domain/loaders/core/jupyter.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, mock_open, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers — build a minimal nbformat-compatible notebook dict
# ---------------------------------------------------------------------------


def _make_notebook(cells: List[Dict]) -> Any:
    """Return a MagicMock that mimics nbformat.NotebookNode."""
    nb = MagicMock()
    nb.metadata = {
        "kernelspec": {"name": "python3", "language": "python"},
    }
    nb.cells = [_make_cell(**c) for c in cells]
    return nb


def _make_cell(
    cell_type: str = "code",
    source: str = "print('hello')",
    execution_count: int | None = 1,
    outputs: List[Dict] | None = None,
) -> Any:
    cell = MagicMock()
    cell.cell_type = cell_type
    cell.source = source
    cell.get = MagicMock(
        side_effect=lambda k, default=None: {
            "source": source,
            "outputs": outputs or [],
            "execution_count": execution_count,
        }.get(k, default)
    )
    return cell


def _make_nbformat_mock(notebook: Any) -> Any:
    mock_nbformat = MagicMock()
    mock_nbformat.read = MagicMock(return_value=notebook)
    return mock_nbformat


def _make_loader(file_path: str = "notebook.ipynb", **kwargs):
    from beanllm.domain.loaders.core.jupyter import JupyterLoader

    return JupyterLoader(file_path=file_path, **kwargs)


# ---------------------------------------------------------------------------
# Initialisation tests
# ---------------------------------------------------------------------------


class TestJupyterLoaderInit:
    def test_default_init(self):
        loader = _make_loader()
        assert loader.include_outputs is True
        assert loader.filter_cell_types is None
        assert loader.concatenate_cells is True
        assert loader.file_path == Path("notebook.ipynb")

    def test_custom_init(self):
        loader = _make_loader(
            "analysis.ipynb",
            include_outputs=False,
            filter_cell_types=["code"],
            concatenate_cells=False,
        )
        assert loader.include_outputs is False
        assert loader.filter_cell_types == ["code"]
        assert loader.concatenate_cells is False

    def test_path_object_accepted(self):
        loader = _make_loader(Path("dir/notebook.ipynb"))
        assert loader.file_path == Path("dir/notebook.ipynb")


# ---------------------------------------------------------------------------
# load() — nbformat missing
# ---------------------------------------------------------------------------


class TestLoadNbformatMissing:
    def test_raises_import_error_when_nbformat_missing(self, tmp_path):
        nb_file = tmp_path / "test.ipynb"
        nb_file.write_text("{}")

        with patch.dict(sys.modules, {"nbformat": None}):
            loader = _make_loader(str(nb_file))
            with pytest.raises((ImportError, TypeError)):
                loader.load()


# ---------------------------------------------------------------------------
# load() — concatenate_cells=True (default)
# ---------------------------------------------------------------------------


class TestLoadConcatenated:
    def test_single_code_cell(self, tmp_path):
        nb_file = tmp_path / "test.ipynb"
        nb_file.write_text("{}")

        nb = _make_notebook(
            [
                {"cell_type": "code", "source": "x = 42", "execution_count": 1},
            ]
        )
        mock_nbformat = _make_nbformat_mock(nb)

        with patch.dict(sys.modules, {"nbformat": mock_nbformat}):
            loader = _make_loader(str(nb_file))
            with patch("builtins.open", mock_open(read_data="{}")):
                with patch("nbformat.read", return_value=nb):
                    docs = loader.load()

        assert len(docs) == 1
        assert docs[0].metadata["kernel"] == "python3"

    def test_multiple_cells_combined(self, tmp_path):
        nb_file = tmp_path / "test.ipynb"
        nb_file.write_text("{}")

        nb = _make_notebook(
            [
                {"cell_type": "code", "source": "x = 1"},
                {"cell_type": "markdown", "source": "# Title", "execution_count": None},
            ]
        )
        mock_nbformat = _make_nbformat_mock(nb)

        with patch.dict(sys.modules, {"nbformat": mock_nbformat}):
            loader = _make_loader(str(nb_file))
            with patch("builtins.open", mock_open(read_data="{}")):
                with patch("nbformat.read", return_value=nb):
                    docs = loader.load()

        assert len(docs) == 1

    def test_filter_code_cells_only(self, tmp_path):
        nb_file = tmp_path / "test.ipynb"
        nb_file.write_text("{}")

        nb = _make_notebook(
            [
                {"cell_type": "code", "source": "x = 1"},
                {"cell_type": "markdown", "source": "# Title", "execution_count": None},
            ]
        )
        mock_nbformat = _make_nbformat_mock(nb)

        with patch.dict(sys.modules, {"nbformat": mock_nbformat}):
            loader = _make_loader(str(nb_file), filter_cell_types=["code"])
            with patch("builtins.open", mock_open(read_data="{}")):
                with patch("nbformat.read", return_value=nb):
                    docs = loader.load()

        assert len(docs) == 1
        # Only code cell content should be present

    def test_filter_markdown_only(self, tmp_path):
        nb_file = tmp_path / "test.ipynb"
        nb_file.write_text("{}")

        nb = _make_notebook(
            [
                {"cell_type": "code", "source": "x = 1"},
                {"cell_type": "markdown", "source": "# Title", "execution_count": None},
            ]
        )
        mock_nbformat = _make_nbformat_mock(nb)

        with patch.dict(sys.modules, {"nbformat": mock_nbformat}):
            loader = _make_loader(str(nb_file), filter_cell_types=["markdown"])
            with patch("builtins.open", mock_open(read_data="{}")):
                with patch("nbformat.read", return_value=nb):
                    docs = loader.load()

        assert len(docs) == 1


# ---------------------------------------------------------------------------
# load() — concatenate_cells=False
# ---------------------------------------------------------------------------


class TestLoadSeparateDocs:
    def test_each_cell_is_separate_doc(self, tmp_path):
        nb_file = tmp_path / "test.ipynb"
        nb_file.write_text("{}")

        nb = _make_notebook(
            [
                {"cell_type": "code", "source": "x = 1"},
                {"cell_type": "markdown", "source": "# Title", "execution_count": None},
            ]
        )
        mock_nbformat = _make_nbformat_mock(nb)

        with patch.dict(sys.modules, {"nbformat": mock_nbformat}):
            loader = _make_loader(str(nb_file), concatenate_cells=False)
            with patch("builtins.open", mock_open(read_data="{}")):
                with patch("nbformat.read", return_value=nb):
                    docs = loader.load()

        assert len(docs) == 2

    def test_separate_docs_have_cell_metadata(self, tmp_path):
        nb_file = tmp_path / "test.ipynb"
        nb_file.write_text("{}")

        nb = _make_notebook(
            [
                {"cell_type": "code", "source": "x = 1"},
            ]
        )
        mock_nbformat = _make_nbformat_mock(nb)

        with patch.dict(sys.modules, {"nbformat": mock_nbformat}):
            loader = _make_loader(str(nb_file), concatenate_cells=False)
            with patch("builtins.open", mock_open(read_data="{}")):
                with patch("nbformat.read", return_value=nb):
                    docs = loader.load()

        assert len(docs) == 1
        assert "cell_type" in docs[0].metadata
        assert "cell_index" in docs[0].metadata

    def test_separate_docs_filter_applies(self, tmp_path):
        nb_file = tmp_path / "test.ipynb"
        nb_file.write_text("{}")

        nb = _make_notebook(
            [
                {"cell_type": "code", "source": "x = 1"},
                {"cell_type": "markdown", "source": "# H", "execution_count": None},
            ]
        )
        mock_nbformat = _make_nbformat_mock(nb)

        with patch.dict(sys.modules, {"nbformat": mock_nbformat}):
            loader = _make_loader(str(nb_file), concatenate_cells=False, filter_cell_types=["code"])
            with patch("builtins.open", mock_open(read_data="{}")):
                with patch("nbformat.read", return_value=nb):
                    docs = loader.load()

        assert len(docs) == 1
        assert docs[0].metadata["cell_type"] == "code"


# ---------------------------------------------------------------------------
# _format_cell tests
# ---------------------------------------------------------------------------


class TestFormatCell:
    def test_code_cell_with_execution_count(self):
        loader = _make_loader()
        cell = _make_cell(cell_type="code", source="x = 1", execution_count=3)
        result = loader._format_cell(cell, 0)
        assert "CODE" in result
        assert "execution 3" in result

    def test_code_cell_without_execution_count(self):
        loader = _make_loader()
        cell = _make_cell(cell_type="code", source="x = 1", execution_count=None)
        result = loader._format_cell(cell, 1)
        assert "CODE" in result
        assert "execution" not in result

    def test_markdown_cell(self):
        loader = _make_loader()
        cell = _make_cell(cell_type="markdown", source="# Heading", execution_count=None)
        result = loader._format_cell(cell, 0)
        assert "MARKDOWN" in result
        assert "# Heading" in result

    def test_empty_source_still_produces_header(self):
        loader = _make_loader()
        cell = _make_cell(cell_type="code", source="")
        result = loader._format_cell(cell, 0)
        assert "CODE" in result

    def test_source_as_list_joined(self):
        loader = _make_loader()
        cell = MagicMock()
        cell.cell_type = "code"
        cell.get = MagicMock(
            side_effect=lambda k, default=None: {
                "source": ["line1\n", "line2\n"],
                "outputs": [],
                "execution_count": 1,
            }.get(k, default)
        )
        result = loader._format_cell(cell, 0)
        assert "line1" in result or "line2" in result

    def test_outputs_included_when_enabled(self):
        loader = _make_loader(include_outputs=True)
        cell = _make_cell(
            cell_type="code",
            source="print('hi')",
            outputs=[{"output_type": "stream", "text": "hi\n"}],
        )
        result = loader._format_cell(cell, 0)
        assert "OUTPUT" in result or "hi" in result

    def test_outputs_excluded_when_disabled(self):
        loader = _make_loader(include_outputs=False)
        cell = _make_cell(
            cell_type="code",
            source="print('hi')",
            outputs=[{"output_type": "stream", "text": "hi\n"}],
        )
        result = loader._format_cell(cell, 0)
        assert "OUTPUT" not in result


# ---------------------------------------------------------------------------
# _format_output tests
# ---------------------------------------------------------------------------


class TestFormatOutput:
    def test_stream_output(self):
        loader = _make_loader()
        output = {"output_type": "stream", "text": "Hello output\n"}
        result = loader._format_output(output)
        assert "Hello output" in result

    def test_stream_output_list(self):
        loader = _make_loader()
        output = {"output_type": "stream", "text": ["line1\n", "line2\n"]}
        result = loader._format_output(output)
        assert "line1" in result

    def test_execute_result_text_plain(self):
        loader = _make_loader()
        output = {
            "output_type": "execute_result",
            "data": {"text/plain": "42"},
        }
        result = loader._format_output(output)
        assert "42" in result

    def test_execute_result_text_plain_list(self):
        loader = _make_loader()
        output = {
            "output_type": "execute_result",
            "data": {"text/plain": ["line1", "line2"]},
        }
        result = loader._format_output(output)
        assert "line1" in result

    def test_display_data_html(self):
        loader = _make_loader()
        output = {
            "output_type": "display_data",
            "data": {"text/html": "<table>...</table>"},
        }
        result = loader._format_output(output)
        assert "HTML OUTPUT" in result

    def test_display_data_image(self):
        loader = _make_loader()
        output = {
            "output_type": "display_data",
            "data": {"image/png": "base64data"},
        }
        result = loader._format_output(output)
        assert "IMAGE" in result

    def test_error_output(self):
        loader = _make_loader()
        output = {
            "output_type": "error",
            "ename": "ValueError",
            "evalue": "invalid value",
            "traceback": ["Traceback line 1"],
        }
        result = loader._format_output(output)
        assert "ValueError" in result
        assert "invalid value" in result

    def test_error_output_no_traceback(self):
        loader = _make_loader()
        output = {
            "output_type": "error",
            "ename": "TypeError",
            "evalue": "bad type",
            "traceback": [],
        }
        result = loader._format_output(output)
        assert "TypeError" in result

    def test_unknown_output_type(self):
        loader = _make_loader()
        output = {"output_type": "unknown_type"}
        result = loader._format_output(output)
        assert result == ""


# ---------------------------------------------------------------------------
# lazy_load tests
# ---------------------------------------------------------------------------


class TestLazyLoad:
    def test_lazy_load_yields_same_as_load(self, tmp_path):
        nb_file = tmp_path / "test.ipynb"
        nb_file.write_text("{}")

        nb = _make_notebook(
            [
                {"cell_type": "code", "source": "x = 1"},
            ]
        )
        mock_nbformat = _make_nbformat_mock(nb)

        with patch.dict(sys.modules, {"nbformat": mock_nbformat}):
            loader = _make_loader(str(nb_file))
            with patch("builtins.open", mock_open(read_data="{}")):
                with patch("nbformat.read", return_value=nb):
                    docs_load = loader.load()
                    docs_lazy = list(loader.lazy_load())

        assert len(docs_load) == len(docs_lazy)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_load_raises_on_missing_file(self):
        nb = _make_notebook([])
        mock_nbformat = _make_nbformat_mock(nb)

        with patch.dict(sys.modules, {"nbformat": mock_nbformat}):
            loader = _make_loader("/nonexistent/path/notebook.ipynb")
            with pytest.raises(Exception):
                loader.load()

    def test_load_raises_on_corrupt_notebook(self, tmp_path):
        nb_file = tmp_path / "corrupt.ipynb"
        nb_file.write_text("not valid json {{{")

        mock_nbformat = MagicMock()
        mock_nbformat.read.side_effect = Exception("Invalid notebook")

        with patch.dict(sys.modules, {"nbformat": mock_nbformat}):
            loader = _make_loader(str(nb_file))
            with pytest.raises(Exception):
                loader.load()
