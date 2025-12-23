"""
Document Loaders 테스트 - 문서 로더 테스트
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from llmkit.domain.loaders import Document, DocumentLoader
from llmkit.domain.loaders.loaders import TextLoader, CSVLoader, DirectoryLoader


class TestTextLoader:
    """TextLoader 테스트"""

    @pytest.fixture
    def text_file(self, tmp_path):
        """임시 텍스트 파일"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Hello world\nThis is a test", encoding="utf-8")
        return file_path

    def test_load_text_file(self, text_file):
        """텍스트 파일 로딩 테스트"""
        loader = TextLoader(text_file)
        docs = loader.load()

        assert isinstance(docs, list)
        assert len(docs) > 0
        assert isinstance(docs[0], Document)
        assert "Hello world" in docs[0].content

    def test_lazy_load(self, text_file):
        """지연 로딩 테스트"""
        loader = TextLoader(text_file)
        docs = list(loader.lazy_load())

        assert isinstance(docs, list)
        assert len(docs) > 0


class TestCSVLoader:
    """CSVLoader 테스트"""

    @pytest.fixture
    def csv_file(self, tmp_path):
        """임시 CSV 파일"""
        file_path = tmp_path / "test.csv"
        file_path.write_text("name,age\nJohn,30\nJane,25", encoding="utf-8")
        return file_path

    def test_load_csv_file(self, csv_file):
        """CSV 파일 로딩 테스트"""
        try:
            loader = CSVLoader(csv_file)
            docs = loader.load()

            assert isinstance(docs, list)
            assert len(docs) > 0
        except (ImportError, AttributeError):
            pytest.skip("CSV loader not available")


class TestDocumentLoader:
    """DocumentLoader 팩토리 테스트"""

    @pytest.fixture
    def text_file(self, tmp_path):
        """임시 텍스트 파일"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Hello world", encoding="utf-8")
        return file_path

    def test_load_auto_detect(self, text_file):
        """자동 감지 로딩 테스트"""
        docs = DocumentLoader.load(text_file)

        assert isinstance(docs, list)
        assert len(docs) > 0

    def test_load_explicit_type(self, text_file):
        """명시적 타입 지정 로딩 테스트"""
        docs = DocumentLoader.load(text_file, loader_type="text")

        assert isinstance(docs, list)
        assert len(docs) > 0

    def test_get_loader_text(self, text_file):
        """텍스트 로더 가져오기 테스트"""
        loader = DocumentLoader.get_loader(text_file)

        assert loader is not None
        assert isinstance(loader, TextLoader)

    def test_get_loader_directory(self, tmp_path):
        """디렉토리 로더 가져오기 테스트"""
        (tmp_path / "file1.txt").write_text("Content 1")
        (tmp_path / "file2.txt").write_text("Content 2")

        loader = DocumentLoader.get_loader(tmp_path)

        assert loader is not None
        assert isinstance(loader, DirectoryLoader)


