"""
데이터 모델 테스트 (PageData, TableData, ImageData, PDFLoadConfig, PDFLoadResult)
"""

import pytest

from src.beanllm.domain.loaders.pdf.models import (
    ImageData,
    PageData,
    PDFLoadConfig,
    PDFLoadResult,
    TableData,
)


class TestPageData:
    """PageData 모델 테스트"""

    def test_page_data_creation(self):
        """PageData 생성 테스트"""
        page = PageData(
            page=0,
            text="Test content",
            width=595.0,
            height=842.0,
            metadata={"test": "value"},
        )

        assert page.page == 0
        assert page.text == "Test content"
        assert page.width == 595.0
        assert page.height == 842.0
        assert page.metadata["test"] == "value"

    def test_page_data_to_dict(self):
        """PageData to_dict 변환 테스트"""
        page = PageData(page=0, text="Test", width=595.0, height=842.0, metadata={"key": "val"})

        data = page.to_dict()

        assert isinstance(data, dict)
        assert data["page"] == 0
        assert data["text"] == "Test"
        assert data["width"] == 595.0
        assert data["height"] == 842.0
        assert data["metadata"]["key"] == "val"


class TestTableData:
    """TableData 모델 테스트"""

    def test_table_data_creation(self):
        """TableData 생성 테스트"""
        table = TableData(
            page=0,
            table_index=0,
            data=[["A", "B"], ["1", "2"]],
            bbox=(10.0, 20.0, 100.0, 200.0),
            confidence=0.95,
        )

        assert table.page == 0
        assert table.table_index == 0
        assert len(table.data) == 2
        assert table.bbox == (10.0, 20.0, 100.0, 200.0)
        assert table.confidence == 0.95

    def test_table_data_to_dict(self):
        """TableData to_dict 변환 테스트"""
        table = TableData(
            page=0,
            table_index=0,
            data=[["A", "B"], ["1", "2"]],
            bbox=(10.0, 20.0, 100.0, 200.0),
        )

        data = table.to_dict()

        assert isinstance(data, dict)
        assert data["page"] == 0
        assert data["table_index"] == 0
        assert data["data"] == [["A", "B"], ["1", "2"]]
        assert data["bbox"] == (10.0, 20.0, 100.0, 200.0)

    def test_table_data_with_dataframe(self):
        """TableData DataFrame 변환 테스트"""
        try:
            import pandas as pd

            df = pd.DataFrame([["A", "B"], ["1", "2"]], columns=["col1", "col2"])

            table = TableData(
                page=0,
                table_index=0,
                data=df,
                bbox=(10.0, 20.0, 100.0, 200.0),
            )

            data = table.to_dict()
            assert "data" in data
            assert isinstance(data["data"], list)
        except ImportError:
            pytest.skip("pandas not installed")


class TestImageData:
    """ImageData 모델 테스트"""

    def test_image_data_creation(self):
        """ImageData 생성 테스트"""
        image = ImageData(
            page=0,
            image_index=0,
            image=b"fake_image_bytes",
            format="png",
            width=800,
            height=600,
            bbox=(10.0, 20.0, 100.0, 200.0),
            size=1024,
        )

        assert image.page == 0
        assert image.image_index == 0
        assert image.format == "png"
        assert image.width == 800
        assert image.height == 600
        assert image.size == 1024

    def test_image_data_to_dict(self):
        """ImageData to_dict 변환 테스트 (이미지 데이터 제외)"""
        image = ImageData(
            page=0,
            image_index=0,
            image=b"fake_bytes",
            format="jpeg",
            width=800,
            height=600,
            bbox=(10.0, 20.0, 100.0, 200.0),
            size=2048,
        )

        data = image.to_dict()

        assert isinstance(data, dict)
        assert "image" not in data  # 이미지 데이터는 제외
        assert data["format"] == "jpeg"
        assert data["width"] == 800
        assert data["height"] == 600


class TestPDFLoadConfig:
    """PDFLoadConfig 모델 테스트"""

    def test_config_default_values(self):
        """기본값 테스트"""
        config = PDFLoadConfig()

        assert config.strategy == "auto"
        assert config.extract_tables is True
        assert config.extract_images is False
        assert config.to_markdown is False
        assert config.enable_ocr is False
        assert config.layout_analysis is False
        assert config.max_pages is None
        assert config.page_range is None

    def test_config_custom_values(self):
        """커스텀 값 테스트"""
        config = PDFLoadConfig(
            strategy="fast",
            extract_tables=False,
            extract_images=True,
            max_pages=10,
            page_range=(0, 5),
        )

        assert config.strategy == "fast"
        assert config.extract_tables is False
        assert config.extract_images is True
        assert config.max_pages == 10
        assert config.page_range == (0, 5)

    def test_config_to_dict(self):
        """Config to_dict 변환 테스트"""
        config = PDFLoadConfig(strategy="accurate", extract_tables=True)

        data = config.to_dict()

        assert isinstance(data, dict)
        assert data["strategy"] == "accurate"
        assert data["extract_tables"] is True

    def test_config_from_dict(self):
        """Config from_dict 생성 테스트"""
        data = {
            "strategy": "fast",
            "extract_tables": False,
            "extract_images": True,
            "to_markdown": False,
            "enable_ocr": False,
            "layout_analysis": False,
            "max_pages": None,
            "page_range": None,
            "pymupdf_text_mode": "text",
            "pymupdf_extract_fonts": False,
            "pymupdf_extract_links": False,
            "pdfplumber_layout": False,
            "pdfplumber_extract_chars": False,
            "pdfplumber_extract_words": False,
            "pdfplumber_extract_hyperlinks": False,
            "pdfplumber_x_tolerance": 3.0,
            "pdfplumber_y_tolerance": 3.0,
        }

        config = PDFLoadConfig.from_dict(data)

        assert config.strategy == "fast"
        assert config.extract_tables is False
        assert config.extract_images is True


class TestPDFLoadResult:
    """PDFLoadResult 모델 테스트"""

    def test_result_creation(self):
        """PDFLoadResult 생성 테스트"""
        page1 = PageData(page=0, text="Page 1", width=595.0, height=842.0)
        page2 = PageData(page=1, text="Page 2", width=595.0, height=842.0)

        result = PDFLoadResult(
            pages=[page1, page2],
            metadata={"total_pages": 2, "engine": "PyMuPDF"},
        )

        assert len(result.pages) == 2
        assert result.pages[0].text == "Page 1"
        assert result.pages[1].text == "Page 2"
        assert result.metadata["total_pages"] == 2
        assert len(result.tables) == 0
        assert len(result.images) == 0

    def test_result_with_tables_and_images(self):
        """테이블 및 이미지 포함 테스트"""
        page = PageData(page=0, text="Test", width=595.0, height=842.0)
        table = TableData(
            page=0,
            table_index=0,
            data=[["A"]],
            bbox=(0.0, 0.0, 100.0, 100.0),
        )
        image = ImageData(
            page=0,
            image_index=0,
            image=b"test",
            format="png",
            width=100,
            height=100,
            bbox=(0.0, 0.0, 100.0, 100.0),
            size=100,
        )

        result = PDFLoadResult(
            pages=[page],
            tables=[table],
            images=[image],
            metadata={"total_pages": 1},
        )

        assert len(result.pages) == 1
        assert len(result.tables) == 1
        assert len(result.images) == 1

    def test_result_to_dict(self):
        """PDFLoadResult to_dict 변환 테스트"""
        page = PageData(page=0, text="Test", width=595.0, height=842.0)
        result = PDFLoadResult(pages=[page], metadata={"total_pages": 1})

        data = result.to_dict()

        assert isinstance(data, dict)
        assert "pages" in data
        assert "tables" in data
        assert "images" in data
        assert "metadata" in data
        assert len(data["pages"]) == 1
