"""
메타데이터 추출기 테스트 (TableExtractor, ImageExtractor)
"""

from pathlib import Path

import pytest

from src.beanllm.domain.loaders.pdf import beanPDFLoader
from src.beanllm.domain.loaders.pdf.extractors import ImageExtractor, TableExtractor

# 테스트 픽스처 경로
FIXTURES_DIR = Path(__file__).parent.parent.parent.parent / "fixtures" / "pdf"
SIMPLE_PDF = FIXTURES_DIR / "simple.pdf"
TABLES_PDF = FIXTURES_DIR / "tables.pdf"
IMAGES_PDF = FIXTURES_DIR / "images.pdf"


class TestTableExtractor:
    """TableExtractor 테스트"""

    def test_extractor_initialization(self):
        """추출기 초기화 테스트"""
        if not TABLES_PDF.exists():
            pytest.skip(f"Test fixture not found: {TABLES_PDF}")

        loader = beanPDFLoader(TABLES_PDF, extract_tables=True)
        docs = loader.load()

        extractor = TableExtractor(docs)
        assert extractor.documents == docs

    def test_get_all_tables(self):
        """모든 테이블 추출 테스트"""
        if not TABLES_PDF.exists():
            pytest.skip(f"Test fixture not found: {TABLES_PDF}")

        loader = beanPDFLoader(TABLES_PDF, extract_tables=True)
        docs = loader.load()

        extractor = TableExtractor(docs)
        tables = extractor.get_all_tables()

        # tables.pdf에는 테이블이 있을 것으로 예상
        assert isinstance(tables, list)

        if len(tables) > 0:
            # 첫 번째 테이블 검증
            table = tables[0]
            assert "page" in table
            assert "table_index" in table
            assert "rows" in table
            assert "cols" in table
            assert "confidence" in table
            assert "source" in table

    def test_get_tables_by_page(self):
        """페이지별 테이블 추출 테스트"""
        if not TABLES_PDF.exists():
            pytest.skip(f"Test fixture not found: {TABLES_PDF}")

        loader = beanPDFLoader(TABLES_PDF, extract_tables=True)
        docs = loader.load()

        extractor = TableExtractor(docs)
        page_0_tables = extractor.get_tables_by_page(0)

        assert isinstance(page_0_tables, list)
        # 모든 테이블이 page 0에 속해야 함
        assert all(t["page"] == 0 for t in page_0_tables)

    def test_get_high_quality_tables(self):
        """고품질 테이블 필터링 테스트"""
        if not TABLES_PDF.exists():
            pytest.skip(f"Test fixture not found: {TABLES_PDF}")

        loader = beanPDFLoader(TABLES_PDF, extract_tables=True)
        docs = loader.load()

        extractor = TableExtractor(docs)
        high_quality = extractor.get_high_quality_tables(min_confidence=0.5)

        assert isinstance(high_quality, list)
        # 모든 테이블의 신뢰도가 0.5 이상이어야 함
        assert all(t["confidence"] >= 0.5 for t in high_quality)

    def test_get_tables_by_size(self):
        """크기별 테이블 필터링 테스트"""
        if not TABLES_PDF.exists():
            pytest.skip(f"Test fixture not found: {TABLES_PDF}")

        loader = beanPDFLoader(TABLES_PDF, extract_tables=True)
        docs = loader.load()

        extractor = TableExtractor(docs)
        large_tables = extractor.get_tables_by_size(min_rows=2, min_cols=2)

        assert isinstance(large_tables, list)
        # 모든 테이블이 조건을 만족해야 함
        assert all(t["rows"] >= 2 and t["cols"] >= 2 for t in large_tables)

    def test_get_summary(self):
        """요약 정보 테스트"""
        if not TABLES_PDF.exists():
            pytest.skip(f"Test fixture not found: {TABLES_PDF}")

        loader = beanPDFLoader(TABLES_PDF, extract_tables=True)
        docs = loader.load()

        extractor = TableExtractor(docs)
        summary = extractor.get_summary()

        assert "total_tables" in summary
        assert "pages_with_tables" in summary
        assert "avg_confidence" in summary
        assert "tables_by_page" in summary
        assert "high_quality_count" in summary

    def test_export_to_markdown(self):
        """Markdown 내보내기 테스트"""
        if not TABLES_PDF.exists():
            pytest.skip(f"Test fixture not found: {TABLES_PDF}")

        loader = beanPDFLoader(TABLES_PDF, extract_tables=True)
        docs = loader.load()

        extractor = TableExtractor(docs)
        markdown = extractor.export_to_markdown()

        assert isinstance(markdown, str)
        assert "# Extracted Tables" in markdown

    def test_no_tables(self):
        """테이블이 없는 문서 테스트"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")

        loader = beanPDFLoader(SIMPLE_PDF, extract_tables=True)
        docs = loader.load()

        extractor = TableExtractor(docs)
        tables = extractor.get_all_tables()

        assert isinstance(tables, list)
        assert len(tables) == 0

        summary = extractor.get_summary()
        assert summary["total_tables"] == 0


class TestImageExtractor:
    """ImageExtractor 테스트"""

    def test_extractor_initialization(self):
        """추출기 초기화 테스트"""
        if not IMAGES_PDF.exists():
            pytest.skip(f"Test fixture not found: {IMAGES_PDF}")

        loader = beanPDFLoader(IMAGES_PDF, extract_images=True, strategy="fast")
        docs = loader.load()

        extractor = ImageExtractor(docs)
        assert extractor.documents == docs

    def test_get_all_images(self):
        """모든 이미지 추출 테스트"""
        if not IMAGES_PDF.exists():
            pytest.skip(f"Test fixture not found: {IMAGES_PDF}")

        loader = beanPDFLoader(IMAGES_PDF, extract_images=True, strategy="fast")
        docs = loader.load()

        extractor = ImageExtractor(docs)
        images = extractor.get_all_images()

        assert isinstance(images, list)

        # images.pdf에는 그래픽 요소가 있을 수 있음
        if len(images) > 0:
            # 첫 번째 이미지 검증
            img = images[0]
            assert "page" in img
            assert "image_index" in img
            assert "format" in img
            assert "width" in img
            assert "height" in img
            assert "size" in img
            assert "source" in img

    def test_get_images_by_page(self):
        """페이지별 이미지 추출 테스트"""
        if not IMAGES_PDF.exists():
            pytest.skip(f"Test fixture not found: {IMAGES_PDF}")

        loader = beanPDFLoader(IMAGES_PDF, extract_images=True, strategy="fast")
        docs = loader.load()

        extractor = ImageExtractor(docs)
        page_0_images = extractor.get_images_by_page(0)

        assert isinstance(page_0_images, list)
        # 모든 이미지가 page 0에 속해야 함
        assert all(img["page"] == 0 for img in page_0_images)

    def test_get_images_by_size(self):
        """크기별 이미지 필터링 테스트"""
        if not IMAGES_PDF.exists():
            pytest.skip(f"Test fixture not found: {IMAGES_PDF}")

        loader = beanPDFLoader(IMAGES_PDF, extract_images=True, strategy="fast")
        docs = loader.load()

        extractor = ImageExtractor(docs)
        large_images = extractor.get_images_by_size(min_width=50, min_height=50)

        assert isinstance(large_images, list)
        # 모든 이미지가 조건을 만족해야 함
        assert all(img["width"] >= 50 and img["height"] >= 50 for img in large_images)

    def test_get_large_images(self):
        """큰 이미지 필터링 테스트"""
        if not IMAGES_PDF.exists():
            pytest.skip(f"Test fixture not found: {IMAGES_PDF}")

        loader = beanPDFLoader(IMAGES_PDF, extract_images=True, strategy="fast")
        docs = loader.load()

        extractor = ImageExtractor(docs)
        large = extractor.get_large_images(min_dimension=100)

        assert isinstance(large, list)
        # 모든 이미지의 width 또는 height가 100 이상이어야 함
        assert all(img["width"] >= 100 or img["height"] >= 100 for img in large)

    def test_get_summary(self):
        """요약 정보 테스트"""
        if not IMAGES_PDF.exists():
            pytest.skip(f"Test fixture not found: {IMAGES_PDF}")

        loader = beanPDFLoader(IMAGES_PDF, extract_images=True, strategy="fast")
        docs = loader.load()

        extractor = ImageExtractor(docs)
        summary = extractor.get_summary()

        assert "total_images" in summary
        assert "pages_with_images" in summary
        assert "images_by_page" in summary
        assert "formats" in summary
        assert "avg_width" in summary
        assert "avg_height" in summary
        assert "total_size" in summary

    def test_export_manifest(self):
        """매니페스트 내보내기 테스트"""
        if not IMAGES_PDF.exists():
            pytest.skip(f"Test fixture not found: {IMAGES_PDF}")

        loader = beanPDFLoader(IMAGES_PDF, extract_images=True, strategy="fast")
        docs = loader.load()

        extractor = ImageExtractor(docs)
        manifest = extractor.export_manifest()

        assert isinstance(manifest, str)
        assert "# Image Manifest" in manifest

    def test_no_images(self):
        """이미지가 없는 문서 테스트"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")

        loader = beanPDFLoader(SIMPLE_PDF, extract_images=True, strategy="fast")
        docs = loader.load()

        extractor = ImageExtractor(docs)
        images = extractor.get_all_images()

        assert isinstance(images, list)
        # simple.pdf에는 이미지가 없을 가능성이 높음

        summary = extractor.get_summary()
        assert summary["total_images"] == 0
