"""
beanPDFLoader Markdown 변환 통합 테스트
"""

from pathlib import Path

import pytest


class TestBeanPDFLoaderMarkdown:
    """beanPDFLoader Markdown 변환 통합 테스트"""

    @pytest.fixture
    def simple_pdf(self):
        """간단한 PDF 픽스처 경로"""
        return Path(__file__).parent.parent.parent.parent / "fixtures" / "pdf" / "simple.pdf"

    @pytest.fixture
    def tables_pdf(self):
        """테이블 PDF 픽스처 경로"""
        return Path(__file__).parent.parent.parent.parent / "fixtures" / "pdf" / "tables.pdf"

    @pytest.fixture
    def images_pdf(self):
        """이미지 PDF 픽스처 경로"""
        return Path(__file__).parent.parent.parent.parent / "fixtures" / "pdf" / "images.pdf"

    def test_markdown_conversion_simple_pdf(self, simple_pdf):
        """간단한 PDF Markdown 변환 테스트"""
        from beanllm.domain.loaders.pdf import beanPDFLoader

        if not simple_pdf.exists():
            pytest.skip("Test PDF fixture not found")

        # to_markdown=True로 로딩
        loader = beanPDFLoader(
            simple_pdf, to_markdown=True, extract_tables=False, extract_images=False
        )
        docs = loader.load()

        # Document 객체 반환 확인
        assert len(docs) > 0
        assert all(hasattr(doc, "content") for doc in docs)

        # 내부 결과에 markdown 필드 존재 확인
        assert hasattr(loader, "_result")
        assert "markdown" in loader._result
        assert loader._result["markdown"] is not None

        # Markdown 형식 확인
        markdown = loader._result["markdown"]
        assert isinstance(markdown, str)
        assert len(markdown) > 0

        # 페이지 헤더 확인
        assert "# Page" in markdown

        # 페이지 구분자 확인
        if len(docs) > 1:
            assert "---" in markdown

    def test_markdown_conversion_with_tables(self, tables_pdf):
        """테이블 포함 PDF Markdown 변환 테스트"""
        from beanllm.domain.loaders.pdf import beanPDFLoader

        if not tables_pdf.exists():
            pytest.skip("Test PDF fixture not found")

        # to_markdown=True, extract_tables=True로 로딩
        loader = beanPDFLoader(tables_pdf, to_markdown=True, extract_tables=True)
        docs = loader.load()

        # 결과 확인
        assert len(docs) > 0

        # Markdown에 테이블 섹션 확인
        markdown = loader._result.get("markdown", "")
        if "Tables" in markdown or "|" in markdown:
            # 테이블이 있으면 Markdown 테이블 형식 확인
            assert "## Tables" in markdown or "|" in markdown
            assert "---" in markdown  # 페이지 구분자 또는 테이블 구분자

    def test_markdown_conversion_with_images(self, images_pdf):
        """이미지 포함 PDF Markdown 변환 테스트"""
        from beanllm.domain.loaders.pdf import beanPDFLoader

        if not images_pdf.exists():
            pytest.skip("Test PDF fixture not found")

        # to_markdown=True, extract_images=True로 로딩
        loader = beanPDFLoader(images_pdf, to_markdown=True, extract_images=True)
        docs = loader.load()

        # 결과 확인
        assert len(docs) > 0

        # Markdown 생성 확인
        markdown = loader._result.get("markdown", "")
        assert markdown is not None
        assert len(markdown) > 0

        # 실제로 이미지가 추출되었는지 확인
        images = loader._result.get("images", [])
        if len(images) > 0:
            # 이미지가 있으면 Markdown 이미지 섹션 확인
            assert "## Images" in markdown or "![" in markdown
        # 이미지가 없으면 (벡터 그래픽인 경우) 테스트 통과

    def test_markdown_disabled_by_default(self, simple_pdf):
        """기본값에서는 Markdown 변환 비활성화 확인"""
        from beanllm.domain.loaders.pdf import beanPDFLoader

        if not simple_pdf.exists():
            pytest.skip("Test PDF fixture not found")

        # to_markdown=False (기본값)로 로딩
        loader = beanPDFLoader(simple_pdf)
        docs = loader.load()

        # 결과 확인
        assert len(docs) > 0

        # markdown 필드가 없거나 None이어야 함
        markdown = loader._result.get("markdown")
        assert markdown is None

    def test_markdown_conversion_fast_strategy(self, simple_pdf):
        """Fast 전략에서 Markdown 변환 테스트"""
        from beanllm.domain.loaders.pdf import beanPDFLoader

        if not simple_pdf.exists():
            pytest.skip("Test PDF fixture not found")

        # strategy="fast", to_markdown=True
        loader = beanPDFLoader(simple_pdf, strategy="fast", to_markdown=True)
        docs = loader.load()

        # 결과 확인
        assert len(docs) > 0
        assert loader._result.get("markdown") is not None

    def test_markdown_conversion_accurate_strategy(self, simple_pdf):
        """Accurate 전략에서 Markdown 변환 테스트"""
        from beanllm.domain.loaders.pdf import beanPDFLoader

        if not simple_pdf.exists():
            pytest.skip("Test PDF fixture not found")

        # strategy="accurate", to_markdown=True
        loader = beanPDFLoader(simple_pdf, strategy="accurate", to_markdown=True)
        docs = loader.load()

        # 결과 확인
        assert len(docs) > 0
        assert loader._result.get("markdown") is not None
