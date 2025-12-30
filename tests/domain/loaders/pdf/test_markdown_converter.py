"""
MarkdownConverter 단위 테스트
"""

import pytest
from beanllm.domain.loaders.pdf.utils.markdown_converter import MarkdownConverter


class TestMarkdownConverter:
    """MarkdownConverter 테스트"""

    @pytest.fixture
    def converter(self):
        """기본 converter 인스턴스"""
        return MarkdownConverter()

    @pytest.fixture
    def sample_result(self):
        """샘플 PDF 추출 결과"""
        return {
            "pages": [
                {
                    "page": 0,
                    "text": "This is a simple text.\nAnother line here.",
                    "width": 612.0,
                    "height": 792.0,
                    "metadata": {},
                },
                {
                    "page": 1,
                    "text": "Second page content.",
                    "width": 612.0,
                    "height": 792.0,
                    "metadata": {},
                },
            ],
            "tables": [],
            "images": [],
            "metadata": {"total_pages": 2, "engine": "PyMuPDF"},
        }

    @pytest.fixture
    def result_with_tables(self):
        """테이블이 포함된 결과"""
        return {
            "pages": [
                {
                    "page": 0,
                    "text": "Page with a table",
                    "width": 612.0,
                    "height": 792.0,
                    "metadata": {},
                }
            ],
            "tables": [
                {
                    "page": 0,
                    "table_index": 0,
                    "data": [
                        {"Name": "Alice", "Age": "30"},
                        {"Name": "Bob", "Age": "25"},
                    ],
                    "bbox": (100, 100, 500, 200),
                    "confidence": 0.95,
                }
            ],
            "images": [],
            "metadata": {"total_pages": 1},
        }

    @pytest.fixture
    def result_with_images(self):
        """이미지가 포함된 결과"""
        return {
            "pages": [
                {
                    "page": 0,
                    "text": "Page with an image",
                    "width": 612.0,
                    "height": 792.0,
                    "metadata": {},
                }
            ],
            "tables": [],
            "images": [
                {
                    "page": 0,
                    "image_index": 0,
                    "format": "png",
                    "width": 800,
                    "height": 600,
                    "bbox": (50, 50, 550, 450),
                    "size": 102400,
                }
            ],
            "metadata": {"total_pages": 1},
        }

    def test_basic_conversion(self, converter, sample_result):
        """기본 변환 테스트"""
        markdown = converter.convert_to_markdown(sample_result)

        # 페이지 헤더 확인
        assert "# Page 1" in markdown
        assert "# Page 2" in markdown

        # 텍스트 내용 확인
        assert "This is a simple text." in markdown
        assert "Second page content." in markdown

        # 페이지 구분자 확인
        assert "---" in markdown

    def test_conversion_with_tables(self, converter, result_with_tables):
        """테이블 변환 테스트"""
        markdown = converter.convert_to_markdown(result_with_tables)

        # 테이블 섹션 헤더 확인
        assert "## Tables" in markdown

        # Markdown 테이블 형식 확인
        assert "| Name | Age |" in markdown
        assert "| --- | --- |" in markdown
        assert "| Alice | 30 |" in markdown
        assert "| Bob | 25 |" in markdown

    def test_conversion_with_images(self, converter, result_with_images):
        """이미지 변환 테스트"""
        markdown = converter.convert_to_markdown(result_with_images)

        # 이미지 섹션 헤더 확인
        assert "## Images" in markdown

        # 이미지 링크 확인
        assert "![Image 1](image_p1_0.png)" in markdown

        # 이미지 크기 정보 확인
        assert "800x600 pixels" in markdown

    def test_table_conversion_with_2d_list(self, converter):
        """2D 리스트 형식 테이블 변환 테스트"""
        table = {
            "page": 0,
            "table_index": 0,
            "data": [
                ["Header1", "Header2"],
                ["Data1", "Data2"],
                ["Data3", "Data4"],
            ],
            "bbox": (0, 0, 100, 100),
        }

        markdown = converter._convert_table_to_markdown(table)

        # Markdown 테이블 형식 확인
        assert "| Header1 | Header2 |" in markdown
        assert "| --- | --- |" in markdown
        assert "| Data1 | Data2 |" in markdown
        assert "| Data3 | Data4 |" in markdown

    def test_image_conversion(self, converter):
        """이미지 링크 변환 테스트"""
        image = {
            "page": 2,
            "image_index": 3,
            "format": "jpeg",
            "width": 1024,
            "height": 768,
            "bbox": (0, 0, 100, 100),
        }

        markdown = converter._convert_image_to_markdown(image)

        # 이미지 링크 확인 (page 2 = Page 3)
        assert "![Image 4](image_p3_3.jpeg)" in markdown
        assert "1024x768 pixels" in markdown

    def test_clean_text(self, converter):
        """텍스트 정리 테스트"""
        # 연속된 빈 줄 제거
        text = "Line 1\n\n\n\nLine 2\n\n\n\n\nLine 3"
        cleaned = converter._clean_text(text)

        # 최대 2개의 연속된 줄바꿈만 허용
        assert "\n\n\n" not in cleaned
        assert "Line 1" in cleaned
        assert "Line 2" in cleaned
        assert "Line 3" in cleaned

    def test_group_by_page(self, converter):
        """페이지별 그룹화 테스트"""
        items = [
            {"page": 0, "data": "item1"},
            {"page": 0, "data": "item2"},
            {"page": 1, "data": "item3"},
            {"page": 2, "data": "item4"},
        ]

        grouped = converter._group_by_page(items)

        assert len(grouped) == 3
        assert len(grouped[0]) == 2
        assert len(grouped[1]) == 1
        assert len(grouped[2]) == 1

    def test_empty_result(self, converter):
        """빈 결과 변환 테스트"""
        result = {
            "pages": [],
            "tables": [],
            "images": [],
            "metadata": {},
        }

        markdown = converter.convert_to_markdown(result)

        # 빈 문자열 반환
        assert markdown == ""

    def test_custom_page_separator(self):
        """커스텀 페이지 구분자 테스트"""
        converter = MarkdownConverter(page_separator="\n\n***\n\n")

        result = {
            "pages": [
                {"page": 0, "text": "Page 1", "width": 612, "height": 792, "metadata": {}},
                {"page": 1, "text": "Page 2", "width": 612, "height": 792, "metadata": {}},
            ],
            "tables": [],
            "images": [],
            "metadata": {},
        }

        markdown = converter.convert_to_markdown(result)

        # 커스텀 구분자 확인
        assert "***" in markdown
        assert "---" not in markdown

    def test_custom_image_prefix(self):
        """커스텀 이미지 접두사 테스트"""
        converter = MarkdownConverter(image_prefix="fig")

        image = {
            "page": 0,
            "image_index": 0,
            "format": "png",
            "width": 800,
            "height": 600,
            "bbox": (0, 0, 100, 100),
        }

        markdown = converter._convert_image_to_markdown(image)

        # 커스텀 접두사 확인
        assert "fig_p1_0.png" in markdown
