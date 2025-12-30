"""
LayoutAnalyzer 단위 테스트
"""

import pytest
from beanllm.domain.loaders.pdf.utils.layout_analyzer import LayoutAnalyzer, Block


class TestLayoutAnalyzer:
    """LayoutAnalyzer 테스트"""

    @pytest.fixture
    def analyzer(self):
        """기본 analyzer 인스턴스"""
        return LayoutAnalyzer()

    @pytest.fixture
    def sample_page_data(self):
        """샘플 페이지 데이터"""
        return {
            "text": "Sample text content",
            "width": 612.0,
            "height": 792.0,
            "metadata": {
                "blocks": [
                    {
                        "type": "text",
                        "bbox": (50, 100, 400, 150),
                        "text": "First block",
                        "size": 12.0,
                    },
                    {
                        "type": "text",
                        "bbox": (50, 200, 400, 250),
                        "text": "Second block",
                        "size": 12.0,
                    },
                ]
            },
        }

    @pytest.fixture
    def multi_column_page(self):
        """다단 레이아웃 페이지 데이터"""
        return {
            "text": "Multi-column content",
            "width": 612.0,
            "height": 792.0,
            "metadata": {
                "blocks": [
                    # 왼쪽 컬럼
                    {"type": "text", "bbox": (50, 100, 250, 150), "text": "Left col 1"},
                    {"type": "text", "bbox": (50, 200, 250, 250), "text": "Left col 2"},
                    # 오른쪽 컬럼
                    {"type": "text", "bbox": (300, 100, 500, 150), "text": "Right col 1"},
                    {"type": "text", "bbox": (300, 200, 500, 250), "text": "Right col 2"},
                ]
            },
        }

    def test_analyzer_initialization(self):
        """Analyzer 초기화 테스트"""
        analyzer = LayoutAnalyzer(
            header_threshold=0.85,
            footer_threshold=0.15,
            multi_column_gap=40.0,
            heading_size_ratio=1.5,
        )

        assert analyzer.header_threshold == 0.85
        assert analyzer.footer_threshold == 0.15
        assert analyzer.multi_column_gap == 40.0
        assert analyzer.heading_size_ratio == 1.5

    def test_detect_blocks(self, analyzer, sample_page_data):
        """블록 감지 테스트"""
        blocks = analyzer.detect_blocks(sample_page_data)

        assert len(blocks) == 2
        assert all(isinstance(b, Block) for b in blocks)
        assert blocks[0].content == "First block"
        assert blocks[1].content == "Second block"

    def test_detect_blocks_no_metadata(self, analyzer):
        """메타데이터 없는 경우 블록 감지 테스트"""
        page_data = {
            "text": "Simple text content",
            "width": 612.0,
            "height": 792.0,
            "metadata": {},
        }

        blocks = analyzer.detect_blocks(page_data)

        assert len(blocks) == 1
        assert blocks[0].block_type == "text"
        assert blocks[0].content == "Simple text content"

    def test_restore_single_column_order(self, analyzer):
        """단일 컬럼 읽기 순서 복원 테스트"""
        blocks = [
            Block("text", (50, 200, 400, 250), "Block 2"),
            Block("text", (50, 100, 400, 150), "Block 1"),
            Block("text", (50, 300, 400, 350), "Block 3"),
        ]

        reading_order = analyzer._restore_single_column_order(blocks)

        # y0 기준 정렬: Block 1 (y0=100), Block 2 (y0=200), Block 3 (y0=300)
        assert reading_order == [1, 0, 2]

    def test_restore_multi_column_order(self, analyzer):
        """다단 컬럼 읽기 순서 복원 테스트"""
        blocks = [
            Block("text", (50, 200, 250, 250), "Left 2"),  # 0
            Block("text", (50, 100, 250, 150), "Left 1"),  # 1
            Block("text", (300, 200, 500, 250), "Right 2"),  # 2
            Block("text", (300, 100, 500, 150), "Right 1"),  # 3
        ]

        reading_order = analyzer._restore_multi_column_order(blocks)

        # 왼쪽 컬럼 먼저, 각 컬럼 내에서 위→아래
        # Left 1 (idx=1), Left 2 (idx=0), Right 1 (idx=3), Right 2 (idx=2)
        assert reading_order == [1, 0, 3, 2]

    def test_detect_multi_column_true(self, analyzer):
        """다단 레이아웃 감지 (True) 테스트"""
        blocks = [
            Block("text", (50, 100, 250, 150), "Left"),
            Block("text", (300, 100, 500, 150), "Right"),
        ]

        is_multi = analyzer.detect_multi_column(blocks, 612.0)

        assert is_multi is True

    def test_detect_multi_column_false(self, analyzer):
        """다단 레이아웃 감지 (False) 테스트"""
        blocks = [
            Block("text", (50, 100, 400, 150), "Block 1"),
            Block("text", (50, 200, 400, 250), "Block 2"),
        ]

        is_multi = analyzer.detect_multi_column(blocks, 612.0)

        assert is_multi is False

    def test_remove_header_footer(self, analyzer):
        """헤더/푸터 제거 테스트"""
        page_height = 792.0
        blocks = [
            Block("text", (50, 50, 400, 70), "Footer"),  # y0=50 (footer area)
            Block("text", (50, 100, 400, 150), "Content 1"),
            Block("text", (50, 400, 400, 450), "Content 2"),
            Block("text", (50, 720, 400, 750), "Header"),  # y0=720 (header area)
        ]

        filtered = analyzer.remove_header_footer(blocks, page_height)

        # Content 블록만 남아야 함
        assert len(filtered) == 2
        assert filtered[0].content == "Content 1"
        assert filtered[1].content == "Content 2"

    def test_analyze_layout_single_column(self, analyzer, sample_page_data):
        """단일 컬럼 레이아웃 분석 테스트"""
        result = analyzer.analyze_layout(sample_page_data)

        assert "blocks" in result
        assert "reading_order" in result
        assert "is_multi_column" in result
        assert "columns" in result

        assert result["is_multi_column"] is False
        assert result["columns"] == 1
        assert len(result["blocks"]) == 2

    def test_analyze_layout_multi_column(self, analyzer, multi_column_page):
        """다단 컬럼 레이아웃 분석 테스트"""
        result = analyzer.analyze_layout(multi_column_page)

        assert result["is_multi_column"] is True
        assert result["columns"] == 2
        assert len(result["blocks"]) == 4

    def test_count_columns(self, analyzer):
        """컬럼 수 계산 테스트"""
        # 2개 컬럼
        blocks_2col = [
            Block("text", (50, 100, 250, 150), "Left"),
            Block("text", (300, 100, 500, 150), "Right"),
        ]

        count = analyzer._count_columns(blocks_2col, 612.0)
        assert count == 2

        # 1개 컬럼
        blocks_1col = [
            Block("text", (50, 100, 400, 150), "Block 1"),
            Block("text", (50, 200, 400, 250), "Block 2"),
        ]

        count = analyzer._count_columns(blocks_1col, 612.0)
        assert count == 1

    def test_cluster_coordinates(self, analyzer):
        """좌표 클러스터링 테스트"""
        coords = [50, 55, 60, 300, 305, 310]

        clusters = analyzer._cluster_coordinates(coords, threshold=20.0)

        # 2개의 클러스터: [50, 55, 60], [300, 305, 310]
        assert len(clusters) == 2
        assert len(clusters[0]) == 3
        assert len(clusters[1]) == 3
