"""
Layout Analyzer - PDF 레이아웃 분석

PDF 문서의 레이아웃을 분석하여 구조화된 정보를 추출합니다.

Features:
- 블록 감지 (제목, 본문, 표, 이미지)
- Reading order 복원
- 다단 레이아웃 처리
- 헤더/푸터 제거
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    from beanllm.utils.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


@dataclass
class Block:
    """레이아웃 블록"""

    block_type: str  # "heading", "text", "table", "image"
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    content: str
    confidence: float = 1.0
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LayoutAnalyzer:
    """
    PDF 레이아웃 분석기

    레이아웃 구조를 분석하여 읽기 순서를 복원하고 블록을 감지합니다.

    Example:
        ```python
        from beanllm.domain.loaders.pdf.utils import LayoutAnalyzer

        analyzer = LayoutAnalyzer()
        layout_info = analyzer.analyze_layout(page_data)
        blocks = layout_info["blocks"]
        reading_order = layout_info["reading_order"]
        ```
    """

    def __init__(
        self,
        header_threshold: float = 0.9,
        footer_threshold: float = 0.1,
        multi_column_gap: float = 30.0,
        heading_size_ratio: float = 1.2,
    ):
        """
        Args:
            header_threshold: 헤더 감지 임계값 (페이지 높이 비율, 0-1)
            footer_threshold: 푸터 감지 임계값 (페이지 높이 비율, 0-1)
            multi_column_gap: 다단 레이아웃 감지를 위한 최소 간격 (포인트)
            heading_size_ratio: 제목 감지를 위한 폰트 크기 비율
        """
        self.header_threshold = header_threshold
        self.footer_threshold = footer_threshold
        self.multi_column_gap = multi_column_gap
        self.heading_size_ratio = heading_size_ratio

    def analyze_layout(self, page_data: Dict) -> Dict:
        """
        페이지 레이아웃 분석

        Args:
            page_data: 페이지 데이터 딕셔너리
                - text: 텍스트 내용
                - width: 페이지 너비
                - height: 페이지 높이
                - metadata: 메타데이터 (블록, 폰트 정보 등)

        Returns:
            Dict: 레이아웃 분석 결과
                - blocks: 감지된 블록 리스트
                - reading_order: 읽기 순서 인덱스 리스트
                - is_multi_column: 다단 레이아웃 여부
                - columns: 감지된 컬럼 수
        """
        page_width = page_data.get("width", 612.0)
        page_height = page_data.get("height", 792.0)
        metadata = page_data.get("metadata", {})

        # 블록 감지
        blocks = self.detect_blocks(page_data)

        # 헤더/푸터 제거
        blocks = self.remove_header_footer(blocks, page_height)

        # 다단 레이아웃 감지
        is_multi_column = self.detect_multi_column(blocks, page_width)
        columns = self._count_columns(blocks, page_width) if is_multi_column else 1

        # Reading order 복원
        reading_order = self.restore_reading_order(blocks, is_multi_column)

        return {
            "blocks": blocks,
            "reading_order": reading_order,
            "is_multi_column": is_multi_column,
            "columns": columns,
        }

    def detect_blocks(self, page_data: Dict) -> List[Block]:
        """
        블록 감지 (제목, 본문, 표, 이미지)

        Args:
            page_data: 페이지 데이터

        Returns:
            List[Block]: 감지된 블록 리스트
        """
        blocks = []
        metadata = page_data.get("metadata", {})

        # PyMuPDF 블록 정보가 있는 경우
        if "blocks" in metadata:
            for block_info in metadata["blocks"]:
                block = self._parse_block(block_info, page_data)
                if block:
                    blocks.append(block)
        else:
            # 블록 정보가 없으면 텍스트를 단일 블록으로 처리
            text = page_data.get("text", "")
            if text.strip():
                block = Block(
                    block_type="text",
                    bbox=(0, 0, page_data.get("width", 612), page_data.get("height", 792)),
                    content=text,
                    confidence=0.5,
                )
                blocks.append(block)

        return blocks

    def _parse_block(self, block_info: Dict, page_data: Dict) -> Optional[Block]:
        """
        블록 정보 파싱

        Args:
            block_info: 블록 정보 딕셔너리
            page_data: 페이지 데이터

        Returns:
            Optional[Block]: 파싱된 블록 또는 None
        """
        block_type = block_info.get("type", "text")
        bbox = block_info.get("bbox", (0, 0, 100, 100))
        content = block_info.get("text", "")

        # 빈 블록 제외
        if not content.strip() and block_type == "text":
            return None

        # 제목 감지 (폰트 크기 기반)
        if block_type == "text" and self._is_heading(block_info, page_data):
            block_type = "heading"

        return Block(
            block_type=block_type,
            bbox=bbox,
            content=content,
            confidence=block_info.get("confidence", 1.0),
            metadata=block_info.get("metadata", {}),
        )

    def _is_heading(self, block_info: Dict, page_data: Dict) -> bool:
        """
        제목 여부 판단 (폰트 크기 기반)

        Args:
            block_info: 블록 정보
            page_data: 페이지 데이터

        Returns:
            bool: 제목 여부
        """
        # 폰트 정보가 있는 경우
        fonts = page_data.get("metadata", {}).get("fonts", [])
        if not fonts:
            return False

        # 평균 폰트 크기 계산
        font_sizes = [f.get("size", 12.0) for f in fonts if "size" in f]
        if not font_sizes:
            return False

        avg_size = sum(font_sizes) / len(font_sizes)

        # 블록의 폰트 크기
        block_size = block_info.get("size", avg_size)

        # 평균보다 heading_size_ratio배 이상 크면 제목
        return block_size >= avg_size * self.heading_size_ratio

    def restore_reading_order(
        self, blocks: List[Block], is_multi_column: bool = False
    ) -> List[int]:
        """
        읽기 순서 복원

        Args:
            blocks: 블록 리스트
            is_multi_column: 다단 레이아웃 여부

        Returns:
            List[int]: 읽기 순서 인덱스 리스트
        """
        if not blocks:
            return []

        if is_multi_column:
            return self._restore_multi_column_order(blocks)
        else:
            return self._restore_single_column_order(blocks)

    def _restore_single_column_order(self, blocks: List[Block]) -> List[int]:
        """
        단일 컬럼 읽기 순서 복원 (위→아래)

        Args:
            blocks: 블록 리스트

        Returns:
            List[int]: 읽기 순서 인덱스
        """
        # y0 (상단) 기준 정렬
        indexed_blocks = [(i, block) for i, block in enumerate(blocks)]
        sorted_blocks = sorted(indexed_blocks, key=lambda x: x[1].bbox[1])

        return [i for i, _ in sorted_blocks]

    def _restore_multi_column_order(self, blocks: List[Block]) -> List[int]:
        """
        다단 컬럼 읽기 순서 복원 (왼쪽→오른쪽, 위→아래)

        Args:
            blocks: 블록 리스트

        Returns:
            List[int]: 읽기 순서 인덱스
        """
        if not blocks:
            return []

        # 컬럼별로 블록 그룹화
        columns = self._group_by_columns(blocks)

        # 각 컬럼 내에서 y 좌표 기준 정렬
        reading_order = []
        for column_blocks in columns:
            # (index, block) 정렬
            sorted_column = sorted(column_blocks, key=lambda x: x[1].bbox[1])
            reading_order.extend([i for i, _ in sorted_column])

        return reading_order

    def _group_by_columns(self, blocks: List[Block]) -> List[List[Tuple[int, Block]]]:
        """
        블록을 컬럼별로 그룹화

        Args:
            blocks: 블록 리스트

        Returns:
            List[List[Tuple[int, Block]]]: 컬럼별 블록 리스트
        """
        indexed_blocks = [(i, block) for i, block in enumerate(blocks)]

        # x 좌표 기준 정렬
        sorted_blocks = sorted(indexed_blocks, key=lambda x: x[1].bbox[0])

        # 간격 기반 컬럼 분리
        columns = []
        current_column = []

        for i, (idx, block) in enumerate(sorted_blocks):
            if not current_column:
                current_column.append((idx, block))
            else:
                # 이전 블록과의 간격 확인
                prev_block = current_column[-1][1]
                gap = block.bbox[0] - prev_block.bbox[2]

                if gap > self.multi_column_gap:
                    # 새 컬럼 시작
                    columns.append(current_column)
                    current_column = [(idx, block)]
                else:
                    current_column.append((idx, block))

        # 마지막 컬럼 추가
        if current_column:
            columns.append(current_column)

        return columns

    def detect_multi_column(self, blocks: List[Block], page_width: float) -> bool:
        """
        다단 레이아웃 감지

        Args:
            blocks: 블록 리스트
            page_width: 페이지 너비

        Returns:
            bool: 다단 레이아웃 여부
        """
        if len(blocks) < 2:
            return False

        # x 좌표 기준으로 블록들 분석
        x_coords = [block.bbox[0] for block in blocks]

        # 블록들의 시작 x 좌표를 그룹화
        x_groups = self._cluster_coordinates(x_coords, threshold=self.multi_column_gap)

        # 2개 이상의 그룹이 있으면 다단 레이아웃
        return len(x_groups) >= 2

    def _cluster_coordinates(self, coords: List[float], threshold: float) -> List[List[float]]:
        """
        좌표를 임계값 기준으로 클러스터링

        Args:
            coords: 좌표 리스트
            threshold: 클러스터링 임계값

        Returns:
            List[List[float]]: 클러스터된 좌표 그룹
        """
        if not coords:
            return []

        sorted_coords = sorted(coords)
        clusters = [[sorted_coords[0]]]

        for coord in sorted_coords[1:]:
            if coord - clusters[-1][-1] <= threshold:
                clusters[-1].append(coord)
            else:
                clusters.append([coord])

        return clusters

    def _count_columns(self, blocks: List[Block], page_width: float) -> int:
        """
        컬럼 수 계산

        Args:
            blocks: 블록 리스트
            page_width: 페이지 너비

        Returns:
            int: 감지된 컬럼 수
        """
        if not blocks:
            return 1

        x_coords = [block.bbox[0] for block in blocks]
        x_groups = self._cluster_coordinates(x_coords, threshold=self.multi_column_gap)

        return len(x_groups)

    def remove_header_footer(
        self, blocks: List[Block], page_height: float
    ) -> List[Block]:
        """
        헤더/푸터 제거

        Args:
            blocks: 블록 리스트
            page_height: 페이지 높이

        Returns:
            List[Block]: 헤더/푸터가 제거된 블록 리스트
        """
        filtered_blocks = []

        header_boundary = page_height * self.header_threshold
        footer_boundary = page_height * self.footer_threshold

        for block in blocks:
            y0, y1 = block.bbox[1], block.bbox[3]

            # 헤더 영역 (페이지 상단)
            if y0 >= header_boundary:
                continue

            # 푸터 영역 (페이지 하단)
            if y1 <= footer_boundary:
                continue

            filtered_blocks.append(block)

        return filtered_blocks
