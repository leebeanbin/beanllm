"""
Marker Engine - Core Extraction Mixin

marker-pdf 결과에서 텍스트, 테이블, 이미지를 추출하는 로직을 담당합니다.
"""

import re
from typing import Any, Dict, List


class MarkerExtractionMixin:
    """
    marker-pdf 추출 결과를 파싱하는 Mixin.

    - 페이지 분리
    - Markdown 테이블 추출
    - 이미지 형식 변환
    """

    def _split_into_pages(self, full_text: str, metadata: Dict) -> List[Dict]:
        """
        전체 Markdown 텍스트를 페이지별로 분리

        Args:
            full_text: 전체 Markdown 텍스트
            metadata: marker-pdf 메타데이터

        Returns:
            List[Dict]: 페이지 데이터 리스트
        """
        num_pages = metadata.get("num_pages", 1)

        if num_pages == 1:
            return [
                {
                    "page": 0,
                    "text": full_text,
                    "width": 612.0,
                    "height": 792.0,
                    "metadata": {"source": "marker-pdf"},
                }
            ]

        text_length = len(full_text)
        chunk_size = text_length // num_pages

        pages = []
        for i in range(num_pages):
            start = i * chunk_size
            end = start + chunk_size if i < num_pages - 1 else text_length

            pages.append(
                {
                    "page": i,
                    "text": full_text[start:end],
                    "width": 612.0,
                    "height": 792.0,
                    "metadata": {"source": "marker-pdf"},
                }
            )

        return pages

    def _extract_tables_from_markdown(self, markdown_text: str, pages: List[Dict]) -> List[Dict]:
        """
        Markdown 텍스트에서 테이블 추출

        Args:
            markdown_text: Markdown 텍스트
            pages: 페이지 데이터 리스트

        Returns:
            List[Dict]: 테이블 데이터 리스트
        """
        tables: List[Dict[str, Any]] = []

        table_pattern = r"\|[^\n]+\|\n\|[-\s|]+\|\n(?:\|[^\n]+\|\n)+"

        for match in re.finditer(table_pattern, markdown_text):
            table_text = match.group()

            position = match.start()
            page_idx = self._estimate_page_from_position(position, len(markdown_text), len(pages))

            table_data = self._parse_markdown_table(table_text)

            if table_data:
                tables.append(
                    {
                        "page": page_idx,
                        "table_index": len([t for t in tables if t["page"] == page_idx]),
                        "data": table_data,
                        "bbox": (0, 0, 612, 100),
                        "confidence": 0.95,
                        "metadata": {"source": "marker-pdf", "format": "markdown"},
                    }
                )

        return tables

    def _estimate_page_from_position(self, position: int, total_length: int, num_pages: int) -> int:
        """
        텍스트 위치에서 페이지 번호 추정

        Args:
            position: 텍스트 내 위치
            total_length: 전체 텍스트 길이
            num_pages: 총 페이지 수

        Returns:
            int: 추정된 페이지 인덱스 (0-based)
        """
        if num_pages == 0:
            return 0
        page_idx = int((position / total_length) * num_pages)
        return min(page_idx, num_pages - 1)

    def _parse_markdown_table(self, table_text: str) -> List[List[str]]:
        """
        Markdown 테이블 파싱

        Args:
            table_text: Markdown 테이블 텍스트

        Returns:
            List[List[str]]: 2D 리스트 형식 테이블 데이터
        """
        lines = table_text.strip().split("\n")
        if len(lines) < 3:
            return []

        table_data = []

        for i, line in enumerate(lines):
            if i == 1:
                continue

            cells = [cell.strip() for cell in line.split("|")[1:-1]]
            if cells:
                table_data.append(cells)

        return table_data

    def _convert_images(self, images: Dict) -> List[Dict]:
        """
        marker-pdf 이미지를 ImageData 형식으로 변환

        Args:
            images: marker-pdf 이미지 딕셔너리

        Returns:
            List[Dict]: 이미지 데이터 리스트
        """
        image_list = []

        for idx, (image_name, image_data) in enumerate(images.items()):
            image_list.append(
                {
                    "page": 0,
                    "image_index": idx,
                    "image": image_data,
                    "format": "png",
                    "width": 800,
                    "height": 600,
                    "bbox": (0, 0, 800, 600),
                    "size": len(image_data) if isinstance(image_data, bytes) else 0,
                    "metadata": {"source": "marker-pdf", "name": image_name},
                }
            )

        return image_list
