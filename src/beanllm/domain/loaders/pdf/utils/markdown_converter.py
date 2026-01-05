"""
Markdown Converter - PDF to Markdown 변환기

PDFLoadResult를 Markdown 형식으로 변환합니다.

Features:
- 텍스트 → Markdown 변환
- 제목 레벨 자동 감지 (폰트 크기 기반)
- 테이블 → Markdown 테이블
- 이미지 → ![image](path) 링크
- 페이지 구분자 삽입
"""

import re
from typing import Dict, List, Optional

try:
    from beanllm.utils.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


class MarkdownConverter:
    """
    PDF 추출 결과를 Markdown으로 변환

    Example:
        ```python
        from beanllm.domain.loaders.pdf.utils import MarkdownConverter
        from beanllm.domain.loaders.pdf import beanPDFLoader

        loader = beanPDFLoader("document.pdf", extract_tables=True)
        docs = loader.load()

        converter = MarkdownConverter()
        markdown = converter.convert_to_markdown(loader._result)
        print(markdown)
        ```
    """

    def __init__(
        self,
        page_separator: str = "\n\n---\n\n",
        heading_threshold: float = 1.2,
        image_prefix: str = "image",
    ):
        """
        Args:
            page_separator: 페이지 구분자 (기본: "\\n\\n---\\n\\n")
            heading_threshold: 제목 감지 임계값 (평균 폰트 크기 대비 배율, 기본: 1.2)
            image_prefix: 이미지 파일명 접두사 (기본: "image")
        """
        self.page_separator = page_separator
        self.heading_threshold = heading_threshold
        self.image_prefix = image_prefix

    def convert_to_markdown(self, result: Dict) -> str:
        """
        PDFLoadResult를 Markdown으로 변환

        Args:
            result: PDFLoadResult.to_dict() 또는 extract() 결과

        Returns:
            str: Markdown 형식 텍스트
        """
        markdown_parts = []
        pages = result.get("pages", [])
        tables = result.get("tables", [])
        images = result.get("images", [])

        # 테이블 및 이미지를 페이지별로 그룹화
        tables_by_page = self._group_by_page(tables)
        images_by_page = self._group_by_page(images)

        for page_data in pages:
            page_num = page_data.get("page", 0)
            text = page_data.get("text", "")
            metadata = page_data.get("metadata", {})

            # 페이지 헤더
            page_markdown = f"# Page {page_num + 1}\n\n"

            # 제목 감지 및 텍스트 변환
            if metadata.get("fonts") or metadata.get("blocks"):
                # 폰트 정보가 있으면 제목 감지
                converted_text = self._convert_text_with_headings(text, metadata)
            else:
                # 폰트 정보가 없으면 일반 텍스트
                converted_text = self._clean_text(text)

            page_markdown += converted_text

            # 테이블 추가
            if page_num in tables_by_page:
                page_markdown += "\n\n## Tables\n\n"
                for table in tables_by_page[page_num]:
                    table_md = self._convert_table_to_markdown(table)
                    page_markdown += table_md + "\n\n"

            # 이미지 추가
            if page_num in images_by_page:
                page_markdown += "\n\n## Images\n\n"
                for image in images_by_page[page_num]:
                    image_md = self._convert_image_to_markdown(image)
                    page_markdown += image_md + "\n\n"

            markdown_parts.append(page_markdown.strip())

        return self.page_separator.join(markdown_parts)

    def _group_by_page(self, items: List[Dict]) -> Dict[int, List[Dict]]:
        """
        아이템을 페이지별로 그룹화

        Args:
            items: 테이블 또는 이미지 리스트

        Returns:
            Dict[int, List[Dict]]: 페이지 번호를 키로 하는 딕셔너리
        """
        grouped = {}
        for item in items:
            page = item.get("page", 0)
            if page not in grouped:
                grouped[page] = []
            grouped[page].append(item)
        return grouped

    def _convert_text_with_headings(self, text: str, metadata: Dict) -> str:
        """
        폰트 크기 기반 제목 감지 및 변환

        Args:
            text: 원본 텍스트
            metadata: 페이지 메타데이터 (폰트 정보 포함)

        Returns:
            str: Markdown 형식 텍스트
        """
        # 폰트 정보에서 평균 크기 계산
        fonts = metadata.get("fonts", [])
        if not fonts:
            return self._clean_text(text)

        # 평균 폰트 크기 계산
        font_sizes = [font.get("size", 12.0) for font in fonts if "size" in font]
        if not font_sizes:
            return self._clean_text(text)

        avg_size = sum(font_sizes) / len(font_sizes)

        # 제목 감지 (평균 크기의 heading_threshold배 이상)
        headings = self._detect_headings(metadata, avg_size)

        # 제목이 없으면 일반 텍스트
        if not headings:
            return self._clean_text(text)

        # 텍스트를 줄 단위로 분리하고 제목 변환
        lines = text.split("\n")
        converted_lines = []

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                converted_lines.append("")
                continue

            # 제목 여부 확인
            heading_level = self._get_heading_level(line_stripped, headings)
            if heading_level > 0:
                converted_lines.append(f"{'#' * heading_level} {line_stripped}")
            else:
                converted_lines.append(line_stripped)

        return "\n".join(converted_lines)

    def _detect_headings(self, metadata: Dict, avg_size: float) -> List[Dict]:
        """
        폰트 크기 기반 제목 감지

        Args:
            metadata: 페이지 메타데이터
            avg_size: 평균 폰트 크기

        Returns:
            List[Dict]: 제목 정보 리스트
                - text: 제목 텍스트
                - size: 폰트 크기
                - level: 제목 레벨 (1-3)
        """
        headings = []
        fonts = metadata.get("fonts", [])

        for font in fonts:
            size = font.get("size", avg_size)
            text = font.get("text", "")

            # 평균 크기 이상이고 텍스트가 있으면 제목으로 판단
            if size >= avg_size * self.heading_threshold and text.strip():
                # 크기에 따라 레벨 결정
                if size >= avg_size * 2.0:
                    level = 1
                elif size >= avg_size * 1.5:
                    level = 2
                else:
                    level = 3

                headings.append({"text": text.strip(), "size": size, "level": level})

        return headings

    def _get_heading_level(self, line: str, headings: List[Dict]) -> int:
        """
        라인이 제목인지 확인하고 레벨 반환

        Args:
            line: 텍스트 라인
            headings: 제목 리스트

        Returns:
            int: 제목 레벨 (0이면 제목 아님)
        """
        for heading in headings:
            if heading["text"] in line:
                return heading["level"]
        return 0

    def _clean_text(self, text: str) -> str:
        """
        텍스트 정리 (불필요한 공백 제거 등)

        Args:
            text: 원본 텍스트

        Returns:
            str: 정리된 텍스트
        """
        # 연속된 빈 줄 제거 (최대 2개까지만 유지)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 줄 끝 공백 제거
        lines = [line.rstrip() for line in text.split("\n")]

        return "\n".join(lines).strip()

    def _convert_table_to_markdown(self, table: Dict) -> str:
        """
        테이블을 Markdown 테이블로 변환

        Args:
            table: TableData.to_dict() 결과

        Returns:
            str: Markdown 테이블
        """
        data = table.get("data", [])
        if not data:
            return ""

        # DataFrame인 경우 (to_dict("records") 형식)
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            # 헤더 추출
            headers = list(data[0].keys())
            rows = [[str(row.get(h, "")) for h in headers] for row in data]

            # Markdown 테이블 생성
            markdown = "| " + " | ".join(headers) + " |\n"
            markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"

            for row in rows:
                markdown += "| " + " | ".join(row) + " |\n"

            return markdown.strip()

        # 2D 리스트인 경우
        elif isinstance(data, list) and len(data) > 0:
            # 첫 행을 헤더로 사용
            headers = [str(cell) for cell in data[0]]
            rows = [[str(cell) for cell in row] for row in data[1:]]

            # Markdown 테이블 생성
            markdown = "| " + " | ".join(headers) + " |\n"
            markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"

            for row in rows:
                markdown += "| " + " | ".join(row) + " |\n"

            return markdown.strip()

        return ""

    def _convert_image_to_markdown(self, image: Dict) -> str:
        """
        이미지를 Markdown 이미지 링크로 변환

        Args:
            image: ImageData.to_dict() 결과

        Returns:
            str: Markdown 이미지 링크
        """
        page = image.get("page", 0)
        image_index = image.get("image_index", 0)
        format_ext = image.get("format", "png")
        width = image.get("width", 0)
        height = image.get("height", 0)

        # 이미지 파일명 생성
        filename = f"{self.image_prefix}_p{page + 1}_{image_index}.{format_ext}"

        # Markdown 이미지 링크
        markdown = f"![Image {image_index + 1}]({filename})"

        # 이미지 크기 정보 추가 (선택적)
        if width > 0 and height > 0:
            markdown += f"\n*Size: {width}x{height} pixels*"

        return markdown
