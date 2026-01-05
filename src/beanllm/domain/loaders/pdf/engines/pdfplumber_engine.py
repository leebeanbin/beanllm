"""
PDFPlumber Engine - Accurate Layer

pdfplumber를 사용한 정확한 PDF 파싱 엔진
- 속도: ~15초/100페이지
- 정확도: 95%
- 특화: 테이블 추출, 레이아웃 보존
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from .base import BasePDFEngine

try:
    from beanllm.utils.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


class PDFPlumberEngine(BasePDFEngine):
    """
    pdfplumber 기반 PDF 파싱 엔진 (Accurate Layer)

    정확한 텍스트 추출과 테이블 추출에 최적화된 엔진입니다.
    레이아웃을 보존하면서 텍스트를 추출합니다.

    Example:
        ```python
        from beanllm.domain.loaders.pdf.engines import PDFPlumberEngine

        engine = PDFPlumberEngine()
        result = engine.extract("document.pdf", {
            "extract_tables": True,
            "extract_images": False,
            "max_pages": None
        })
        ```
    """

    def __init__(self, name: Optional[str] = None):
        """
        Args:
            name: 엔진 이름 (기본값: "PDFPlumber")
        """
        super().__init__(name=name or "PDFPlumber")

    def _check_dependencies(self) -> None:
        """pdfplumber 라이브러리 확인"""
        try:
            import pdfplumber
        except ImportError:
            raise ImportError(
                "pdfplumber is required for PDFPlumberEngine. "
                "Install it with: pip install pdfplumber"
            )

    def extract(
        self,
        pdf_path: Union[str, Path],
        config: Dict,
    ) -> Dict:
        """
        PDF 파일에서 텍스트 및 메타데이터 추출 (pdfplumber 사용)

        Args:
            pdf_path: PDF 파일 경로
            config: 추출 설정 딕셔너리
                - extract_tables: bool - 테이블 추출 여부 (기본: True)
                - extract_images: bool - 이미지 추출 여부 (기본: False, pdfplumber는 이미지 추출 약함)
                - max_pages: Optional[int] - 최대 페이지 수
                - page_range: Optional[tuple] - (start, end) 페이지 범위

        Returns:
            Dict containing:
                - pages: List[Dict] - 페이지별 데이터
                - tables: List[Dict] - 추출된 테이블 (extract_tables=True일 때)
                - metadata: Dict - 전체 문서 메타데이터

        Raises:
            FileNotFoundError: PDF 파일이 없을 때
            Exception: PDF 파싱 실패 시
        """
        import pdfplumber

        start_time = time.time()
        pdf_path = self._validate_pdf_path(pdf_path)

        # 설정 추출
        extract_tables = config.get("extract_tables", True)
        max_pages = config.get("max_pages")
        page_range = config.get("page_range")

        try:
            pages_data = []
            tables_data = []

            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)

                # 페이지 범위 결정
                if page_range:
                    start_page, end_page = page_range
                    pages_to_process = range(start_page, min(end_page, total_pages))
                elif max_pages:
                    pages_to_process = range(min(max_pages, total_pages))
                else:
                    pages_to_process = range(total_pages)

                # 각 페이지 처리
                for page_num in pages_to_process:
                    if page_num >= total_pages:
                        break

                    page = pdf.pages[page_num]

                    # 텍스트 추출 옵션
                    layout_preserve = (
                        config.get("pdfplumber_layout", False) or 
                        config.get("layout_analysis", False)
                    )
                    x_tolerance = config.get("pdfplumber_x_tolerance", 3.0)
                    y_tolerance = config.get("pdfplumber_y_tolerance", 3.0)
                    
                    if layout_preserve:
                        # 레이아웃 보존 텍스트 추출
                        text = page.extract_text(
                            layout=True,
                            x_tolerance=x_tolerance,
                            y_tolerance=y_tolerance,
                        )
                    else:
                        # 기본 텍스트 추출 (공백 허용도 조정 가능)
                        text = page.extract_text(
                            x_tolerance=x_tolerance,
                            y_tolerance=y_tolerance,
                        )

                    # 고급: 문자/단어 단위 정보
                    extract_chars = (
                        config.get("pdfplumber_extract_chars", False) or 
                        config.get("layout_analysis", False)
                    )
                    extract_words = (
                        config.get("pdfplumber_extract_words", False) or 
                        config.get("layout_analysis", False)
                    )
                    
                    chars_info = None
                    words_info = None
                    if extract_chars or extract_words:
                        try:
                            # 문자 단위 정보 (위치, 폰트, 크기)
                            chars_info = [
                                {
                                    "text": char["text"],
                                    "x0": char["x0"],
                                    "y0": char["y0"],
                                    "x1": char["x1"],
                                    "y1": char["y1"],
                                    "size": char.get("size", 0),
                                }
                                for char in page.chars[:1000]  # 최대 1000개만 (성능 고려)
                            ]

                            # 단어 단위 정보
                            words_info = [
                                {
                                    "text": word["text"],
                                    "x0": word["x0"],
                                    "y0": word["y0"],
                                    "x1": word["x1"],
                                    "y1": word["y1"],
                                }
                                for word in page.words[:500]  # 최대 500개만
                            ]
                        except Exception as e:
                            logger.warning(f"Failed to extract chars/words: {e}")

                    # 페이지 메타데이터
                    page_rect = page.bbox
                    page_metadata = {
                        "page_number": page_num + 1,  # 1-based for user
                    }

                    # 고급: 하이퍼링크 추출
                    extract_hyperlinks = (
                        config.get("pdfplumber_extract_hyperlinks", False) or 
                        config.get("layout_analysis", False)
                    )
                    if extract_hyperlinks:
                        try:
                            hyperlinks = page.hyperlinks
                            if hyperlinks:
                                page_metadata["hyperlinks"] = [
                                    {
                                        "uri": link.get("uri", ""),
                                        "x0": link.get("x0", 0),
                                        "y0": link.get("y0", 0),
                                        "x1": link.get("x1", 0),
                                        "y1": link.get("y1", 0),
                                    }
                                    for link in hyperlinks
                                ]
                        except Exception as e:
                            logger.debug(f"Failed to extract hyperlinks: {e}")

                    page_data = {
                        "page": page_num,  # 0-based
                        "text": text or "",  # None일 수 있음
                        "width": page_rect[2] - page_rect[0] if page_rect else 0.0,
                        "height": page_rect[3] - page_rect[1] if page_rect else 0.0,
                        "metadata": page_metadata,
                    }

                    # 문자/단어 정보 추가 (있는 경우)
                    if chars_info:
                        page_data["chars"] = chars_info
                    if words_info:
                        page_data["words"] = words_info

                    pages_data.append(page_data)

                    # 테이블 추출 (요청된 경우)
                    if extract_tables:
                        page_tables = self._extract_tables_from_page(page, page_num)
                        tables_data.extend(page_tables)

            processing_time = time.time() - start_time

            result = {
                "pages": pages_data,
                "metadata": {
                    "total_pages": total_pages,
                    "engine": self.name,
                    "processing_time": processing_time,
                    "file_path": str(pdf_path),
                    "file_size": pdf_path.stat().st_size,
                },
            }

            # 테이블이 있으면 추가
            if tables_data:
                result["tables"] = tables_data

            logger.info(
                f"PDFPlumber extracted {len(pages_data)} pages, {len(tables_data)} tables "
                f"from {pdf_path} in {processing_time:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"PDFPlumber extraction failed for {pdf_path}: {e}")
            raise

    def _extract_tables_from_page(
        self,
        page: "pdfplumber.Page",  # type: ignore
        page_num: int,
    ) -> List[Dict]:
        """
        페이지에서 테이블 추출

        Args:
            page: pdfplumber Page 객체
            page_num: 페이지 번호 (0-based)

        Returns:
            테이블 정보 리스트
        """
        tables = []

        try:
            # 테이블 추출
            extracted_tables = page.extract_tables()

            for table_index, table in enumerate(extracted_tables):
                if not table or len(table) < 1:
                    continue

                # 테이블 bbox 찾기 (첫 번째 셀의 위치 사용)
                bbox = None
                try:
                    # pdfplumber는 테이블의 정확한 bbox를 직접 제공하지 않음
                    # 대략적인 위치 추정
                    if table and len(table) > 0 and len(table[0]) > 0:
                        # 첫 번째 셀의 위치로 추정
                        cells = page.find_tables()
                        if table_index < len(cells):
                            table_obj = cells[table_index]
                            bbox = (
                                table_obj.bbox[0],
                                table_obj.bbox[1],
                                table_obj.bbox[2],
                                table_obj.bbox[3],
                            )
                except Exception:
                    # bbox 추출 실패 시 None
                    bbox = None

                # 테이블 데이터 정리 (빈 행/열 제거)
                cleaned_table = [row for row in table if any(cell and str(cell).strip() for cell in row)]

                if not cleaned_table:
                    continue

                # pandas DataFrame 변환
                dataframe = None
                markdown = None
                csv_str = None

                try:
                    import pandas as pd

                    # 첫 번째 행을 헤더로 사용
                    if len(cleaned_table) > 1:
                        headers = [str(cell) if cell else f"Column_{i}" for i, cell in enumerate(cleaned_table[0])]
                        data_rows = cleaned_table[1:]
                        dataframe = pd.DataFrame(data_rows, columns=headers)

                        # Markdown 변환
                        markdown = dataframe.to_markdown(index=False)

                        # CSV 변환
                        csv_str = dataframe.to_csv(index=False)
                    else:
                        # 헤더만 있는 경우
                        headers = [str(cell) if cell else f"Column_{i}" for i, cell in enumerate(cleaned_table[0])]
                        dataframe = pd.DataFrame(columns=headers)
                        markdown = "| " + " | ".join(headers) + " |\n"
                        markdown += "| " + " | ".join(["---"] * len(headers)) + " |"
                        csv_str = ",".join(headers)

                except ImportError:
                    logger.warning("pandas not available, skipping DataFrame conversion")
                except Exception as e:
                    logger.warning(f"Failed to convert table to DataFrame: {e}")

                # 신뢰도 계산 (간단한 휴리스틱)
                confidence = self._calculate_table_confidence(cleaned_table)

                table_info = {
                    "page": page_num,
                    "table_index": table_index,
                    "data": cleaned_table,
                    "bbox": bbox,
                    "confidence": confidence,
                    "format": "list",
                    "metadata": {
                        "rows": len(cleaned_table),
                        "cols": max(len(row) for row in cleaned_table) if cleaned_table else 0,
                    },
                }

                # DataFrame 및 포맷 추가 (있는 경우)
                if dataframe is not None:
                    table_info["dataframe"] = dataframe
                    table_info["format"] = "dataframe"
                if markdown:
                    table_info["markdown"] = markdown
                if csv_str:
                    table_info["csv"] = csv_str

                tables.append(table_info)

        except Exception as e:
            logger.warning(f"Failed to extract tables from page {page_num}: {e}")

        return tables

    def _calculate_table_confidence(self, table: List[List]) -> float:
        """
        테이블 추출 신뢰도 계산

        Args:
            table: 추출된 테이블 데이터

        Returns:
            0.0 ~ 1.0 신뢰도 점수
        """
        if not table or len(table) < 2:
            return 0.3

        score = 1.0

        # 빈 셀 비율
        total_cells = sum(len(row) for row in table)
        if total_cells == 0:
            return 0.0

        empty_cells = sum(
            1
            for row in table
            for cell in row
            if not cell or (isinstance(cell, str) and cell.strip() == "")
        )
        empty_ratio = empty_cells / total_cells
        score -= empty_ratio * 0.3

        # 행 길이 일관성
        row_lengths = [len(row) for row in table]
        if len(set(row_lengths)) > 1:
            score -= 0.2

        # 최소 행/열 수
        if len(table) < 2:
            score -= 0.2
        if max(row_lengths) < 2:
            score -= 0.2

        return max(0.0, min(1.0, score))

