"""
PDF 데이터 모델

PDF 로딩 및 추출 결과를 표현하는 데이터 클래스들

참고: 내부 엔진에서 사용하는 모델이며, 최종적으로는 Document 타입으로 변환됩니다.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union


@dataclass
class PageData:
    """
    단일 페이지 데이터

    Attributes:
        page: 페이지 번호 (0-based)
        text: 추출된 텍스트
        width: 페이지 너비 (포인트)
        height: 페이지 높이 (포인트)
        metadata: 추가 메타데이터 (폰트, 레이아웃 등)
    """

    page: int
    text: str
    width: float
    height: float
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "page": self.page,
            "text": self.text,
            "width": self.width,
            "height": self.height,
            "metadata": self.metadata,
        }


@dataclass
class TableData:
    """
    추출된 테이블 데이터

    Attributes:
        page: 테이블이 있는 페이지 번호
        table_index: 페이지 내 테이블 인덱스
        data: 테이블 데이터 (2D 리스트 또는 pandas DataFrame)
        bbox: 테이블 위치 (x0, y0, x1, y1)
        confidence: 추출 신뢰도 (0.0 ~ 1.0)
        format: 데이터 포맷 ("dataframe", "list", "markdown", "csv")
    """

    page: int
    table_index: int
    data: Union[List[List], "pandas.DataFrame"]  # type: ignore
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    confidence: float = 1.0
    format: str = "dataframe"
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        result = {
            "page": self.page,
            "table_index": self.table_index,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "format": self.format,
            "metadata": self.metadata,
        }

        # DataFrame은 to_dict()로 변환
        if hasattr(self.data, "to_dict"):
            result["data"] = self.data.to_dict("records")
        else:
            result["data"] = self.data

        return result


@dataclass
class ImageData:
    """
    추출된 이미지 데이터

    Attributes:
        page: 이미지가 있는 페이지 번호
        image_index: 페이지 내 이미지 인덱스
        image: 이미지 데이터 (bytes 또는 PIL Image)
        format: 이미지 포맷 ("png", "jpeg", etc.)
        width: 이미지 너비 (픽셀)
        height: 이미지 높이 (픽셀)
        bbox: 이미지 위치 (x0, y0, x1, y1)
        size: 파일 크기 (bytes)
    """

    page: int
    image_index: int
    image: Union[bytes, "PIL.Image.Image"]  # type: ignore
    format: str
    width: int
    height: int
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    size: int
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """딕셔너리로 변환 (이미지 데이터는 제외)"""
        return {
            "page": self.page,
            "image_index": self.image_index,
            "format": self.format,
            "width": self.width,
            "height": self.height,
            "bbox": self.bbox,
            "size": self.size,
            "metadata": self.metadata,
        }


@dataclass
class PDFLoadConfig:
    """
    PDF 로딩 설정

    Attributes:
        strategy: 파싱 전략
            - "auto": 자동 선택 (기본값)
            - "fast": PyMuPDF (빠른 처리)
            - "accurate": pdfplumber (정확한 테이블 추출)
            - "ml": marker-pdf (구조 보존 Markdown)
        extract_tables: 테이블 추출 여부
        extract_images: 이미지 추출 여부
        to_markdown: Markdown 변환 여부
        enable_ocr: OCR 활성화 여부
        layout_analysis: 레이아웃 분석 여부
        max_pages: 최대 처리 페이지 수 (None이면 전체)
        page_range: 처리할 페이지 범위 (start, end) (None이면 전체)
        
        # PyMuPDF 고급 옵션
        pymupdf_text_mode: str = "text"  # "text", "dict", "rawdict", "html", "xml", "json"
        pymupdf_extract_fonts: bool = False  # 폰트 정보 추출
        pymupdf_extract_links: bool = False  # 링크 추출
        
        # pdfplumber 고급 옵션
        pdfplumber_layout: bool = False  # 레이아웃 보존 텍스트
        pdfplumber_extract_chars: bool = False  # 문자 단위 정보
        pdfplumber_extract_words: bool = False  # 단어 단위 정보
        pdfplumber_extract_hyperlinks: bool = False  # 하이퍼링크 추출
        pdfplumber_x_tolerance: float = 3.0  # 수평 공백 허용도
        pdfplumber_y_tolerance: float = 3.0  # 수직 공백 허용도
    """

    strategy: str = "auto"
    extract_tables: bool = True
    extract_images: bool = False
    to_markdown: bool = False
    enable_ocr: bool = False
    layout_analysis: bool = False
    max_pages: Optional[int] = None
    page_range: Optional[tuple[int, int]] = None
    
    # PyMuPDF 고급 옵션
    pymupdf_text_mode: str = "text"
    pymupdf_extract_fonts: bool = False
    pymupdf_extract_links: bool = False
    
    # pdfplumber 고급 옵션
    pdfplumber_layout: bool = False
    pdfplumber_extract_chars: bool = False
    pdfplumber_extract_words: bool = False
    pdfplumber_extract_hyperlinks: bool = False
    pdfplumber_x_tolerance: float = 3.0
    pdfplumber_y_tolerance: float = 3.0

    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "strategy": self.strategy,
            "extract_tables": self.extract_tables,
            "extract_images": self.extract_images,
            "to_markdown": self.to_markdown,
            "enable_ocr": self.enable_ocr,
            "layout_analysis": self.layout_analysis,
            "max_pages": self.max_pages,
            "page_range": self.page_range,
            # PyMuPDF 고급 옵션
            "pymupdf_text_mode": self.pymupdf_text_mode,
            "pymupdf_extract_fonts": self.pymupdf_extract_fonts,
            "pymupdf_extract_links": self.pymupdf_extract_links,
            # pdfplumber 고급 옵션
            "pdfplumber_layout": self.pdfplumber_layout,
            "pdfplumber_extract_chars": self.pdfplumber_extract_chars,
            "pdfplumber_extract_words": self.pdfplumber_extract_words,
            "pdfplumber_extract_hyperlinks": self.pdfplumber_extract_hyperlinks,
            "pdfplumber_x_tolerance": self.pdfplumber_x_tolerance,
            "pdfplumber_y_tolerance": self.pdfplumber_y_tolerance,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PDFLoadConfig":
        """딕셔너리에서 생성"""
        return cls(**data)


@dataclass
class PDFLoadResult:
    """
    PDF 로딩 결과

    Attributes:
        pages: 추출된 페이지 데이터 리스트
        tables: 추출된 테이블 리스트 (extract_tables=True일 때)
        images: 추출된 이미지 리스트 (extract_images=True일 때)
        markdown: Markdown 변환 결과 (to_markdown=True일 때)
        metadata: 전체 문서 메타데이터
            - total_pages: 전체 페이지 수
            - engine: 사용된 엔진 이름
            - strategy: 사용된 전략
            - processing_time: 처리 시간 (초)
            - quality_score: 품질 점수 (0.0 ~ 1.0)
    """

    pages: List[PageData]
    tables: List[TableData] = field(default_factory=list)
    images: List[ImageData] = field(default_factory=list)
    markdown: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            "pages": [page.to_dict() for page in self.pages],
            "tables": [table.to_dict() for table in self.tables],
            "images": [image.to_dict() for image in self.images],
            "markdown": self.markdown,
            "metadata": self.metadata,
        }

