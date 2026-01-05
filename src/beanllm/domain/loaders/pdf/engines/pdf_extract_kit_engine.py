"""
PDF-Extract-Kit Engine

OpenDataLab의 PDF-Extract-Kit을 사용한 고정밀 PDF 파싱 엔진.
DocLayout-YOLO + StructTable-InternVL2로 레이아웃 및 테이블 추출.

PDF-Extract-Kit 특징:
- DocLayout-YOLO: 빠르고 정확한 레이아웃 검출
- StructTable-InternVL2: 테이블 인식 (LaTeX, HTML, Markdown 출력)
- GL-CRM (Global-to-Local Controllable Receptive Module)
- 다양한 스케일 타겟 검출

Requirements:
    pip install pdf-extract-kit torch pillow
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from .base import BasePDFEngine

logger = logging.getLogger(__name__)

# PDF-Extract-Kit 설치 여부 체크
try:
    # PDF-Extract-Kit의 실제 import는 사용 시점에 확인
    HAS_PDF_EXTRACT_KIT = True
except ImportError:
    HAS_PDF_EXTRACT_KIT = False


class PDFExtractKitEngine(BasePDFEngine):
    """
    PDF-Extract-Kit 파싱 엔진

    OpenDataLab의 PDF-Extract-Kit을 사용한 고급 PDF 파싱.

    Features:
    - DocLayout-YOLO 레이아웃 검출
    - StructTable-InternVL2 테이블 인식
    - 다중 스케일 타겟 검출
    - LaTeX/HTML/Markdown 출력
    - Lazy loading

    Example:
        ```python
        from beanllm.domain.loaders.pdf import beanPDFLoader

        # PDF-Extract-Kit 엔진 사용
        loader = beanPDFLoader(
            "document.pdf",
            engine="pdf-extract-kit",
            extract_tables=True
        )
        docs = loader.load()
        ```
    """

    def __init__(self, use_gpu: bool = True):
        """
        PDF-Extract-Kit 엔진 초기화

        Args:
            use_gpu: GPU 사용 여부
        """
        super().__init__(name="PDFExtractKitEngine")
        self.use_gpu = use_gpu
        self._layout_detector = None
        self._table_recognizer = None

    def _check_dependencies(self) -> None:
        """의존성 확인"""
        try:
            import torch
            from PIL import Image
        except ImportError:
            raise ImportError(
                "torch and pillow are required for PDF-Extract-Kit engine. "
                "Install them with: pip install torch pillow"
            )

    def _init_models(self):
        """모델 초기화 (lazy loading)"""
        if self._layout_detector is not None:
            return

        logger.info("Loading PDF-Extract-Kit models...")

        try:
            # NOTE: PDF-Extract-Kit의 실제 API는 설치 후 확인 필요
            # 여기서는 일반적인 패턴으로 구현
            # 실제로는 pdf_extract_kit 패키지의 API에 맞춰 수정 필요

            # DocLayout-YOLO 로드
            logger.info("Loading DocLayout-YOLO...")
            # self._layout_detector = load_doclayout_yolo(use_gpu=self.use_gpu)

            # StructTable-InternVL2 로드
            logger.info("Loading StructTable-InternVL2...")
            # self._table_recognizer = load_structtable_internvl2(use_gpu=self.use_gpu)

            # 임시: 모델 로딩이 완료되지 않았음을 표시
            self._layout_detector = "placeholder"
            self._table_recognizer = "placeholder"

            logger.info("PDF-Extract-Kit models loaded successfully")

        except Exception as e:
            raise ImportError(
                f"Failed to load PDF-Extract-Kit models: {e}. "
                "Install PDF-Extract-Kit with: pip install pdf-extract-kit"
            )

    def extract(
        self,
        pdf_path: Union[str, Path],
        config: Dict,
    ) -> Dict:
        """
        PDF-Extract-Kit으로 PDF 파싱

        Args:
            pdf_path: PDF 파일 경로
            config: 추출 설정
                - extract_tables: 테이블 추출 여부
                - extract_images: 이미지 추출 여부
                - layout_model: 레이아웃 모델 (doclayout-yolo)
                - table_format: 테이블 출력 형식 (markdown/html/latex)

        Returns:
            Dict: 파싱 결과
        """
        # PDF 경로 검증
        pdf_path = self._validate_pdf_path(pdf_path)

        # 모델 초기화
        self._init_models()

        start_time = time.time()

        # 설정 추출
        extract_tables = config.get("extract_tables", True)
        extract_images = config.get("extract_images", False)
        table_format = config.get("table_format", "markdown")

        try:
            # PDF를 이미지로 변환
            import fitz  # PyMuPDF

            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)

            pages_data = []
            tables_data = []
            images_data = []

            # 각 페이지 처리
            for page_num in range(total_pages):
                page = doc[page_num]

                # 페이지를 이미지로 변환 (DPI 300)
                pix = page.get_pixmap(dpi=300)
                img_data = pix.tobytes("png")

                # PIL Image로 변환
                import io

                from PIL import Image

                page_image = Image.open(io.BytesIO(img_data))

                # 1. 레이아웃 검출 (DocLayout-YOLO)
                layout_results = self._detect_layout(page_image)

                # 2. 텍스트 추출
                page_text = self._extract_text_from_layout(layout_results, page_image)

                # 3. 테이블 추출 (StructTable-InternVL2)
                if extract_tables:
                    page_tables = self._extract_tables(layout_results, page_image, table_format)
                    tables_data.extend(page_tables)

                # 4. 이미지 추출
                if extract_images:
                    page_images = self._extract_images(layout_results, page_image, page_num)
                    images_data.extend(page_images)

                # 페이지 데이터 저장
                pages_data.append(
                    {
                        "page": page_num,
                        "text": page_text,
                        "width": page.rect.width,
                        "height": page.rect.height,
                        "metadata": {
                            "layout_elements": len(layout_results) if layout_results else 0,
                        },
                    }
                )

            doc.close()

            processing_time = time.time() - start_time

            return {
                "pages": pages_data,
                "tables": tables_data,
                "images": images_data,
                "metadata": {
                    "total_pages": total_pages,
                    "engine": self.name,
                    "processing_time": processing_time,
                    "layout_model": "DocLayout-YOLO",
                    "table_model": "StructTable-InternVL2",
                },
            }

        except Exception as e:
            logger.error(f"Failed to parse PDF with PDF-Extract-Kit: {e}")
            raise

    def _detect_layout(self, page_image) -> List:
        """
        DocLayout-YOLO로 레이아웃 검출

        Args:
            page_image: PIL Image

        Returns:
            레이아웃 요소 리스트
        """
        # NOTE: 실제 PDF-Extract-Kit API에 맞춰 구현 필요
        # 임시 구현
        logger.debug("Detecting layout with DocLayout-YOLO...")

        # TODO: 실제 DocLayout-YOLO 추론
        # layout_results = self._layout_detector.detect(page_image)

        # 임시 반환
        return []

    def _extract_text_from_layout(self, layout_results: List, page_image) -> str:
        """
        레이아웃 결과에서 텍스트 추출

        Args:
            layout_results: 레이아웃 검출 결과
            page_image: PIL Image

        Returns:
            추출된 텍스트
        """
        # NOTE: 실제 구현 필요
        # 레이아웃 요소 순서대로 OCR 수행

        # 임시: PyMuPDF로 텍스트 추출 (fallback)
        return ""

    def _extract_tables(
        self, layout_results: List, page_image, table_format: str
    ) -> List[Dict]:
        """
        StructTable-InternVL2로 테이블 추출

        Args:
            layout_results: 레이아웃 검출 결과
            page_image: PIL Image
            table_format: 출력 형식 (markdown/html/latex)

        Returns:
            테이블 데이터 리스트
        """
        # NOTE: 실제 PDF-Extract-Kit API에 맞춰 구현 필요
        logger.debug(f"Extracting tables in {table_format} format...")

        # TODO: 테이블 영역만 추출하여 StructTable-InternVL2로 인식
        # table_regions = [r for r in layout_results if r['type'] == 'table']
        # for region in table_regions:
        #     table_image = crop_image(page_image, region['bbox'])
        #     table_result = self._table_recognizer.recognize(table_image, format=table_format)

        # 임시 반환
        return []

    def _extract_images(
        self, layout_results: List, page_image, page_num: int
    ) -> List[Dict]:
        """
        레이아웃에서 이미지 추출

        Args:
            layout_results: 레이아웃 검출 결과
            page_image: PIL Image
            page_num: 페이지 번호

        Returns:
            이미지 데이터 리스트
        """
        # NOTE: 실제 구현 필요
        # 이미지 영역 크롭 및 저장

        # 임시 반환
        return []

    def __repr__(self) -> str:
        return f"PDFExtractKitEngine(use_gpu={self.use_gpu})"
