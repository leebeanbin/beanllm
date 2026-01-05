"""
beanPDFLoader - 고급 PDF 로더

다층 아키텍처를 통한 최적화된 PDF 처리:
- Fast Layer: PyMuPDF (빠른 처리)
- Accurate Layer: pdfplumber (정확한 테이블 추출)
- ML Layer: marker-pdf (구조 보존 Markdown 변환)
- Advanced Layer (2024-2025):
  - PDF-Extract-Kit: DocLayout-YOLO + StructTable-InternVL2
  - Docling: DocLayNet + TableFormer (IBM)

기존 PDFLoader와 호환되면서 고급 기능을 제공합니다.
"""

from pathlib import Path
from typing import List, Optional, Union

from ..base import BaseDocumentLoader
from ..security import validate_file_path
from ..types import Document
from .models import PDFLoadConfig

try:
    from beanllm.utils.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


class beanPDFLoader(BaseDocumentLoader):
    """
    beanPDFLoader - 고급 PDF 로더

    다층 아키텍처를 통한 최적화된 PDF 처리:
    - Fast Layer: PyMuPDF (빠른 처리, 이미지 추출)
    - Accurate Layer: pdfplumber (정확한 테이블 추출)
    - ML Layer: marker-pdf (구조 보존 Markdown 변환)
    - Advanced Layer (2024-2025):
      - PDF-Extract-Kit: DocLayout-YOLO + StructTable-InternVL2
      - Docling: DocLayNet + TableFormer (IBM, 고정밀)

    Example:
        ```python
        from beanllm.domain.loaders.pdf import beanPDFLoader

        # 기본 사용 (자동 전략 선택)
        loader = beanPDFLoader("document.pdf")
        docs = loader.load()

        # 테이블 추출
        loader = beanPDFLoader("report.pdf", extract_tables=True)
        docs = loader.load()

        # 이미지 추출
        loader = beanPDFLoader("images.pdf", extract_images=True)
        docs = loader.load()

        # 명시적 전략 선택
        loader = beanPDFLoader("large.pdf", strategy="fast")
        docs = loader.load()

        # 최신 엔진 사용 (2024-2025)
        # PDF-Extract-Kit (레이아웃 + 테이블 고정밀)
        loader = beanPDFLoader("complex.pdf", strategy="pdf-extract-kit")
        docs = loader.load()

        # Docling (구조적 충실도 최우선)
        loader = beanPDFLoader("report.pdf", strategy="docling", extract_tables=True)
        docs = loader.load()
        ```
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        strategy: str = "auto",
        extract_tables: bool = True,
        extract_images: bool = False,
        to_markdown: bool = False,
        enable_ocr: bool = False,
        layout_analysis: bool = False,
        max_pages: Optional[int] = None,
        page_range: Optional[tuple[int, int]] = None,
        password: Optional[str] = None,
        validate_path: bool = True,
        # PyMuPDF 고급 옵션
        pymupdf_text_mode: str = "text",
        pymupdf_extract_fonts: bool = False,
        pymupdf_extract_links: bool = False,
        # pdfplumber 고급 옵션
        pdfplumber_layout: bool = False,
        pdfplumber_extract_chars: bool = False,
        pdfplumber_extract_words: bool = False,
        pdfplumber_extract_hyperlinks: bool = False,
        pdfplumber_x_tolerance: float = 3.0,
        pdfplumber_y_tolerance: float = 3.0,
    ):
        """
        Args:
            file_path: PDF 파일 경로
            strategy: 파싱 전략
                - "auto": 자동 선택 (기본값)
                - "fast": PyMuPDF (빠른 처리)
                - "accurate": pdfplumber (정확한 테이블 추출)
                - "ml": marker-pdf (ML 기반 Markdown 변환)
                - "pdf-extract-kit": PDF-Extract-Kit (DocLayout-YOLO + StructTable, 2024-2025)
                - "docling": Docling (DocLayNet + TableFormer, IBM, 2024-2025)
            extract_tables: 테이블 추출 여부
            extract_images: 이미지 추출 여부
            to_markdown: Markdown 변환 여부
            enable_ocr: OCR 활성화 여부 (향후 구현)
            layout_analysis: 레이아웃 분석 여부 (향후 구현)
            max_pages: 최대 처리 페이지 수 (None이면 전체)
            page_range: 처리할 페이지 범위 (start, end) (None이면 전체)
            password: PDF 비밀번호
            validate_path: 경로 검증 여부 (기본: True, Path Traversal 방지)
        """
        # 경로 검증 (Path Traversal 방지)
        if validate_path:
            self.file_path = validate_file_path(file_path)
        else:
            self.file_path = Path(file_path)

        self.password = password

        # Config 생성
        self.config = PDFLoadConfig(
            strategy=strategy,
            extract_tables=extract_tables,
            extract_images=extract_images,
            to_markdown=to_markdown,
            enable_ocr=enable_ocr,
            layout_analysis=layout_analysis,
            max_pages=max_pages,
            page_range=page_range,
            # PyMuPDF 고급 옵션
            pymupdf_text_mode=pymupdf_text_mode,
            pymupdf_extract_fonts=pymupdf_extract_fonts,
            pymupdf_extract_links=pymupdf_extract_links,
            # pdfplumber 고급 옵션
            pdfplumber_layout=pdfplumber_layout,
            pdfplumber_extract_chars=pdfplumber_extract_chars,
            pdfplumber_extract_words=pdfplumber_extract_words,
            pdfplumber_extract_hyperlinks=pdfplumber_extract_hyperlinks,
            pdfplumber_x_tolerance=pdfplumber_x_tolerance,
            pdfplumber_y_tolerance=pdfplumber_y_tolerance,
        )

        # 엔진 초기화
        self._engines = {}
        self._init_engines()

        # 의존성 확인
        self._check_dependencies()

    def _init_engines(self) -> None:
        """사용 가능한 엔진 초기화"""
        # PyMuPDF Engine (Fast Layer)
        try:
            from .engines.pymupdf_engine import PyMuPDFEngine

            self._engines["fast"] = PyMuPDFEngine()
            logger.debug("PyMuPDF engine initialized")
        except ImportError as e:
            logger.warning(f"PyMuPDF engine not available: {e}")

        # PDFPlumber Engine (Accurate Layer)
        try:
            from .engines.pdfplumber_engine import PDFPlumberEngine

            self._engines["accurate"] = PDFPlumberEngine()
            logger.debug("PDFPlumber engine initialized")
        except ImportError as e:
            logger.warning(f"PDFPlumber engine not available: {e}")

        # ML Layer (marker-pdf, optional)
        try:
            from .engines.marker_engine import MarkerEngine

            self._engines["ml"] = MarkerEngine(use_gpu=False)
            logger.debug("Marker engine initialized (ML Layer)")
        except ImportError as e:
            logger.debug(f"Marker engine not available: {e}")

        # PDF-Extract-Kit Engine (2024-2025, optional)
        try:
            from .engines.pdf_extract_kit_engine import PDFExtractKitEngine

            self._engines["pdf-extract-kit"] = PDFExtractKitEngine(use_gpu=False)
            logger.debug("PDF-Extract-Kit engine initialized")
        except ImportError as e:
            logger.debug(f"PDF-Extract-Kit engine not available: {e}")

        # Docling Engine (IBM, 2024-2025, optional)
        try:
            from .engines.docling_engine import DoclingEngine

            self._engines["docling"] = DoclingEngine(use_gpu=False)
            logger.debug("Docling engine initialized")
        except ImportError as e:
            logger.debug(f"Docling engine not available: {e}")

        if not self._engines:
            raise ImportError(
                "No PDF engines available. "
                "Install at least one: pip install PyMuPDF pdfplumber"
            )

    def _check_dependencies(self) -> None:
        """필수 의존성 확인"""
        # 엔진이 하나라도 있으면 OK
        if not self._engines:
            raise ImportError(
                "No PDF engines available. "
                "Install at least one: pip install PyMuPDF pdfplumber"
            )

    def load(self) -> List[Document]:
        """
        PDF 로딩 (페이지별 문서)

        Returns:
            Document 리스트 (각 페이지가 하나의 Document)

        Raises:
            FileNotFoundError: PDF 파일이 없을 때
            ImportError: 필수 라이브러리가 없을 때
            Exception: PDF 파싱 실패 시
        """
        # 파일 검증
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.file_path}")

        # 전략 선택
        strategy = self._select_strategy()

        # 엔진 실행
        result = self._execute_strategy(strategy)

        # 결과 저장 (외부 접근용)
        self._result = result

        # Markdown 변환 (to_markdown=True일 때)
        if self.config.to_markdown:
            markdown_text = self._convert_to_markdown(result)
            result["markdown"] = markdown_text

        # Document 리스트로 변환
        documents = self._convert_to_documents(result)

        logger.info(
            f"beanPDFLoader loaded {len(documents)} pages from {self.file_path} "
            f"(strategy: {strategy})"
        )

        return documents

    def lazy_load(self):
        """
        지연 로딩 (제너레이터)

        Yields:
            Document 객체
        """
        yield from self.load()

    def load_streaming(self):
        """
        스트리밍 로딩 (메모리 효율적 페이지별 처리)

        대용량 PDF를 메모리에 한 번에 올리지 않고 페이지별로 스트리밍 처리합니다.
        각 페이지가 처리되는 즉시 yield되므로 메모리 사용량이 일정하게 유지됩니다.

        Yields:
            Document 객체 (페이지별)

        Example:
            ```python
            loader = beanPDFLoader("large.pdf")
            for doc in loader.load_streaming():
                print(f"Page {doc.metadata['page']}: {len(doc.content)} chars")
                process_document(doc)  # 즉시 처리 가능
            ```
        """
        # 파일 검증
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.file_path}")

        # 전략 선택
        strategy = self._select_strategy()

        if strategy not in self._engines:
            raise ValueError(f"Strategy '{strategy}' not available")

        engine = self._engines[strategy]

        # Config를 딕셔너리로 변환
        config_dict = self.config.to_dict()

        # 엔진이 스트리밍을 지원하는지 확인
        if hasattr(engine, 'extract_streaming'):
            # 스트리밍 지원 엔진
            for page_result in engine.extract_streaming(self.file_path, config_dict):
                # 페이지별 Document 생성
                metadata = {
                    "source": str(self.file_path),
                    "page": page_result["page"],
                    "engine": strategy,
                    "strategy": strategy,
                    "width": page_result.get("width", 0.0),
                    "height": page_result.get("height", 0.0),
                }

                # 페이지 메타데이터 추가
                if "metadata" in page_result:
                    metadata.update(page_result["metadata"])

                # 테이블 정보 추가
                if "tables" in page_result:
                    metadata["tables"] = [
                        {
                            "table_index": table.get("table_index"),
                            "rows": table.get("metadata", {}).get("rows", 0),
                            "cols": table.get("metadata", {}).get("cols", 0),
                            "confidence": table.get("confidence", 0.0),
                            "has_dataframe": "dataframe" in table,
                            "has_markdown": "markdown" in table,
                            "has_csv": "csv" in table,
                        }
                        for table in page_result["tables"]
                    ]

                # 이미지 정보 추가
                if "images" in page_result:
                    metadata["images"] = [
                        {
                            "image_index": img.get("image_index"),
                            "format": img.get("format"),
                            "width": img.get("width"),
                            "height": img.get("height"),
                            "size": img.get("size"),
                        }
                        for img in page_result["images"]
                    ]

                document = Document(
                    content=page_result.get("text", ""),
                    metadata=metadata,
                )

                yield document

        else:
            # 스트리밍 미지원 엔진 - lazy_load로 fallback
            logger.warning(
                f"Engine '{strategy}' does not support streaming, "
                f"falling back to lazy_load"
            )
            yield from self.lazy_load()

    def _select_strategy(self) -> str:
        """
        PDF 특성 기반 자동 전략 선택

        Returns:
            "fast" 또는 "accurate"
        """
        # 명시적 전략이 있으면 사용
        if self.config.strategy != "auto":
            if self.config.strategy in self._engines:
                return self.config.strategy
            else:
                logger.warning(
                    f"Strategy '{self.config.strategy}' not available, "
                    f"falling back to auto"
                )

        # 자동 선택 로직
        # 테이블 추출이 필요하면 accurate
        if self.config.extract_tables:
            if "accurate" in self._engines:
                return "accurate"
            else:
                logger.warning("Table extraction requested but accurate engine not available")

        # 이미지 추출이 필요하면 fast (PyMuPDF가 이미지 추출에 강함)
        if self.config.extract_images:
            if "fast" in self._engines:
                return "fast"

        # 페이지 수 확인 (간단한 휴리스틱)
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(self.file_path)
            page_count = len(doc)
            doc.close()

            # 대용량 문서는 fast
            if page_count > 100:
                if "fast" in self._engines:
                    return "fast"
        except Exception:
            pass

        # 기본값: accurate (정확도 우선)
        if "accurate" in self._engines:
            return "accurate"
        elif "fast" in self._engines:
            return "fast"
        else:
            # 사용 가능한 첫 번째 엔진
            return list(self._engines.keys())[0]

    def _execute_strategy(self, strategy: str) -> dict:
        """
        선택된 전략으로 엔진 실행

        Args:
            strategy: "fast" 또는 "accurate"

        Returns:
            엔진 추출 결과 딕셔너리
        """
        if strategy not in self._engines:
            raise ValueError(f"Strategy '{strategy}' not available")

        engine = self._engines[strategy]

        # Config를 딕셔너리로 변환
        config_dict = self.config.to_dict()

        # 엔진 실행
        result = engine.extract(self.file_path, config_dict)

        return result

    def _convert_to_documents(self, result: dict) -> List[Document]:
        """
        엔진 결과를 Document 리스트로 변환

        Args:
            result: 엔진 추출 결과

        Returns:
            Document 리스트
        """
        documents = []

        # 페이지별로 Document 생성
        for page_data in result.get("pages", []):
            # 기본 메타데이터
            metadata = {
                "source": str(self.file_path),
                "page": page_data["page"],
                "total_pages": result["metadata"].get("total_pages", 0),
                "engine": result["metadata"].get("engine", "unknown"),
                "strategy": result["metadata"].get("engine", "unknown"),
                "width": page_data.get("width", 0.0),
                "height": page_data.get("height", 0.0),
            }

            # 페이지 메타데이터 추가
            if "metadata" in page_data:
                metadata.update(page_data["metadata"])

            # 테이블 정보 추가 (해당 페이지의 테이블)
            page_num = page_data["page"]
            page_tables = [
                table
                for table in result.get("tables", [])
                if table.get("page") == page_num
            ]
            if page_tables:
                metadata["tables"] = [
                    {
                        "table_index": table.get("table_index"),
                        "rows": table.get("metadata", {}).get("rows", 0),
                        "cols": table.get("metadata", {}).get("cols", 0),
                        "confidence": table.get("confidence", 0.0),
                        "has_dataframe": "dataframe" in table,
                        "has_markdown": "markdown" in table,
                        "has_csv": "csv" in table,
                    }
                    for table in page_tables
                ]

            # 이미지 정보 추가 (해당 페이지의 이미지)
            page_images = [
                img
                for img in result.get("images", [])
                if img.get("page") == page_num
            ]
            if page_images:
                metadata["images"] = [
                    {
                        "image_index": img.get("image_index"),
                        "format": img.get("format"),
                        "width": img.get("width"),
                        "height": img.get("height"),
                        "size": img.get("size"),
                    }
                    for img in page_images
                ]

            # Document 생성
            document = Document(
                content=page_data.get("text", ""),
                metadata=metadata,
            )

            documents.append(document)

        return documents

    def _convert_to_markdown(self, result: dict) -> str:
        """
        추출 결과를 Markdown으로 변환

        Args:
            result: 엔진 추출 결과

        Returns:
            str: Markdown 형식 텍스트
        """
        from .utils import MarkdownConverter

        converter = MarkdownConverter()
        markdown_text = converter.convert_to_markdown(result)

        return markdown_text

