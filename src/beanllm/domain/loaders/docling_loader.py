"""
Docling Loader

Docling 고급 문서 로더
"""

import logging
import mmap
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from .base import BaseDocumentLoader
from .security import validate_file_path
from .types import Document

try:
    from beanllm.utils.logger import get_logger
except ImportError:
    def get_logger(name: str):
        return logging.getLogger(name)

logger = get_logger(__name__)

class DoclingLoader(BaseDocumentLoader):
    """
    Docling 로더 (IBM, 2024-2025)

    IBM의 최신 문서 파싱 라이브러리로 Office 파일을 고품질로 파싱합니다.

    지원 포맷:
    - PDF: 고급 레이아웃 분석, 표 추출
    - DOCX: Word 문서
    - XLSX: Excel 스프레드시트
    - PPTX: PowerPoint 프레젠테이션
    - HTML: 웹 페이지
    - Images: PNG, JPG (OCR)
    - Markdown: .md 파일

    Features:
    - 고급 레이아웃 분석 (테이블, 그림, 캡션)
    - OCR 통합 (EasyOCR, Tesseract)
    - 구조 보존 (헤더, 리스트, 표)
    - Markdown/HTML 출력
    - GPU 가속 지원

    Docling vs PyPDF/python-docx:
    - Docling: 고급 레이아웃 분석, 표 추출, OCR, 멀티포맷
    - PyPDF: 단순 텍스트 추출
    - python-docx: DOCX 전용

    Example:
        ```python
        from beanllm.domain.loaders import DoclingLoader

        # PDF with 표 추출
        loader = DoclingLoader(
            file_path="document.pdf",
            extract_tables=True,
            extract_images=True
        )
        docs = loader.load()

        # DOCX
        loader = DoclingLoader(file_path="document.docx")
        docs = loader.load()

        # XLSX
        loader = DoclingLoader(
            file_path="spreadsheet.xlsx",
            include_sheet_names=True
        )
        docs = loader.load()

        # PPTX
        loader = DoclingLoader(file_path="presentation.pptx")
        docs = loader.load()
        ```

    Requirements:
        pip install docling

    References:
        - https://github.com/DS4SD/docling
        - https://ds4sd.github.io/docling/
    """

    def __init__(
        self,
        file_path: str,
        extract_tables: bool = True,
        extract_images: bool = False,
        ocr_enabled: bool = False,
        output_format: str = "markdown",
        include_metadata: bool = True,
        **kwargs,
    ):
        """
        Args:
            file_path: 파일 경로 (.pdf, .docx, .xlsx, .pptx, .html, .md, 이미지)
            extract_tables: 표 추출 여부 (기본: True)
            extract_images: 이미지 추출 여부 (기본: False)
            ocr_enabled: OCR 활성화 (이미지/스캔 PDF용) (기본: False)
            output_format: 출력 포맷 ("markdown", "text") (기본: "markdown")
            include_metadata: 메타데이터 포함 여부 (기본: True)
            **kwargs: 추가 파라미터
        """
        self.file_path = file_path
        self.extract_tables = extract_tables
        self.extract_images = extract_images
        self.ocr_enabled = ocr_enabled
        self.output_format = output_format.lower()
        self.include_metadata = include_metadata
        self.kwargs = kwargs

        # 출력 포맷 검증
        valid_formats = ["markdown", "text"]
        if self.output_format not in valid_formats:
            raise ValueError(
                f"Invalid output_format: {self.output_format}. "
                f"Available: {valid_formats}"
            )

    def load(self) -> List[Document]:
        """Docling으로 문서 로딩"""
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.document_converter import DocumentConverter
        except ImportError:
            raise ImportError(
                "docling is required for DoclingLoader. "
                "Install it with: pip install docling"
            )

        # 파일 존재 확인
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        logger.info(f"Loading document with Docling: {self.file_path}")

        try:
            # DocumentConverter 생성
            converter = DocumentConverter()

            # 문서 변환
            result = converter.convert(self.file_path)

            # 문서 내용 추출
            if self.output_format == "markdown":
                content = result.document.export_to_markdown()
            else:  # text
                content = result.document.export_to_text()

            # 메타데이터 생성
            metadata = self._extract_metadata(result)

            # Document 생성
            doc = Document(
                content=content,
                metadata=metadata if self.include_metadata else {},
                source=self.file_path,
            )

            logger.info(
                f"Docling loaded: {self.file_path}, "
                f"length={len(content)}, "
                f"format={self.output_format}"
            )

            return [doc]

        except Exception as e:
            logger.error(f"Docling loading failed: {self.file_path}, error: {e}")
            raise

    def _extract_metadata(self, result) -> Dict[str, Any]:
        """
        메타데이터 추출

        Args:
            result: Docling 변환 결과

        Returns:
            메타데이터 딕셔너리
        """
        metadata = {
            "source": self.file_path,
            "file_name": os.path.basename(self.file_path),
            "file_type": os.path.splitext(self.file_path)[1].lower(),
            "loader": "DoclingLoader",
            "output_format": self.output_format,
        }

        # Docling 메타데이터 추가
        try:
            doc = result.document

            # 문서 제목
            if hasattr(doc, "title") and doc.title:
                metadata["title"] = doc.title

            # 작성자
            if hasattr(doc, "author") and doc.author:
                metadata["author"] = doc.author

            # 페이지 수 (PDF용)
            if hasattr(doc, "num_pages"):
                metadata["num_pages"] = doc.num_pages

            # 생성일
            if hasattr(doc, "creation_date") and doc.creation_date:
                metadata["creation_date"] = str(doc.creation_date)

            # 수정일
            if hasattr(doc, "modification_date") and doc.modification_date:
                metadata["modification_date"] = str(doc.modification_date)

            # 표 개수
            if self.extract_tables and hasattr(doc, "tables"):
                metadata["num_tables"] = len(doc.tables) if doc.tables else 0

            # 이미지 개수
            if self.extract_images and hasattr(doc, "pictures"):
                metadata["num_images"] = len(doc.pictures) if doc.pictures else 0

        except Exception as e:
            logger.warning(f"Failed to extract some metadata: {e}")

        return metadata

    def load_and_split(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> List[Document]:
        """
        문서 로딩 및 청킹

        Args:
            chunk_size: 청크 크기 (기본: 1000)
            chunk_overlap: 청크 오버랩 (기본: 200)

        Returns:
            청크된 Document 리스트
        """
        # 문서 로드
        docs = self.load()

        # 청킹
        try:
            from ..splitters import RecursiveCharacterTextSplitter
        except ImportError:
            logger.warning(
                "RecursiveCharacterTextSplitter not available, "
                "returning unsplit documents"
            )
            return docs

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        split_docs = []
        for doc in docs:
            chunks = splitter.split_text(doc.content)
            for i, chunk in enumerate(chunks):
                metadata = doc.metadata.copy()
                metadata["chunk_index"] = i
                metadata["total_chunks"] = len(chunks)

                split_docs.append(
                    Document(
                        content=chunk,
                        metadata=metadata,
                        source=doc.source,
                    )
                )

        logger.info(f"Split into {len(split_docs)} chunks")

        return split_docs
