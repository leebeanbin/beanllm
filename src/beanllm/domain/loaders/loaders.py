"""
Loaders Implementations - 문서 로더 구현체들
"""

import csv
from pathlib import Path
from typing import List, Optional, Union

from .base import BaseDocumentLoader
from .types import Document

try:
    from ...utils.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


class TextLoader(BaseDocumentLoader):
    """
    텍스트 파일 로더

    Example:
        ```python
        from beanllm.domain.loaders import TextLoader

        loader = TextLoader("file.txt", encoding="utf-8")
        docs = loader.load()
        ```
    """

    def __init__(
        self, file_path: Union[str, Path], encoding: str = "utf-8", autodetect_encoding: bool = True
    ):
        """
        Args:
            file_path: 파일 경로
            encoding: 인코딩
            autodetect_encoding: 인코딩 자동 감지
        """
        self.file_path = Path(file_path)
        self.encoding = encoding
        self.autodetect_encoding = autodetect_encoding

    def load(self) -> List[Document]:
        """파일 로딩"""
        try:
            content = self._read_file()
            return [
                Document(
                    content=content,
                    metadata={"source": str(self.file_path), "encoding": self.encoding},
                )
            ]
        except Exception as e:
            logger.error(f"Failed to load {self.file_path}: {e}")
            raise

    def lazy_load(self):
        """지연 로딩"""
        yield from self.load()

    def _read_file(self) -> str:
        """파일 읽기"""
        # 인코딩 자동 감지
        if self.autodetect_encoding:
            try:
                with open(self.file_path, "r", encoding=self.encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                # UTF-8 실패 시 다른 인코딩 시도
                for encoding in ["cp949", "euc-kr", "latin-1"]:
                    try:
                        with open(self.file_path, "r", encoding=encoding) as f:
                            content = f.read()
                            self.encoding = encoding
                            logger.info(f"Auto-detected encoding: {encoding}")
                            return content
                    except UnicodeDecodeError:
                        continue
                raise
        else:
            with open(self.file_path, "r", encoding=self.encoding) as f:
                return f.read()


class PDFLoader(BaseDocumentLoader):
    """
    PDF 로더

    Example:
        ```python
        from beanllm.domain.loaders import PDFLoader

        loader = PDFLoader("document.pdf")
        docs = loader.load()  # 페이지별로 분리

        # 특정 페이지만
        loader = PDFLoader("document.pdf", pages=[1, 2, 3])
        ```
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        pages: Optional[List[int]] = None,
        password: Optional[str] = None,
    ):
        """
        Args:
            file_path: PDF 경로
            pages: 로딩할 페이지 번호 (None이면 전체)
            password: PDF 비밀번호
        """
        self.file_path = Path(file_path)
        self.pages = pages
        self.password = password

        # pypdf 확인
        try:
            import pypdf

            self.pypdf = pypdf
        except ImportError:
            raise ImportError("pypdf is required for PDFLoader. Install it with: pip install pypdf")

    def load(self) -> List[Document]:
        """PDF 로딩 (페이지별 문서)"""
        documents = []

        try:
            with open(self.file_path, "rb") as f:
                pdf_reader = self.pypdf.PdfReader(f, password=self.password)

                # 페이지 선택
                pages_to_load = self.pages or range(len(pdf_reader.pages))

                for page_num in pages_to_load:
                    if page_num >= len(pdf_reader.pages):
                        logger.warning(f"Page {page_num} out of range")
                        continue

                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()

                    documents.append(
                        Document(
                            content=text,
                            metadata={
                                "source": str(self.file_path),
                                "page": page_num,
                                "total_pages": len(pdf_reader.pages),
                            },
                        )
                    )

            logger.info(f"Loaded {len(documents)} pages from {self.file_path}")
            return documents

        except Exception as e:
            logger.error(f"Failed to load PDF {self.file_path}: {e}")
            raise

    def lazy_load(self):
        """지연 로딩"""
        yield from self.load()


class CSVLoader(BaseDocumentLoader):
    """
    CSV 로더

    Example:
        ```python
        from beanllm.domain.loaders import CSVLoader

        # 행별로 문서 생성
        loader = CSVLoader("data.csv")
        docs = loader.load()

        # 특정 컬럼만 content로
        loader = CSVLoader("data.csv", content_columns=["text", "description"])
        ```
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
        encoding: str = "utf-8",
    ):
        """
        Args:
            file_path: CSV 경로
            content_columns: content로 사용할 컬럼들 (None이면 전체)
            metadata_columns: metadata로 저장할 컬럼들
            encoding: 인코딩
        """
        self.file_path = Path(file_path)
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns
        self.encoding = encoding

    def load(self) -> List[Document]:
        """CSV 로딩 (행별 문서)"""
        documents = []

        try:
            with open(self.file_path, "r", encoding=self.encoding) as f:
                reader = csv.DictReader(f)

                for i, row in enumerate(reader):
                    # Content 생성
                    if self.content_columns:
                        content_parts = [
                            f"{col}: {row.get(col, '')}"
                            for col in self.content_columns
                            if col in row
                        ]
                        content = "\n".join(content_parts)
                    else:
                        # 모든 컬럼 사용
                        content = "\n".join([f"{k}: {v}" for k, v in row.items()])

                    # Metadata
                    metadata = {"source": str(self.file_path), "row": i}

                    if self.metadata_columns:
                        for col in self.metadata_columns:
                            if col in row:
                                metadata[col] = row[col]

                    documents.append(Document(content=content, metadata=metadata))

            logger.info(f"Loaded {len(documents)} rows from {self.file_path}")
            return documents

        except Exception as e:
            logger.error(f"Failed to load CSV {self.file_path}: {e}")
            raise

    def lazy_load(self):
        """지연 로딩"""
        with open(self.file_path, "r", encoding=self.encoding) as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader):
                # Content
                if self.content_columns:
                    content_parts = [
                        f"{col}: {row.get(col, '')}" for col in self.content_columns if col in row
                    ]
                    content = "\n".join(content_parts)
                else:
                    content = "\n".join([f"{k}: {v}" for k, v in row.items()])

                # Metadata
                metadata = {"source": str(self.file_path), "row": i}

                if self.metadata_columns:
                    for col in self.metadata_columns:
                        if col in row:
                            metadata[col] = row[col]

                yield Document(content=content, metadata=metadata)


class DirectoryLoader(BaseDocumentLoader):
    """
    디렉토리 로더 (재귀)

    Example:
        ```python
        from beanllm.domain.loaders import DirectoryLoader

        # 모든 .txt 파일
        loader = DirectoryLoader("./docs", glob="**/*.txt")
        docs = loader.load()

        # 모든 파일 (자동 감지)
        loader = DirectoryLoader("./docs")
        ```
    """

    def __init__(
        self,
        path: Union[str, Path],
        glob: str = "**/*",
        exclude: Optional[List[str]] = None,
        recursive: bool = True,
    ):
        """
        Args:
            path: 디렉토리 경로
            glob: 파일 패턴
            exclude: 제외할 패턴
            recursive: 재귀 검색
        """
        self.path = Path(path)
        self.glob = glob
        self.exclude = exclude or []
        self.recursive = recursive

    def load(self) -> List[Document]:
        """디렉토리 로딩"""
        from .factory import DocumentLoader

        documents = []

        # 파일 검색
        if self.recursive:
            files = self.path.glob(self.glob)
        else:
            files = self.path.glob(self.glob.replace("**/", ""))

        for file_path in files:
            # 제외 패턴 확인
            if any(file_path.match(pattern) for pattern in self.exclude):
                continue

            # 파일만
            if not file_path.is_file():
                continue

            # 자동 감지해서 로딩
            loader = DocumentLoader.get_loader(file_path)
            if loader:
                try:
                    file_docs = loader.load()
                    documents.extend(file_docs)
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")

        logger.info(f"Loaded {len(documents)} documents from {self.path}")
        return documents

    def lazy_load(self):
        """지연 로딩"""
        from .factory import DocumentLoader

        if self.recursive:
            files = self.path.glob(self.glob)
        else:
            files = self.path.glob(self.glob.replace("**/", ""))

        for file_path in files:
            if any(file_path.match(pattern) for pattern in self.exclude):
                continue

            if not file_path.is_file():
                continue

            loader = DocumentLoader.get_loader(file_path)
            if loader:
                try:
                    yield from loader.lazy_load()
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")


class HTMLLoader(BaseDocumentLoader):
    """
    HTML 로더 (Multi-tier fallback, 2024-2025)

    웹 콘텐츠와 HTML 파일을 로드합니다. 3단계 fallback 전략으로 최고의 품질을 보장합니다:
    1. Trafilatura (추천) - 뉴스/블로그 기사 최적화, 메타데이터 추출
    2. Readability (fallback 1) - Mozilla의 Reader View 알고리즘
    3. BeautifulSoup (fallback 2) - 원시 HTML 파싱

    Features:
    - Multi-tier fallback chain (품질 보장)
    - URL 및 로컬 파일 지원
    - 메타데이터 추출 (title, author, date)
    - JavaScript 렌더링 지원 (선택적)

    Example:
        ```python
        from beanllm.domain.loaders import HTMLLoader

        # URL 로드 (기본: Trafilatura → Readability → BeautifulSoup)
        loader = HTMLLoader("https://example.com/article")
        docs = loader.load()

        # 로컬 HTML 파일
        loader = HTMLLoader("page.html")
        docs = loader.load()

        # fallback chain 커스터마이징
        loader = HTMLLoader(
            "https://example.com",
            fallback_chain=["trafilatura", "beautifulsoup"]  # Readability 제외
        )
        docs = loader.load()
        ```
    """

    def __init__(
        self,
        source: Union[str, Path],
        fallback_chain: Optional[List[str]] = None,
        encoding: str = "utf-8",
        **kwargs,
    ):
        """
        Args:
            source: URL 또는 파일 경로
            fallback_chain: fallback 순서 (기본: ["trafilatura", "readability", "beautifulsoup"])
            encoding: 파일 인코딩 (로컬 파일만 해당)
            **kwargs: 추가 파라미터
                - headers: HTTP 헤더 (URL만 해당)
                - timeout: 타임아웃 초 (URL만 해당, 기본: 10)
        """
        self.source = source
        self.fallback_chain = fallback_chain or ["trafilatura", "readability", "beautifulsoup"]
        self.encoding = encoding
        self.headers = kwargs.get("headers", {})
        self.timeout = kwargs.get("timeout", 10)

        # URL 여부 판단
        self.is_url = isinstance(source, str) and (
            source.startswith("http://") or source.startswith("https://")
        )

    def load(self) -> List[Document]:
        """HTML 로딩"""
        try:
            # HTML 가져오기
            if self.is_url:
                html_content = self._fetch_url()
                metadata = {"source": self.source, "type": "url"}
            else:
                html_content = self._read_file()
                metadata = {"source": str(Path(self.source)), "type": "file"}

            # Multi-tier fallback으로 파싱
            text_content, parser_used = self._parse_html(html_content)

            # 메타데이터 추출 (Trafilatura 사용 시)
            if parser_used == "trafilatura":
                extra_metadata = self._extract_metadata_trafilatura(html_content)
                metadata.update(extra_metadata)

            metadata["parser"] = parser_used

            return [Document(content=text_content, metadata=metadata)]

        except Exception as e:
            logger.error(f"Failed to load HTML from {self.source}: {e}")
            raise

    def lazy_load(self):
        """지연 로딩"""
        yield from self.load()

    def _fetch_url(self) -> str:
        """URL에서 HTML 가져오기"""
        try:
            import requests
        except ImportError:
            raise ImportError("requests is required for URL loading. Install: pip install requests")

        try:
            response = requests.get(self.source, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            response.encoding = response.apparent_encoding or "utf-8"
            return response.text
        except Exception as e:
            logger.error(f"Failed to fetch {self.source}: {e}")
            raise

    def _read_file(self) -> str:
        """로컬 파일에서 HTML 읽기"""
        file_path = Path(self.source)
        with open(file_path, "r", encoding=self.encoding) as f:
            return f.read()

    def _parse_html(self, html_content: str) -> tuple[str, str]:
        """
        Multi-tier fallback으로 HTML 파싱

        Returns:
            (text_content, parser_used)
        """
        for parser in self.fallback_chain:
            try:
                if parser == "trafilatura":
                    text = self._parse_with_trafilatura(html_content)
                    if text and len(text.strip()) > 50:  # 최소 길이 체크
                        logger.info("HTML parsed with Trafilatura")
                        return text, "trafilatura"

                elif parser == "readability":
                    text = self._parse_with_readability(html_content)
                    if text and len(text.strip()) > 50:
                        logger.info("HTML parsed with Readability (fallback 1)")
                        return text, "readability"

                elif parser == "beautifulsoup":
                    text = self._parse_with_beautifulsoup(html_content)
                    if text and len(text.strip()) > 50:
                        logger.info("HTML parsed with BeautifulSoup (fallback 2)")
                        return text, "beautifulsoup"

            except Exception as e:
                logger.warning(f"Parser {parser} failed: {e}")
                continue

        # 모든 파서 실패 시 마지막 수단 (raw text)
        logger.warning("All parsers failed, using raw text extraction")
        return self._parse_with_beautifulsoup(html_content), "beautifulsoup"

    def _parse_with_trafilatura(self, html_content: str) -> str:
        """Trafilatura로 파싱 (추천)"""
        try:
            import trafilatura
        except ImportError:
            raise ImportError(
                "trafilatura is required. Install: pip install trafilatura"
            )

        text = trafilatura.extract(
            html_content,
            include_comments=False,
            include_tables=True,
            no_fallback=False,  # fallback 활성화
        )
        return text or ""

    def _parse_with_readability(self, html_content: str) -> str:
        """Readability로 파싱 (fallback 1)"""
        try:
            from readability import Document as ReadabilityDocument
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "readability-lxml and beautifulsoup4 required. "
                "Install: pip install readability-lxml beautifulsoup4"
            )

        doc = ReadabilityDocument(html_content)
        content_html = doc.summary()

        # BeautifulSoup로 텍스트 추출
        soup = BeautifulSoup(content_html, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        return text

    def _parse_with_beautifulsoup(self, html_content: str) -> str:
        """BeautifulSoup로 파싱 (fallback 2)"""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "beautifulsoup4 required. Install: pip install beautifulsoup4"
            )

        soup = BeautifulSoup(html_content, "html.parser")

        # script, style 태그 제거
        for tag in soup(["script", "style", "meta", "link"]):
            tag.decompose()

        # 텍스트 추출
        text = soup.get_text(separator="\n", strip=True)
        return text

    def _extract_metadata_trafilatura(self, html_content: str) -> dict:
        """Trafilatura로 메타데이터 추출"""
        try:
            import trafilatura
        except ImportError:
            return {}

        try:
            metadata = trafilatura.extract_metadata(html_content)
            if metadata:
                return {
                    "title": metadata.title or "",
                    "author": metadata.author or "",
                    "date": metadata.date or "",
                    "description": metadata.description or "",
                    "sitename": metadata.sitename or "",
                }
        except Exception as e:
            logger.warning(f"Failed to extract metadata: {e}")

        return {}


class JupyterLoader(BaseDocumentLoader):
    """
    Jupyter Notebook 로더 (.ipynb, 2024-2025)

    Jupyter notebook 파일을 로드하여 코드 셀, 마크다운 셀, 출력을 추출합니다.

    Features:
    - 코드 셀 추출 (실행 순서 보존)
    - 마크다운 셀 추출
    - 셀 출력 포함/제외 옵션
    - 메타데이터 보존 (셀 타입, 실행 횟수)

    Example:
        ```python
        from beanllm.domain.loaders import JupyterLoader

        # 기본 (출력 포함)
        loader = JupyterLoader("analysis.ipynb", include_outputs=True)
        docs = loader.load()

        # 코드만 (출력 제외)
        loader = JupyterLoader("notebook.ipynb", include_outputs=False)
        docs = loader.load()

        # 셀 타입 필터링
        loader = JupyterLoader(
            "notebook.ipynb",
            filter_cell_types=["code"]  # 코드 셀만
        )
        docs = loader.load()
        ```
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        include_outputs: bool = True,
        filter_cell_types: Optional[List[str]] = None,
        concatenate_cells: bool = True,
        **kwargs,
    ):
        """
        Args:
            file_path: .ipynb 파일 경로
            include_outputs: 셀 출력 포함 여부 (기본: True)
            filter_cell_types: 포함할 셀 타입 (기본: None = 모두)
                - ["code"]: 코드 셀만
                - ["markdown"]: 마크다운 셀만
                - ["code", "markdown"]: 둘 다
            concatenate_cells: 모든 셀을 하나의 Document로 결합 (기본: True)
            **kwargs: 추가 파라미터
        """
        self.file_path = Path(file_path)
        self.include_outputs = include_outputs
        self.filter_cell_types = filter_cell_types
        self.concatenate_cells = concatenate_cells

    def load(self) -> List[Document]:
        """Jupyter Notebook 로딩"""
        try:
            import nbformat
        except ImportError:
            raise ImportError(
                "nbformat is required for JupyterLoader. Install: pip install nbformat"
            )

        try:
            # Notebook 로드
            with open(self.file_path, "r", encoding="utf-8") as f:
                notebook = nbformat.read(f, as_version=4)

            # 메타데이터 추출
            nb_metadata = {
                "source": str(self.file_path),
                "kernel": notebook.metadata.get("kernelspec", {}).get("name", "unknown"),
                "language": notebook.metadata.get("kernelspec", {}).get("language", "unknown"),
            }

            # 셀 처리
            if self.concatenate_cells:
                # 모든 셀을 하나의 Document로
                content_parts = []

                for idx, cell in enumerate(notebook.cells):
                    # 셀 타입 필터링
                    if self.filter_cell_types and cell.cell_type not in self.filter_cell_types:
                        continue

                    cell_content = self._format_cell(cell, idx)
                    if cell_content:
                        content_parts.append(cell_content)

                combined_content = "\n\n" + "="*80 + "\n\n".join(content_parts)

                return [Document(content=combined_content, metadata=nb_metadata)]

            else:
                # 각 셀을 별도 Document로
                documents = []

                for idx, cell in enumerate(notebook.cells):
                    if self.filter_cell_types and cell.cell_type not in self.filter_cell_types:
                        continue

                    cell_content = self._format_cell(cell, idx)
                    if cell_content:
                        cell_metadata = nb_metadata.copy()
                        cell_metadata.update({
                            "cell_index": idx,
                            "cell_type": cell.cell_type,
                            "execution_count": cell.get("execution_count"),
                        })

                        documents.append(Document(content=cell_content, metadata=cell_metadata))

                return documents

        except Exception as e:
            logger.error(f"Failed to load Jupyter notebook {self.file_path}: {e}")
            raise

    def lazy_load(self):
        """지연 로딩"""
        yield from self.load()

    def _format_cell(self, cell, idx: int) -> str:
        """셀 포맷팅"""
        parts = []

        # 셀 헤더
        cell_type = cell.cell_type.upper()
        exec_count = cell.get("execution_count", "")
        if exec_count:
            header = f"[{idx}] {cell_type} (execution {exec_count})"
        else:
            header = f"[{idx}] {cell_type}"

        parts.append(header)
        parts.append("-" * 80)

        # 셀 소스 코드/마크다운
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)

        if source.strip():
            parts.append(source)

        # 출력 (코드 셀만, include_outputs=True일 때)
        if self.include_outputs and cell.cell_type == "code":
            outputs = cell.get("outputs", [])
            if outputs:
                parts.append("\n--- OUTPUT ---")
                for output in outputs:
                    output_text = self._format_output(output)
                    if output_text:
                        parts.append(output_text)

        return "\n".join(parts)

    def _format_output(self, output) -> str:
        """셀 출력 포맷팅"""
        output_type = output.get("output_type", "")

        if output_type == "stream":
            # 표준 출력/에러
            text = output.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            return text

        elif output_type == "execute_result" or output_type == "display_data":
            # 실행 결과/디스플레이 데이터
            data = output.get("data", {})

            # 텍스트 표현 우선
            if "text/plain" in data:
                text = data["text/plain"]
                if isinstance(text, list):
                    text = "".join(text)
                return text

            # HTML (간단히 표시)
            elif "text/html" in data:
                return "[HTML OUTPUT]"

            # 이미지 (경로 표시)
            elif any(k.startswith("image/") for k in data.keys()):
                image_formats = [k for k in data.keys() if k.startswith("image/")]
                return f"[IMAGE: {', '.join(image_formats)}]"

        elif output_type == "error":
            # 에러
            ename = output.get("ename", "Error")
            evalue = output.get("evalue", "")
            traceback = output.get("traceback", [])

            error_parts = [f"{ename}: {evalue}"]
            if traceback:
                error_parts.append("\n".join(traceback))

            return "\n".join(error_parts)

        return ""


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
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import InputFormat
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
