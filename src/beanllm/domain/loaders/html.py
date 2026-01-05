"""
HTML Loader

HTML 파일 로더 (BeautifulSoup)
"""

import logging
import mmap
import re
from pathlib import Path
from typing import Iterator, List, Optional, Union

from .base import BaseDocumentLoader
from .security import validate_file_path
from .types import Document

try:
    from beanllm.utils.logger import get_logger
except ImportError:
    def get_logger(name: str):
        return logging.getLogger(name)

logger = get_logger(__name__)

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
            import httpx
        except ImportError:
            raise ImportError("requests is required for URL loading. Install: pip install requests")

        try:
            response = httpx.get(self.source, headers=self.headers, timeout=self.timeout)
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
            from bs4 import BeautifulSoup
            from readability import Document as ReadabilityDocument
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


