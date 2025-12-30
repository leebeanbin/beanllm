"""
PyMuPDF Engine - Fast Layer

PyMuPDF (fitz)를 사용한 빠른 PDF 파싱 엔진
- 속도: ~2초/100페이지
- 정확도: 85%
- 특화: 대용량 문서, 이미지 추출
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from .base import BasePDFEngine

try:
    from ....utils.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


class PyMuPDFEngine(BasePDFEngine):
    """
    PyMuPDF 기반 PDF 파싱 엔진 (Fast Layer)

    빠른 처리 속도에 최적화된 엔진입니다.
    대용량 문서나 이미지 추출이 필요한 경우에 적합합니다.

    Example:
        ```python
        from beanllm.domain.loaders.pdf.engines import PyMuPDFEngine

        engine = PyMuPDFEngine()
        result = engine.extract("document.pdf", {
            "extract_tables": False,
            "extract_images": True,
            "max_pages": None
        })
        ```
    """

    def __init__(self, name: Optional[str] = None):
        """
        Args:
            name: 엔진 이름 (기본값: "PyMuPDF")
        """
        super().__init__(name=name or "PyMuPDF")

    def _check_dependencies(self) -> None:
        """PyMuPDF 라이브러리 확인"""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PyMuPDFEngine. "
                "Install it with: pip install PyMuPDF"
            )

    def extract(
        self,
        pdf_path: Union[str, Path],
        config: Dict,
    ) -> Dict:
        """
        PDF 파일에서 텍스트 및 메타데이터 추출 (PyMuPDF 사용)

        Args:
            pdf_path: PDF 파일 경로
            config: 추출 설정 딕셔너리
                - extract_tables: bool - 테이블 추출 여부 (기본: False, PyMuPDF는 테이블 추출 약함)
                - extract_images: bool - 이미지 추출 여부 (기본: False)
                - max_pages: Optional[int] - 최대 페이지 수
                - page_range: Optional[tuple] - (start, end) 페이지 범위

        Returns:
            Dict containing:
                - pages: List[Dict] - 페이지별 데이터
                - images: List[Dict] - 추출된 이미지 (extract_images=True일 때)
                - metadata: Dict - 전체 문서 메타데이터

        Raises:
            FileNotFoundError: PDF 파일이 없을 때
            Exception: PDF 파싱 실패 시
        """
        import fitz  # PyMuPDF

        start_time = time.time()
        pdf_path = self._validate_pdf_path(pdf_path)

        # 설정 추출
        extract_images = config.get("extract_images", False)
        max_pages = config.get("max_pages")
        page_range = config.get("page_range")

        try:
            # PDF 열기
            doc = fitz.open(pdf_path)

            pages_data = []
            images_data = []

            # 페이지 범위 결정
            total_pages = len(doc)
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

                page = doc[page_num]

                # 텍스트 추출 모드 선택
                text_mode = config.get("pymupdf_text_mode", "text")
                layout_analysis = config.get("layout_analysis", False)
                
                # layout_analysis=True이면 자동으로 "dict" 모드 사용
                if layout_analysis and text_mode == "text":
                    text_mode = "dict"
                
                # 텍스트 추출
                try:
                    if text_mode == "dict":
                        text_dict = page.get_text("dict")
                        text = self._extract_text_from_dict(text_dict)
                        structured_text = text_dict
                    elif text_mode in ["rawdict", "html", "xml", "json"]:
                        text = page.get_text(text_mode)
                        structured_text = None
                    else:
                        text = page.get_text()
                        structured_text = None
                except Exception as e:
                    logger.warning(f"Failed to extract text with mode '{text_mode}': {e}")
                    text = page.get_text()  # Fallback
                    structured_text = None

                # 페이지 메타데이터
                page_rect = page.rect
                page_metadata = {
                    "page_number": page_num + 1,  # 1-based for user
                    "rotation": page.rotation,
                }

                # 고급: 폰트 정보 추출
                extract_fonts = config.get("pymupdf_extract_fonts", False) or layout_analysis
                if extract_fonts:
                    try:
                        fonts = page.get_fonts()
                        if fonts:
                            page_metadata["fonts"] = [
                                {
                                    "name": font[3],  # font name
                                    "ext": font[1],  # extension
                                    "type": font[2],  # type
                                }
                                for font in fonts[:10]  # 최대 10개만
                            ]
                    except Exception as e:
                        logger.debug(f"Failed to extract fonts: {e}")

                # 고급: 링크 추출
                extract_links = config.get("pymupdf_extract_links", False) or layout_analysis
                if extract_links:
                    try:
                        links = page.get_links()
                        if links:
                            page_metadata["links"] = [
                                {
                                    "uri": link.get("uri", ""),
                                    "page": link.get("page", -1),
                                    "kind": link.get("kind", 0),
                                }
                                for link in links
                            ]
                    except Exception as e:
                        logger.debug(f"Failed to extract links: {e}")

                page_data = {
                    "page": page_num,  # 0-based
                    "text": text,
                    "width": page_rect.width,
                    "height": page_rect.height,
                    "metadata": page_metadata,
                }

                # 구조화된 텍스트 추가 (있는 경우)
                if structured_text:
                    page_data["structured_text"] = structured_text

                pages_data.append(page_data)

                # 이미지 추출 (요청된 경우)
                if extract_images:
                    page_images = self._extract_images_from_page(page, page_num)
                    images_data.extend(page_images)

            # 문서 메타데이터
            doc_metadata = doc.metadata
            processing_time = time.time() - start_time

            result = {
                "pages": pages_data,
                "metadata": {
                    "total_pages": total_pages,
                    "engine": self.name,
                    "processing_time": processing_time,
                    "file_path": str(pdf_path),
                    "file_size": pdf_path.stat().st_size,
                    "title": doc_metadata.get("title", ""),
                    "author": doc_metadata.get("author", ""),
                    "subject": doc_metadata.get("subject", ""),
                    "creator": doc_metadata.get("creator", ""),
                },
            }

            # 이미지가 있으면 추가
            if images_data:
                result["images"] = images_data

            doc.close()

            logger.info(
                f"PyMuPDF extracted {len(pages_data)} pages from {pdf_path} "
                f"in {processing_time:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"PyMuPDF extraction failed for {pdf_path}: {e}")
            raise

    def _extract_images_from_page(
        self,
        page: "fitz.Page",  # type: ignore
        page_num: int,
    ) -> List[Dict]:
        """
        페이지에서 이미지 추출

        Args:
            page: PyMuPDF Page 객체
            page_num: 페이지 번호 (0-based)

        Returns:
            이미지 정보 리스트
        """
        images = []

        try:
            # 이미지 리스트 가져오기
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                # 이미지 정보
                xref = img[0]
                base_image = page.parent.extract_image(xref)

                # bbox 추출 (PyMuPDF의 get_image_bbox 사용 - 가장 정확)
                bbox = None
                try:
                    # 방법 1: get_image_bbox() 사용 (가장 정확)
                    image_bbox = page.get_image_bbox(img)
                    bbox = (image_bbox.x0, image_bbox.y0, image_bbox.x1, image_bbox.y1)
                except Exception:
                    try:
                        # 방법 2: get_image_rects() 사용
                        if img_index < len(image_blocks):
                            rect = image_blocks[img_index]
                            bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
                    except Exception:
                        # 방법 3: 대체 방법 (이미지 크기로 추정)
                        bbox = (0.0, 0.0, float(base_image["width"]), float(base_image["height"]))

                # 이미지 메타데이터
                image_info = {
                    "page": page_num,
                    "image_index": img_index,
                    "format": base_image["ext"],
                    "width": base_image["width"],
                    "height": base_image["height"],
                    "size": len(base_image["image"]),
                    "bbox": bbox,
                    "metadata": {
                        "xref": xref,
                        "colorspace": base_image.get("colorspace", ""),
                        "bpc": base_image.get("bpc", 8),  # bits per component
                    },
                }

                images.append(image_info)

        except Exception as e:
            logger.warning(f"Failed to extract images from page {page_num}: {e}")

        return images

    def _extract_text_from_dict(self, text_dict: Dict) -> str:
        """
        구조화된 텍스트 딕셔너리에서 일반 텍스트 추출

        Args:
            text_dict: page.get_text("dict") 결과

        Returns:
            추출된 텍스트 문자열
        """
        text_parts = []
        
        if "blocks" in text_dict:
            for block in text_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        if "spans" in line:
                            for span in line["spans"]:
                                if "text" in span:
                                    text_parts.append(span["text"])
                        text_parts.append("\n")
        
        return "".join(text_parts).strip()

