"""
이미지 메타데이터 추출 및 관리

Document 리스트에서 이미지 메타데이터를 추출하여 구조화된 형태로 제공합니다.
"""

from pathlib import Path
from typing import List, Optional


class ImageExtractor:
    """
    이미지 메타데이터 추출기

    Document 리스트에서 이미지 정보를 추출하여 구조화된 형태로 조회할 수 있게 합니다.

    Example:
        ```python
        from beanllm.domain.loaders import beanPDFLoader
        from beanllm.domain.loaders.pdf.extractors import ImageExtractor

        # PDF 로딩
        loader = beanPDFLoader("document.pdf", extract_images=True, strategy="fast")
        docs = loader.load()

        # 이미지 추출
        extractor = ImageExtractor(docs)

        # 모든 이미지 메타데이터
        images = extractor.get_all_images()
        for img in images:
            print(f"Page {img['page']}: {img['format']}, {img['width']}x{img['height']}")

        # 특정 페이지의 이미지만
        page_images = extractor.get_images_by_page(0)

        # 큰 이미지만 (width >= 800)
        large_images = extractor.get_images_by_size(min_width=800)
        ```
    """

    def __init__(self, documents: List):
        """
        Args:
            documents: beanPDFLoader.load() 결과 (Document 리스트)
        """
        self.documents = documents
        self._images_cache = None

    def get_all_images(self) -> List[dict]:
        """
        모든 이미지 메타데이터 추출

        Returns:
            이미지 정보 리스트, 각 항목은:
                - page: 페이지 번호 (0-based)
                - image_index: 페이지 내 이미지 인덱스
                - format: 이미지 포맷 (png, jpeg 등)
                - width: 이미지 너비 (픽셀)
                - height: 이미지 높이 (픽셀)
                - size: 파일 크기 (bytes)
                - source: 소스 파일 경로
        """
        if self._images_cache is not None:
            return self._images_cache

        images = []

        for doc in self.documents:
            if "images" not in doc.metadata:
                continue

            page = doc.metadata.get("page", 0)
            source = doc.metadata.get("source", "")

            for img_meta in doc.metadata["images"]:
                image_info = {
                    "page": page,
                    "image_index": img_meta.get("image_index", 0),
                    "format": img_meta.get("format", ""),
                    "width": img_meta.get("width", 0),
                    "height": img_meta.get("height", 0),
                    "size": img_meta.get("size", 0),
                    "source": source,
                }

                images.append(image_info)

        self._images_cache = images
        return images

    def get_images_by_page(self, page: int) -> List[dict]:
        """
        특정 페이지의 이미지만 추출

        Args:
            page: 페이지 번호 (0-based)

        Returns:
            해당 페이지의 이미지 리스트
        """
        all_images = self.get_all_images()
        return [img for img in all_images if img["page"] == page]

    def get_images_by_format(self, format: str) -> List[dict]:
        """
        특정 포맷의 이미지만 추출

        Args:
            format: 이미지 포맷 (예: "png", "jpeg", "jpg")

        Returns:
            해당 포맷의 이미지 리스트
        """
        all_images = self.get_all_images()
        format_lower = format.lower()
        return [img for img in all_images if img["format"].lower() == format_lower]

    def get_images_by_size(
        self,
        min_width: Optional[int] = None,
        max_width: Optional[int] = None,
        min_height: Optional[int] = None,
        max_height: Optional[int] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
    ) -> List[dict]:
        """
        크기 기준으로 이미지 필터링

        Args:
            min_width: 최소 너비 (픽셀)
            max_width: 최대 너비 (픽셀)
            min_height: 최소 높이 (픽셀)
            max_height: 최대 높이 (픽셀)
            min_size: 최소 파일 크기 (bytes)
            max_size: 최대 파일 크기 (bytes)

        Returns:
            조건을 만족하는 이미지 리스트
        """
        all_images = self.get_all_images()
        filtered = all_images

        if min_width is not None:
            filtered = [img for img in filtered if img["width"] >= min_width]
        if max_width is not None:
            filtered = [img for img in filtered if img["width"] <= max_width]
        if min_height is not None:
            filtered = [img for img in filtered if img["height"] >= min_height]
        if max_height is not None:
            filtered = [img for img in filtered if img["height"] <= max_height]
        if min_size is not None:
            filtered = [img for img in filtered if img["size"] >= min_size]
        if max_size is not None:
            filtered = [img for img in filtered if img["size"] <= max_size]

        return filtered

    def get_large_images(self, min_dimension: int = 800) -> List[dict]:
        """
        큰 이미지만 추출 (width 또는 height가 min_dimension 이상)

        Args:
            min_dimension: 최소 차원 (기본: 800px)

        Returns:
            큰 이미지 리스트
        """
        all_images = self.get_all_images()
        return [
            img for img in all_images
            if img["width"] >= min_dimension or img["height"] >= min_dimension
        ]

    def get_summary(self) -> dict:
        """
        이미지 추출 요약 정보

        Returns:
            요약 정보:
                - total_images: 전체 이미지 수
                - pages_with_images: 이미지가 있는 페이지 수
                - images_by_page: 페이지별 이미지 수
                - formats: 포맷별 이미지 수
                - avg_width: 평균 너비 (픽셀)
                - avg_height: 평균 높이 (픽셀)
                - total_size: 전체 파일 크기 (bytes)
        """
        all_images = self.get_all_images()

        if not all_images:
            return {
                "total_images": 0,
                "pages_with_images": 0,
                "images_by_page": {},
                "formats": {},
                "avg_width": 0,
                "avg_height": 0,
                "total_size": 0,
            }

        pages_with_images = set(img["page"] for img in all_images)

        images_by_page = {}
        for img in all_images:
            page = img["page"]
            images_by_page[page] = images_by_page.get(page, 0) + 1

        formats = {}
        for img in all_images:
            fmt = img["format"]
            formats[fmt] = formats.get(fmt, 0) + 1

        avg_width = sum(img["width"] for img in all_images) / len(all_images)
        avg_height = sum(img["height"] for img in all_images) / len(all_images)
        total_size = sum(img["size"] for img in all_images)

        return {
            "total_images": len(all_images),
            "pages_with_images": len(pages_with_images),
            "images_by_page": images_by_page,
            "formats": formats,
            "avg_width": int(avg_width),
            "avg_height": int(avg_height),
            "total_size": total_size,
        }

    def export_manifest(self, output_path: Optional[str] = None) -> str:
        """
        이미지 매니페스트를 Markdown 형식으로 내보내기

        Args:
            output_path: 출력 파일 경로 (None이면 문자열만 반환)

        Returns:
            Markdown 문자열
        """
        all_images = self.get_all_images()
        summary = self.get_summary()

        md_lines = ["# Image Manifest\n"]
        md_lines.append("## Summary")
        md_lines.append(f"- Total Images: {summary['total_images']}")
        md_lines.append(f"- Pages with Images: {summary['pages_with_images']}")
        md_lines.append(f"- Average Size: {summary['avg_width']}x{summary['avg_height']}px")
        md_lines.append(f"- Total File Size: {summary['total_size']:,} bytes\n")

        md_lines.append("## Formats")
        for fmt, count in summary["formats"].items():
            md_lines.append(f"- {fmt}: {count} images")
        md_lines.append("")

        md_lines.append("## Images by Page\n")
        for img in all_images:
            md_lines.append(
                f"### Page {img['page'] + 1}, Image {img['image_index'] + 1}"
            )
            md_lines.append(f"- Format: {img['format']}")
            md_lines.append(f"- Size: {img['width']}x{img['height']}px")
            md_lines.append(f"- File Size: {img['size']:,} bytes")
            md_lines.append(f"- Source: {img['source']}\n")

        manifest_text = "\n".join(md_lines)

        if output_path:
            Path(output_path).write_text(manifest_text, encoding="utf-8")

        return manifest_text
