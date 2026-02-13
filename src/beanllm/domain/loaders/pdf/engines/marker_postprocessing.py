"""
Marker Engine - Post-processing Mixin

marker-pdf 원시 결과를 PDFLoadResult 형식으로 변환하는 로직을 담당합니다.
"""

from typing import Any, Dict, List

from beanllm.domain.loaders.pdf.engines.marker_extraction import MarkerExtractionMixin


class MarkerPostprocessingMixin(MarkerExtractionMixin):
    """
    marker-pdf 결과를 표준 형식으로 변환하는 Mixin.

    - 페이지/테이블/이미지 조합
    - 메타데이터 구성
    """

    # Mixin이 참조하는 속성 (MarkerEngine에서 초기화됨)
    name: str

    def _convert_marker_result(
        self,
        full_text: str,
        images: Dict,
        marker_metadata: Dict,
        config: Dict,
    ) -> Dict[str, Any]:
        """
        marker-pdf 결과를 PDFLoadResult 형식으로 변환

        Args:
            full_text: marker-pdf가 추출한 Markdown 텍스트
            images: marker-pdf가 추출한 이미지 딕셔너리
            marker_metadata: marker-pdf 메타데이터
            config: 설정 딕셔너리

        Returns:
            Dict: PDFLoadResult 형식 딕셔너리
        """
        pages = self._split_into_pages(full_text, marker_metadata)

        tables: List[Dict[str, Any]] = []
        if config.get("extract_tables", True):
            tables = self._extract_tables_from_markdown(full_text, pages)

        image_list: List[Dict[str, Any]] = []
        if config.get("extract_images", True) and images:
            image_list = self._convert_images(images)

        markdown_text = full_text if config.get("to_markdown", True) else None

        return {
            "pages": pages,
            "tables": tables,
            "images": image_list,
            "markdown": markdown_text,
            "metadata": {
                "total_pages": len(pages),
                "engine": self.name,
                "marker_metadata": marker_metadata,
                "quality_score": 0.98,
            },
        }
