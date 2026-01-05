"""
beanPDFLoader extractors - 메타데이터 추출 및 조회

테이블과 이미지 메타데이터를 구조화하여 효율적으로 조회할 수 있게 합니다.
"""

from .image_extractor import ImageExtractor
from .table_extractor import TableExtractor

__all__ = [
    "TableExtractor",
    "ImageExtractor",
]

