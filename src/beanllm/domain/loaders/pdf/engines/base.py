"""
Base PDF Engine - 추상 기본 클래스

모든 PDF 파싱 엔진이 상속받아야 하는 추상 클래스입니다.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Union

try:
    from beanllm.utils.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


class BasePDFEngine(ABC):
    """
    PDF 파싱 엔진의 추상 기본 클래스

    모든 PDF 엔진은 이 클래스를 상속받아 extract() 메서드를 구현해야 합니다.

    Example:
        ```python
        class MyPDFEngine(BasePDFEngine):
            def extract(self, pdf_path, config):
                # 구현
                return {
                    "pages": [...],
                    "metadata": {...}
                }
        ```
    """

    def __init__(self, name: Optional[str] = None):
        """
        Args:
            name: 엔진 이름 (디버깅/로깅용)
        """
        self.name = name or self.__class__.__name__
        self._check_dependencies()

    @abstractmethod
    def extract(
        self,
        pdf_path: Union[str, Path],
        config: Dict,
    ) -> Dict:
        """
        PDF 파일에서 텍스트 및 메타데이터 추출

        Args:
            pdf_path: PDF 파일 경로
            config: 추출 설정 딕셔너리
                - extract_tables: bool - 테이블 추출 여부
                - extract_images: bool - 이미지 추출 여부
                - max_pages: Optional[int] - 최대 페이지 수
                - page_range: Optional[tuple] - (start, end) 페이지 범위

        Returns:
            Dict containing:
                - pages: List[Dict] - 페이지별 데이터
                    - page: int - 페이지 번호 (0-based)
                    - text: str - 추출된 텍스트
                    - width: float - 페이지 너비
                    - height: float - 페이지 높이
                    - metadata: Dict - 추가 메타데이터
                - tables: List[Dict] - 추출된 테이블 (extract_tables=True일 때)
                - images: List[Dict] - 추출된 이미지 (extract_images=True일 때)
                - metadata: Dict - 전체 문서 메타데이터
                    - total_pages: int
                    - engine: str - 사용된 엔진 이름
                    - processing_time: float - 처리 시간 (초)

        Raises:
            NotImplementedError: 서브클래스에서 구현하지 않은 경우
            Exception: PDF 파싱 실패 시
        """
        raise NotImplementedError(f"{self.__class__.__name__}.extract() must be implemented")

    def _check_dependencies(self) -> None:
        """
        필수 의존성 라이브러리 확인

        서브클래스에서 오버라이드하여 특정 라이브러리 필요 여부 확인
        """
        pass

    def _validate_pdf_path(self, pdf_path: Union[str, Path]) -> Path:
        """
        PDF 경로 검증 및 Path 객체 변환

        Args:
            pdf_path: PDF 파일 경로

        Returns:
            Path 객체

        Raises:
            FileNotFoundError: 파일이 존재하지 않을 때
            ValueError: PDF 파일이 아닐 때
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if not pdf_path.is_file():
            raise ValueError(f"Path is not a file: {pdf_path}")

        if pdf_path.suffix.lower() != ".pdf":
            logger.warning(f"File extension is not .pdf: {pdf_path}")

        return pdf_path

    def get_engine_info(self) -> Dict[str, str]:
        """
        엔진 정보 반환

        Returns:
            엔진 이름 및 버전 정보
        """
        return {
            "name": self.name,
            "class": self.__class__.__name__,
        }

