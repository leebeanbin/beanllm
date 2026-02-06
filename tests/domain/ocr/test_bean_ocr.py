"""
beanOCR 메인 클래스 테스트
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from PIL import Image

from beanllm.domain.ocr import OCRConfig, beanOCR
from beanllm.domain.ocr.engines.base import BaseOCREngine
from beanllm.domain.ocr.models import BoundingBox, OCRTextLine


class MockOCREngine(BaseOCREngine):
    """테스트용 Mock OCR 엔진"""

    def __init__(self):
        super().__init__(name="MockOCR")

    def recognize(self, image, config):
        """Mock OCR 실행"""
        bbox = BoundingBox(x0=10, y0=20, x1=100, y1=50, confidence=0.95)
        line = OCRTextLine(
            text="Mock OCR Result", bbox=bbox, confidence=0.9, language=config.language
        )

        return {
            "text": "Mock OCR Result",
            "lines": [line],
            "confidence": 0.9,
            "language": config.language,
            "metadata": {},
        }


class TestBeanOCRInitialization:
    """beanOCR 초기화 테스트"""

    def test_bean_ocr_init_with_config(self):
        """OCRConfig 객체로 초기화"""
        config = OCRConfig(engine="paddleocr", language="ko")

        # PaddleOCR가 설치되지 않았으면 ImportError 발생
        try:
            import paddleocr  # noqa: F401

            # paddleocr가 설치된 경우 정상 초기화
            ocr = beanOCR(config=config)
            assert ocr.config.engine == "paddleocr"
        except ImportError:
            # paddleocr가 없으면 ImportError 발생 예상
            with pytest.raises(ImportError, match="PaddleOCR is required"):
                beanOCR(config=config)

    def test_bean_ocr_init_with_kwargs(self):
        """kwargs로 초기화"""
        try:
            import paddleocr  # noqa: F401

            # paddleocr가 설치된 경우 정상 초기화
            ocr = beanOCR(engine="paddleocr", language="ko")
            assert ocr.config.engine == "paddleocr"
        except ImportError:
            # paddleocr가 없으면 ImportError 발생 예상
            with pytest.raises(ImportError, match="PaddleOCR is required"):
                beanOCR(engine="paddleocr", language="ko")

    def test_bean_ocr_init_with_mock_engine(self):
        """Mock 엔진으로 초기화"""
        with patch.object(beanOCR, "_create_engine", return_value=MockOCREngine()):
            ocr = beanOCR(engine="paddleocr", language="ko")
            assert ocr.config.engine == "paddleocr"
            assert ocr.config.language == "ko"
            assert ocr._engine is not None


class TestBeanOCRImageLoading:
    """beanOCR 이미지 로딩 테스트"""

    def test_load_numpy_array(self):
        """numpy array 이미지 로드"""
        with patch.object(beanOCR, "_create_engine", return_value=MockOCREngine()):
            ocr = beanOCR(engine="paddleocr")

            # numpy array 생성
            image = np.zeros((100, 100, 3), dtype=np.uint8)

            loaded = ocr._load_image(image)
            assert isinstance(loaded, np.ndarray)
            assert loaded.shape == (100, 100, 3)

    def test_load_pil_image(self):
        """PIL Image 로드"""
        with patch.object(beanOCR, "_create_engine", return_value=MockOCREngine()):
            ocr = beanOCR(engine="paddleocr")

            # PIL Image 생성
            pil_img = Image.new("RGB", (100, 100))

            loaded = ocr._load_image(pil_img)
            assert isinstance(loaded, np.ndarray)
            assert loaded.shape == (100, 100, 3)

    def test_load_image_file(self):
        """이미지 파일 로드"""
        with patch.object(beanOCR, "_create_engine", return_value=MockOCREngine()):
            ocr = beanOCR(engine="paddleocr")

            # 임시 이미지 파일 생성
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                img = Image.new("RGB", (100, 100))
                img.save(f.name)
                temp_path = f.name

            try:
                loaded = ocr._load_image(temp_path)
                assert isinstance(loaded, np.ndarray)
                assert loaded.shape == (100, 100, 3)
            finally:
                Path(temp_path).unlink()

    def test_load_image_file_not_found(self):
        """존재하지 않는 이미지 파일"""
        with patch.object(beanOCR, "_create_engine", return_value=MockOCREngine()):
            ocr = beanOCR(engine="paddleocr")

            with pytest.raises(FileNotFoundError):
                ocr._load_image("nonexistent.jpg")

    def test_load_image_rgba_to_rgb(self):
        """RGBA 이미지를 RGB로 변환"""
        with patch.object(beanOCR, "_create_engine", return_value=MockOCREngine()):
            ocr = beanOCR(engine="paddleocr")

            # RGBA PIL Image 생성
            pil_img = Image.new("RGBA", (100, 100))

            loaded = ocr._load_image(pil_img)
            assert isinstance(loaded, np.ndarray)
            assert loaded.shape == (100, 100, 3)  # RGB로 변환되어야 함


class TestBeanOCRRecognize:
    """beanOCR recognize() 메서드 테스트"""

    def test_recognize_with_numpy_array(self):
        """numpy array로 OCR 실행"""
        with patch.object(beanOCR, "_create_engine", return_value=MockOCREngine()):
            ocr = beanOCR(engine="paddleocr", language="ko")

            image = np.zeros((100, 100, 3), dtype=np.uint8)
            result = ocr.recognize(image)

            assert result.text == "Mock OCR Result"
            assert result.confidence == 0.9
            assert result.engine == "paddleocr"
            assert result.language == "ko"
            assert len(result.lines) == 1

    def test_recognize_with_pil_image(self):
        """PIL Image로 OCR 실행"""
        with patch.object(beanOCR, "_create_engine", return_value=MockOCREngine()):
            ocr = beanOCR(engine="paddleocr")

            pil_img = Image.new("RGB", (100, 100))
            result = ocr.recognize(pil_img)

            assert result.text == "Mock OCR Result"
            assert result.confidence == 0.9

    def test_recognize_with_image_file(self):
        """이미지 파일로 OCR 실행"""
        with patch.object(beanOCR, "_create_engine", return_value=MockOCREngine()):
            ocr = beanOCR(engine="paddleocr")

            # 임시 이미지 파일 생성
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                img = Image.new("RGB", (100, 100))
                img.save(f.name)
                temp_path = f.name

            try:
                result = ocr.recognize(temp_path)
                assert result.text == "Mock OCR Result"
                assert result.confidence == 0.9
            finally:
                Path(temp_path).unlink()

    def test_recognize_processing_time(self):
        """처리 시간 측정"""
        with patch.object(beanOCR, "_create_engine", return_value=MockOCREngine()):
            ocr = beanOCR(engine="paddleocr")

            image = np.zeros((100, 100, 3), dtype=np.uint8)
            result = ocr.recognize(image)

            assert result.processing_time > 0


class TestBeanOCRPDFRecognize:
    """beanOCR recognize_pdf_page() 메서드 테스트"""

    def test_recognize_pdf_page_file_not_found(self):
        """존재하지 않는 PDF 파일"""
        with patch.object(beanOCR, "_create_engine", return_value=MockOCREngine()):
            ocr = beanOCR(engine="paddleocr")

            with pytest.raises(FileNotFoundError):
                ocr.recognize_pdf_page("nonexistent.pdf", page_num=0)

    def test_recognize_pdf_page_mock(self):
        """Mock PDF 페이지 OCR"""
        with patch.object(beanOCR, "_create_engine", return_value=MockOCREngine()):
            ocr = beanOCR(engine="paddleocr")

            # fitz Mock
            mock_doc = MagicMock()
            mock_page = MagicMock()
            mock_pix = MagicMock()

            # pixmap 설정
            mock_pix.samples = np.zeros(100 * 100 * 3, dtype=np.uint8).tobytes()
            mock_pix.height = 100
            mock_pix.width = 100
            mock_pix.n = 3

            mock_page.get_pixmap.return_value = mock_pix
            mock_doc.__getitem__.return_value = mock_page
            mock_doc.__len__.return_value = 5

            # 임시 PDF 파일 생성 (실제 내용은 중요하지 않음)
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(b"fake pdf content")
                temp_path = f.name

            try:
                with patch("fitz.open", return_value=mock_doc):
                    result = ocr.recognize_pdf_page(temp_path, page_num=0)

                    assert result.text == "Mock OCR Result"
                    assert result.confidence == 0.9
                    mock_page.get_pixmap.assert_called_once()
            finally:
                Path(temp_path).unlink()

    def test_recognize_pdf_page_invalid_page_number(self):
        """잘못된 페이지 번호"""
        with patch.object(beanOCR, "_create_engine", return_value=MockOCREngine()):
            ocr = beanOCR(engine="paddleocr")

            # fitz Mock
            mock_doc = MagicMock()
            mock_doc.__len__.return_value = 5  # 5페이지 문서

            # 임시 PDF 파일
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(b"fake pdf")
                temp_path = f.name

            try:
                with patch("fitz.open", return_value=mock_doc):
                    # 페이지 번호 범위 초과
                    with pytest.raises(IndexError, match="Invalid page number"):
                        ocr.recognize_pdf_page(temp_path, page_num=10)
            finally:
                Path(temp_path).unlink()


class TestBeanOCRBatchRecognize:
    """beanOCR batch_recognize() 메서드 테스트"""

    def test_batch_recognize_empty_list(self):
        """빈 리스트 배치 처리"""
        with patch.object(beanOCR, "_create_engine", return_value=MockOCREngine()):
            ocr = beanOCR(engine="paddleocr")

            results = ocr.batch_recognize([])
            assert len(results) == 0

    def test_batch_recognize_multiple_images(self):
        """여러 이미지 배치 처리"""
        with patch.object(beanOCR, "_create_engine", return_value=MockOCREngine()):
            ocr = beanOCR(engine="paddleocr")

            images = [
                np.zeros((100, 100, 3), dtype=np.uint8),
                np.zeros((100, 100, 3), dtype=np.uint8),
                np.zeros((100, 100, 3), dtype=np.uint8),
            ]

            results = ocr.batch_recognize(images)

            assert len(results) == 3
            for result in results:
                assert result.text == "Mock OCR Result"
                assert result.confidence == 0.9


class TestBeanOCRRepr:
    """beanOCR __repr__ 테스트"""

    def test_repr(self):
        """문자열 표현 테스트"""
        with patch.object(beanOCR, "_create_engine", return_value=MockOCREngine()):
            ocr = beanOCR(engine="paddleocr", language="ko", use_gpu=True)
            repr_str = repr(ocr)

            assert "beanOCR" in repr_str
            assert "paddleocr" in repr_str
            assert "ko" in repr_str
            assert "True" in repr_str
