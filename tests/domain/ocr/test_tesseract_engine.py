"""
Tesseract Engine 테스트

Note: pytesseract가 설치되지 않은 경우 대부분의 테스트는 skip됩니다.
      설치: pip install pytesseract && brew install tesseract (macOS)
"""

import numpy as np
import pytest

from beanllm.domain.ocr.models import OCRConfig

# pytesseract 설치 여부 체크
try:
    import pytesseract  # noqa: F401

    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

skip_without_tesseract = pytest.mark.skipif(not HAS_TESSERACT, reason="pytesseract not installed")


class TestTesseractEngineImport:
    """TesseractEngine import 테스트"""

    def test_tesseract_engine_import_without_pytesseract(self):
        """pytesseract 없이 import 시도 (의존성 체크 테스트)"""
        if not HAS_TESSERACT:
            from beanllm.domain.ocr.engines.tesseract_engine import TesseractEngine

            with pytest.raises(ImportError, match="pytesseract is required"):
                TesseractEngine()
        else:
            from beanllm.domain.ocr.engines.tesseract_engine import TesseractEngine

            engine = TesseractEngine()
            assert engine.name == "Tesseract"


@skip_without_tesseract
class TestTesseractEngineWithTesseract:
    """pytesseract가 설치된 경우의 테스트"""

    def test_tesseract_engine_initialization(self):
        """TesseractEngine 초기화 테스트"""
        from beanllm.domain.ocr.engines.tesseract_engine import TesseractEngine

        engine = TesseractEngine()
        assert engine.name == "Tesseract"

    def test_tesseract_language_code_mapping(self):
        """언어 코드 매핑 테스트"""
        from beanllm.domain.ocr.engines.tesseract_engine import TesseractEngine

        engine = TesseractEngine()

        assert engine._get_language_code("ko") == "kor"
        assert engine._get_language_code("en") == "eng"
        assert engine._get_language_code("zh") == "chi_sim"
        assert engine._get_language_code("ja") == "jpn"
        assert engine._get_language_code("auto") == "eng"
        assert engine._get_language_code("unknown") == "eng"

    def test_tesseract_repr(self):
        """__repr__ 테스트"""
        from beanllm.domain.ocr.engines.tesseract_engine import TesseractEngine

        engine = TesseractEngine()
        repr_str = repr(engine)
        assert "TesseractEngine" in repr_str


class TestTesseractEngineIntegration:
    """beanOCR 통합 테스트"""

    @skip_without_tesseract
    def test_tesseract_in_bean_ocr(self):
        """beanOCR에서 Tesseract 사용 테스트"""
        from beanllm.domain.ocr import beanOCR

        ocr = beanOCR(engine="tesseract", language="en")
        assert ocr._engine is not None
        assert ocr._engine.name == "Tesseract"

    def test_tesseract_import_error_handling(self):
        """pytesseract 미설치 시 에러 처리 테스트"""
        if not HAS_TESSERACT:
            from beanllm.domain.ocr import beanOCR

            with pytest.raises(ImportError, match="pytesseract is required"):
                beanOCR(engine="tesseract")
        else:
            from beanllm.domain.ocr import beanOCR

            ocr = beanOCR(engine="tesseract")
            assert ocr._engine.name == "Tesseract"


# 실제 OCR 테스트 (선택적으로만 실행)
@pytest.mark.slow
@skip_without_tesseract
class TestTesseractEngineRealOCR:
    """실제 OCR 기능 테스트 (느림, optional)"""

    def test_tesseract_recognize_simple_image(self):
        """간단한 이미지 OCR 테스트"""
        from beanllm.domain.ocr.engines.tesseract_engine import TesseractEngine

        engine = TesseractEngine()
        config = OCRConfig(language="en", use_gpu=False)

        # 간단한 테스트 이미지 (빈 이미지)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        result = engine.recognize(image, config)

        assert "text" in result
        assert "lines" in result
        assert "confidence" in result
        assert "language" in result
