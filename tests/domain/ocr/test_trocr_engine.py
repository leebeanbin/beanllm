"""
TrOCR Engine 테스트

Note: transformers와 torch가 설치되지 않은 경우 대부분의 테스트는 skip됩니다.
      설치: pip install transformers torch
"""

import numpy as np
import pytest

from beanllm.domain.ocr.models import OCRConfig

# transformers 설치 여부 체크
try:
    import transformers  # noqa: F401
    import torch  # noqa: F401

    HAS_TROCR = True
except ImportError:
    HAS_TROCR = False

skip_without_trocr = pytest.mark.skipif(not HAS_TROCR, reason="transformers/torch not installed")


class TestTrOCREngineImport:
    """TrOCREngine import 테스트"""

    def test_trocr_engine_import_without_dependencies(self):
        """transformers 없이 import 시도 (의존성 체크 테스트)"""
        if not HAS_TROCR:
            from beanllm.domain.ocr.engines.trocr_engine import TrOCREngine

            with pytest.raises(ImportError, match="transformers and torch are required"):
                TrOCREngine()
        else:
            from beanllm.domain.ocr.engines.trocr_engine import TrOCREngine

            engine = TrOCREngine()
            assert engine.name == "TrOCR"


@skip_without_trocr
class TestTrOCREngineWithDependencies:
    """transformers가 설치된 경우의 테스트"""

    def test_trocr_engine_initialization(self):
        """TrOCREngine 초기화 테스트"""
        from beanllm.domain.ocr.engines.trocr_engine import TrOCREngine

        engine = TrOCREngine()
        assert engine.name == "TrOCR"
        assert engine._model is None  # Lazy loading

    def test_trocr_repr(self):
        """__repr__ 테스트"""
        from beanllm.domain.ocr.engines.trocr_engine import TrOCREngine

        engine = TrOCREngine()
        repr_str = repr(engine)
        assert "TrOCREngine" in repr_str
        assert "not loaded" in repr_str


class TestTrOCREngineIntegration:
    """beanOCR 통합 테스트"""

    @skip_without_trocr
    def test_trocr_in_bean_ocr(self):
        """beanOCR에서 TrOCR 사용 테스트"""
        from beanllm.domain.ocr import beanOCR

        ocr = beanOCR(engine="trocr", language="en")
        assert ocr._engine is not None
        assert ocr._engine.name == "TrOCR"

    def test_trocr_import_error_handling(self):
        """transformers 미설치 시 에러 처리 테스트"""
        if not HAS_TROCR:
            from beanllm.domain.ocr import beanOCR

            with pytest.raises(ImportError, match="transformers and torch are required"):
                beanOCR(engine="trocr")
        else:
            from beanllm.domain.ocr import beanOCR

            ocr = beanOCR(engine="trocr")
            assert ocr._engine.name == "TrOCR"


# 실제 OCR 테스트 (매우 느림, 모델 다운로드 필요)
@pytest.mark.slow
@skip_without_trocr
class TestTrOCREngineRealOCR:
    """실제 OCR 기능 테스트 (매우 느림, optional)"""

    def test_trocr_recognize_simple_image(self):
        """간단한 이미지 OCR 테스트"""
        from beanllm.domain.ocr.engines.trocr_engine import TrOCREngine

        engine = TrOCREngine()
        config = OCRConfig(language="en", use_gpu=False)

        # 간단한 테스트 이미지 (빈 이미지)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        result = engine.recognize(image, config)

        assert "text" in result
        assert "lines" in result
        assert "confidence" in result
        assert "language" in result
