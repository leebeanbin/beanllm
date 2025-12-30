"""
Surya Engine 테스트

Note: surya-ocr가 설치되지 않은 경우 대부분의 테스트는 skip됩니다.
      설치: pip install surya-ocr torch
"""

import numpy as np
import pytest

from beanllm.domain.ocr.models import OCRConfig

# surya-ocr 설치 여부 체크
try:
    import surya  # noqa: F401
    import torch  # noqa: F401

    HAS_SURYA = True
except ImportError:
    HAS_SURYA = False

skip_without_surya = pytest.mark.skipif(not HAS_SURYA, reason="surya-ocr/torch not installed")


class TestSuryaEngineImport:
    """SuryaEngine import 테스트"""

    def test_surya_engine_import_without_dependencies(self):
        """surya-ocr 없이 import 시도 (의존성 체크 테스트)"""
        if not HAS_SURYA:
            from beanllm.domain.ocr.engines.surya_engine import SuryaEngine

            with pytest.raises(ImportError, match="surya-ocr and torch are required"):
                SuryaEngine()
        else:
            from beanllm.domain.ocr.engines.surya_engine import SuryaEngine

            engine = SuryaEngine()
            assert engine.name == "Surya"


@skip_without_surya
class TestSuryaEngineWithDependencies:
    """surya-ocr가 설치된 경우의 테스트"""

    def test_surya_engine_initialization(self):
        """SuryaEngine 초기화 테스트"""
        from beanllm.domain.ocr.engines.surya_engine import SuryaEngine

        engine = SuryaEngine()
        assert engine.name == "Surya"
        assert engine._model is None  # Lazy loading

    def test_surya_repr(self):
        """__repr__ 테스트"""
        from beanllm.domain.ocr.engines.surya_engine import SuryaEngine

        engine = SuryaEngine()
        repr_str = repr(engine)
        assert "SuryaEngine" in repr_str
        assert "not loaded" in repr_str


class TestSuryaEngineIntegration:
    """beanOCR 통합 테스트"""

    @skip_without_surya
    def test_surya_in_bean_ocr(self):
        """beanOCR에서 Surya 사용 테스트"""
        from beanllm.domain.ocr import beanOCR

        ocr = beanOCR(engine="surya", language="ko")
        assert ocr._engine is not None
        assert ocr._engine.name == "Surya"

    def test_surya_import_error_handling(self):
        """surya-ocr 미설치 시 에러 처리 테스트"""
        if not HAS_SURYA:
            from beanllm.domain.ocr import beanOCR

            with pytest.raises(ImportError, match="surya-ocr and torch are required"):
                beanOCR(engine="surya")
        else:
            from beanllm.domain.ocr import beanOCR

            ocr = beanOCR(engine="surya")
            assert ocr._engine.name == "Surya"


# 실제 OCR 테스트 (매우 느림, 대용량 모델 다운로드 필요)
@pytest.mark.slow
@skip_without_surya
class TestSuryaEngineRealOCR:
    """실제 OCR 기능 테스트 (매우 느림, optional)"""

    def test_surya_recognize_simple_image(self):
        """간단한 이미지 OCR 테스트"""
        from beanllm.domain.ocr.engines.surya_engine import SuryaEngine

        engine = SuryaEngine()
        config = OCRConfig(language="ko", use_gpu=False)

        # 간단한 테스트 이미지
        image = np.zeros((1000, 800, 3), dtype=np.uint8)

        result = engine.recognize(image, config)

        assert "text" in result
        assert "lines" in result
        assert "confidence" in result
        assert "language" in result
