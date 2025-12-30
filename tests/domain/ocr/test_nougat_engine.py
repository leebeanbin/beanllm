"""
Nougat Engine 테스트

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

    HAS_NOUGAT = True
except ImportError:
    HAS_NOUGAT = False

skip_without_nougat = pytest.mark.skipif(
    not HAS_NOUGAT, reason="transformers/torch not installed"
)


class TestNougatEngineImport:
    """NougatEngine import 테스트"""

    def test_nougat_engine_import_without_dependencies(self):
        """transformers 없이 import 시도 (의존성 체크 테스트)"""
        if not HAS_NOUGAT:
            from beanllm.domain.ocr.engines.nougat_engine import NougatEngine

            with pytest.raises(ImportError, match="torch and transformers are required"):
                NougatEngine()
        else:
            from beanllm.domain.ocr.engines.nougat_engine import NougatEngine

            engine = NougatEngine()
            assert engine.name == "Nougat"


@skip_without_nougat
class TestNougatEngineWithDependencies:
    """transformers가 설치된 경우의 테스트"""

    def test_nougat_engine_initialization(self):
        """NougatEngine 초기화 테스트"""
        from beanllm.domain.ocr.engines.nougat_engine import NougatEngine

        engine = NougatEngine()
        assert engine.name == "Nougat"
        assert engine._model is None  # Lazy loading

    def test_nougat_repr(self):
        """__repr__ 테스트"""
        from beanllm.domain.ocr.engines.nougat_engine import NougatEngine

        engine = NougatEngine()
        repr_str = repr(engine)
        assert "NougatEngine" in repr_str
        assert "not loaded" in repr_str


class TestNougatEngineIntegration:
    """beanOCR 통합 테스트"""

    @skip_without_nougat
    def test_nougat_in_bean_ocr(self):
        """beanOCR에서 Nougat 사용 테스트"""
        from beanllm.domain.ocr import beanOCR

        ocr = beanOCR(engine="nougat", language="en")
        assert ocr._engine is not None
        assert ocr._engine.name == "Nougat"

    def test_nougat_import_error_handling(self):
        """transformers 미설치 시 에러 처리 테스트"""
        if not HAS_NOUGAT:
            from beanllm.domain.ocr import beanOCR

            with pytest.raises(ImportError, match="transformers and torch are required"):
                beanOCR(engine="nougat")
        else:
            from beanllm.domain.ocr import beanOCR

            ocr = beanOCR(engine="nougat")
            assert ocr._engine.name == "Nougat"


# 실제 OCR 테스트 (매우 느림, 대용량 모델 다운로드 필요)
@pytest.mark.slow
@skip_without_nougat
class TestNougatEngineRealOCR:
    """실제 OCR 기능 테스트 (매우 느림, optional)"""

    def test_nougat_recognize_simple_image(self):
        """간단한 이미지 OCR 테스트"""
        from beanllm.domain.ocr.engines.nougat_engine import NougatEngine

        engine = NougatEngine()
        config = OCRConfig(language="en", use_gpu=False)

        # 간단한 테스트 이미지 (논문 페이지 크기)
        image = np.zeros((1000, 800, 3), dtype=np.uint8)

        result = engine.recognize(image, config)

        assert "text" in result
        assert "lines" in result
        assert "confidence" in result
        assert "language" in result
