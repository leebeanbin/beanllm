"""
PaddleOCR Engine 테스트

Note: PaddleOCR가 설치되지 않은 경우 대부분의 테스트는 skip됩니다.
      설치: pip install paddleocr
"""

import numpy as np
import pytest

from beanllm.domain.ocr.models import OCRConfig

# PaddleOCR 설치 여부 체크
try:
    import paddleocr  # noqa: F401

    HAS_PADDLEOCR = True
except ImportError:
    HAS_PADDLEOCR = False

skip_without_paddleocr = pytest.mark.skipif(
    not HAS_PADDLEOCR, reason="PaddleOCR not installed"
)


class TestPaddleOCREngineImport:
    """PaddleOCREngine import 테스트"""

    def test_paddleocr_engine_import_without_paddleocr(self):
        """paddleocr 없이 import 시도 (의존성 체크 테스트)"""
        if not HAS_PADDLEOCR:
            # paddleocr가 없을 때 PaddleOCREngine import는 성공하지만
            # 초기화 시 ImportError 발생
            from beanllm.domain.ocr.engines.paddleocr_engine import (
                PaddleOCREngine,
            )

            with pytest.raises(ImportError, match="PaddleOCR is required"):
                PaddleOCREngine()
        else:
            # paddleocr가 있을 때는 정상 동작
            from beanllm.domain.ocr.engines.paddleocr_engine import (
                PaddleOCREngine,
            )

            engine = PaddleOCREngine()
            assert engine.name == "PaddleOCR"


@skip_without_paddleocr
class TestPaddleOCREngineWithPaddleOCR:
    """PaddleOCR가 설치된 경우의 테스트"""

    def test_paddleocr_engine_initialization(self):
        """PaddleOCREngine 초기화 테스트"""
        from beanllm.domain.ocr.engines.paddleocr_engine import (
            PaddleOCREngine,
        )

        engine = PaddleOCREngine()
        assert engine.name == "PaddleOCR"
        assert len(engine._models) == 0

    def test_paddleocr_language_code_mapping(self):
        """언어 코드 매핑 테스트"""
        from beanllm.domain.ocr.engines.paddleocr_engine import (
            PaddleOCREngine,
        )

        engine = PaddleOCREngine()

        assert engine._get_language_code("ko") == "korean"
        assert engine._get_language_code("en") == "en"
        assert engine._get_language_code("zh") == "ch"
        assert engine._get_language_code("ja") == "japan"
        assert engine._get_language_code("auto") == "ch"
        assert engine._get_language_code("unknown") == "ch"

    def test_paddleocr_model_caching(self):
        """모델 캐싱 테스트 (실제 모델 로드 없이)"""
        from beanllm.domain.ocr.engines.paddleocr_engine import (
            PaddleOCREngine,
        )

        engine = PaddleOCREngine()

        # 모델이 아직 로드되지 않음
        assert len(engine._models) == 0

        # _get_or_create_model 호출 시 모델 캐싱 확인
        model1 = engine._get_or_create_model("ko", use_gpu=False)
        assert len(engine._models) == 1

        # 같은 언어/GPU 설정으로 재호출 시 캐시 사용
        model2 = engine._get_or_create_model("ko", use_gpu=False)
        assert len(engine._models) == 1
        assert model1 is model2  # 같은 객체

        # 다른 언어로 호출 시 새 모델 생성
        model3 = engine._get_or_create_model("en", use_gpu=False)
        assert len(engine._models) == 2
        assert model1 is not model3

    def test_paddleocr_repr(self):
        """__repr__ 테스트"""
        from beanllm.domain.ocr.engines.paddleocr_engine import (
            PaddleOCREngine,
        )

        engine = PaddleOCREngine()

        repr_before = repr(engine)
        assert "PaddleOCREngine" in repr_before
        assert "models_loaded=0" in repr_before

        # 모델 로드
        engine._get_or_create_model("ko", use_gpu=False)

        repr_after = repr(engine)
        assert "models_loaded=1" in repr_after


class TestPaddleOCREngineIntegration:
    """beanOCR 통합 테스트"""

    @skip_without_paddleocr
    def test_paddleocr_in_bean_ocr(self):
        """beanOCR에서 PaddleOCR 사용 테스트"""
        from beanllm.domain.ocr import beanOCR

        ocr = beanOCR(engine="paddleocr", language="ko")
        assert ocr._engine is not None
        assert ocr._engine.name == "PaddleOCR"

    def test_paddleocr_import_error_handling(self):
        """PaddleOCR 미설치 시 에러 처리 테스트"""
        if not HAS_PADDLEOCR:
            from beanllm.domain.ocr import beanOCR

            with pytest.raises(ImportError, match="PaddleOCR is required"):
                beanOCR(engine="paddleocr")
        else:
            # PaddleOCR가 설치된 경우 정상 동작
            from beanllm.domain.ocr import beanOCR

            ocr = beanOCR(engine="paddleocr")
            assert ocr._engine.name == "PaddleOCR"


# 실제 OCR 테스트 (선택적으로만 실행)
@pytest.mark.slow  # --slow 옵션으로 실행 가능
@skip_without_paddleocr
class TestPaddleOCREngineRealOCR:
    """실제 OCR 기능 테스트 (느림, optional)"""

    def test_paddleocr_recognize_simple_image(self):
        """간단한 이미지 OCR 테스트"""
        from beanllm.domain.ocr.engines.paddleocr_engine import (
            PaddleOCREngine,
        )

        engine = PaddleOCREngine()
        config = OCRConfig(language="en", use_gpu=False)

        # 간단한 테스트 이미지 (검정색 배경에 흰색 텍스트)
        # 실제로는 빈 이미지이므로 빈 결과가 예상됨
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        result = engine.recognize(image, config)

        # 빈 이미지이므로 빈 결과 예상
        assert "text" in result
        assert "lines" in result
        assert "confidence" in result
        assert "language" in result
