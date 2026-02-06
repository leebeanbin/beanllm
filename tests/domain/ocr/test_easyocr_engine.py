"""
EasyOCR Engine 테스트

Note: EasyOCR이 설치되지 않은 경우 대부분의 테스트는 skip됩니다.
      설치: pip install easyocr
"""

import numpy as np
import pytest

from beanllm.domain.ocr.models import OCRConfig

# EasyOCR 설치 여부 체크
try:
    import easyocr  # noqa: F401

    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

skip_without_easyocr = pytest.mark.skipif(not HAS_EASYOCR, reason="EasyOCR not installed")


class TestEasyOCREngineImport:
    """EasyOCREngine import 테스트"""

    def test_easyocr_engine_import_without_easyocr(self):
        """easyocr 없이 import 시도 (의존성 체크 테스트)"""
        if not HAS_EASYOCR:
            # easyocr가 없을 때 EasyOCREngine import는 성공하지만
            # 초기화 시 ImportError 발생
            from beanllm.domain.ocr.engines.easyocr_engine import EasyOCREngine

            with pytest.raises(ImportError, match="EasyOCR is required"):
                EasyOCREngine()
        else:
            # easyocr가 있을 때는 정상 동작
            from beanllm.domain.ocr.engines.easyocr_engine import EasyOCREngine

            engine = EasyOCREngine()
            assert engine.name == "EasyOCR"


@skip_without_easyocr
class TestEasyOCREngineWithEasyOCR:
    """EasyOCR이 설치된 경우의 테스트"""

    def test_easyocr_engine_initialization(self):
        """EasyOCREngine 초기화 테스트"""
        from beanllm.domain.ocr.engines.easyocr_engine import EasyOCREngine

        engine = EasyOCREngine()
        assert engine.name == "EasyOCR"
        assert len(engine._readers) == 0

    def test_easyocr_language_code_mapping(self):
        """언어 코드 매핑 테스트"""
        from beanllm.domain.ocr.engines.easyocr_engine import EasyOCREngine

        engine = EasyOCREngine()

        assert engine._get_language_code("ko") == "ko"
        assert engine._get_language_code("en") == "en"
        assert engine._get_language_code("zh") == "ch_sim"
        assert engine._get_language_code("ja") == "ja"
        assert engine._get_language_code("auto") == "en"
        assert engine._get_language_code("unknown") == "en"

    def test_easyocr_reader_caching(self):
        """Reader 캐싱 테스트 (실제 모델 로드 없이)"""
        from beanllm.domain.ocr.engines.easyocr_engine import EasyOCREngine

        engine = EasyOCREngine()

        # Reader가 아직 로드되지 않음
        assert len(engine._readers) == 0

        # _get_or_create_reader 호출 시 Reader 캐싱 확인
        reader1 = engine._get_or_create_reader("ko", use_gpu=False)
        assert len(engine._readers) == 1

        # 같은 언어/GPU 설정으로 재호출 시 캐시 사용
        reader2 = engine._get_or_create_reader("ko", use_gpu=False)
        assert len(engine._readers) == 1
        assert reader1 is reader2  # 같은 객체

        # 다른 언어로 호출 시 새 Reader 생성
        reader3 = engine._get_or_create_reader("ja", use_gpu=False)
        assert len(engine._readers) == 2
        assert reader1 is not reader3

    def test_easyocr_repr(self):
        """__repr__ 테스트"""
        from beanllm.domain.ocr.engines.easyocr_engine import EasyOCREngine

        engine = EasyOCREngine()

        repr_before = repr(engine)
        assert "EasyOCREngine" in repr_before
        assert "readers_loaded=0" in repr_before

        # Reader 로드
        engine._get_or_create_reader("ko", use_gpu=False)

        repr_after = repr(engine)
        assert "readers_loaded=1" in repr_after


class TestEasyOCREngineIntegration:
    """beanOCR 통합 테스트"""

    @skip_without_easyocr
    def test_easyocr_in_bean_ocr(self):
        """beanOCR에서 EasyOCR 사용 테스트"""
        from beanllm.domain.ocr import beanOCR

        ocr = beanOCR(engine="easyocr", language="ko")
        assert ocr._engine is not None
        assert ocr._engine.name == "EasyOCR"

    def test_easyocr_import_error_handling(self):
        """EasyOCR 미설치 시 에러 처리 테스트"""
        if not HAS_EASYOCR:
            from beanllm.domain.ocr import beanOCR

            with pytest.raises(ImportError, match="EasyOCR is required"):
                beanOCR(engine="easyocr")
        else:
            # EasyOCR이 설치된 경우 정상 동작
            from beanllm.domain.ocr import beanOCR

            ocr = beanOCR(engine="easyocr")
            assert ocr._engine.name == "EasyOCR"


# 실제 OCR 테스트 (선택적으로만 실행)
@pytest.mark.slow  # --slow 옵션으로 실행 가능
@skip_without_easyocr
class TestEasyOCREngineRealOCR:
    """실제 OCR 기능 테스트 (느림, optional)"""

    def test_easyocr_recognize_simple_image(self):
        """간단한 이미지 OCR 테스트"""
        from beanllm.domain.ocr.engines.easyocr_engine import EasyOCREngine

        engine = EasyOCREngine()
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
