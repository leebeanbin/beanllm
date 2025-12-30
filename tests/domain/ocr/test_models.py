"""
OCR 데이터 모델 테스트
"""

import pytest

from beanllm.domain.ocr.models import BoundingBox, OCRConfig, OCRResult, OCRTextLine


class TestBoundingBox:
    """BoundingBox 데이터 모델 테스트"""

    def test_bounding_box_creation(self):
        """BoundingBox 생성 테스트"""
        bbox = BoundingBox(x0=10.0, y0=20.0, x1=100.0, y1=50.0, confidence=0.95)

        assert bbox.x0 == 10.0
        assert bbox.y0 == 20.0
        assert bbox.x1 == 100.0
        assert bbox.y1 == 50.0
        assert bbox.confidence == 0.95

    def test_bounding_box_default_confidence(self):
        """BoundingBox 기본 신뢰도 테스트"""
        bbox = BoundingBox(x0=0, y0=0, x1=10, y1=10)
        assert bbox.confidence == 1.0

    def test_bounding_box_width(self):
        """BoundingBox 너비 계산 테스트"""
        bbox = BoundingBox(x0=10, y0=20, x1=100, y1=50)
        assert bbox.width == 90.0

    def test_bounding_box_height(self):
        """BoundingBox 높이 계산 테스트"""
        bbox = BoundingBox(x0=10, y0=20, x1=100, y1=50)
        assert bbox.height == 30.0

    def test_bounding_box_area(self):
        """BoundingBox 면적 계산 테스트"""
        bbox = BoundingBox(x0=10, y0=20, x1=100, y1=50)
        assert bbox.area == 2700.0  # 90 * 30

    def test_bounding_box_center(self):
        """BoundingBox 중심점 계산 테스트"""
        bbox = BoundingBox(x0=10, y0=20, x1=100, y1=50)
        center_x, center_y = bbox.center
        assert center_x == 55.0  # (10 + 100) / 2
        assert center_y == 35.0  # (20 + 50) / 2

    def test_bounding_box_repr(self):
        """BoundingBox 문자열 표현 테스트"""
        bbox = BoundingBox(x0=10, y0=20, x1=100, y1=50, confidence=0.95)
        repr_str = repr(bbox)
        assert "BoundingBox" in repr_str
        assert "10.0" in repr_str
        assert "0.95" in repr_str


class TestOCRTextLine:
    """OCRTextLine 데이터 모델 테스트"""

    def test_ocr_text_line_creation(self):
        """OCRTextLine 생성 테스트"""
        bbox = BoundingBox(x0=10, y0=20, x1=100, y1=50, confidence=0.95)
        line = OCRTextLine(
            text="안녕하세요", bbox=bbox, confidence=0.92, language="ko"
        )

        assert line.text == "안녕하세요"
        assert line.bbox == bbox
        assert line.confidence == 0.92
        assert line.language == "ko"

    def test_ocr_text_line_default_language(self):
        """OCRTextLine 기본 언어 테스트"""
        bbox = BoundingBox(x0=0, y0=0, x1=10, y1=10)
        line = OCRTextLine(text="Hello", bbox=bbox, confidence=0.9)
        assert line.language == "en"

    def test_ocr_text_line_repr(self):
        """OCRTextLine 문자열 표현 테스트"""
        bbox = BoundingBox(x0=10, y0=20, x1=100, y1=50)
        line = OCRTextLine(text="Hello World", bbox=bbox, confidence=0.92)
        repr_str = repr(line)
        assert "OCRTextLine" in repr_str
        assert "0.92" in repr_str


class TestOCRResult:
    """OCRResult 데이터 모델 테스트"""

    def test_ocr_result_creation(self):
        """OCRResult 생성 테스트"""
        bbox1 = BoundingBox(x0=10, y0=20, x1=100, y1=50)
        bbox2 = BoundingBox(x0=10, y0=60, x1=100, y1=90)
        line1 = OCRTextLine(text="Hello", bbox=bbox1, confidence=0.9)
        line2 = OCRTextLine(text="World", bbox=bbox2, confidence=0.85)

        result = OCRResult(
            text="Hello\nWorld",
            lines=[line1, line2],
            language="en",
            confidence=0.875,
            engine="PaddleOCR",
            processing_time=1.23,
            metadata={"test": True},
        )

        assert result.text == "Hello\nWorld"
        assert len(result.lines) == 2
        assert result.language == "en"
        assert result.confidence == 0.875
        assert result.engine == "PaddleOCR"
        assert result.processing_time == 1.23
        assert result.metadata["test"] is True

    def test_ocr_result_line_count(self):
        """OCRResult 라인 수 테스트"""
        bbox = BoundingBox(x0=0, y0=0, x1=10, y1=10)
        lines = [
            OCRTextLine(text="Line 1", bbox=bbox, confidence=0.9),
            OCRTextLine(text="Line 2", bbox=bbox, confidence=0.8),
            OCRTextLine(text="Line 3", bbox=bbox, confidence=0.7),
        ]
        result = OCRResult(
            text="Test",
            lines=lines,
            language="en",
            confidence=0.8,
            engine="Test",
            processing_time=1.0,
        )
        assert result.line_count == 3

    def test_ocr_result_average_line_confidence(self):
        """OCRResult 평균 라인 신뢰도 테스트"""
        bbox = BoundingBox(x0=0, y0=0, x1=10, y1=10)
        lines = [
            OCRTextLine(text="Line 1", bbox=bbox, confidence=0.9),
            OCRTextLine(text="Line 2", bbox=bbox, confidence=0.8),
            OCRTextLine(text="Line 3", bbox=bbox, confidence=0.7),
        ]
        result = OCRResult(
            text="Test",
            lines=lines,
            language="en",
            confidence=0.8,
            engine="Test",
            processing_time=1.0,
        )
        assert result.average_line_confidence == pytest.approx(0.8, rel=1e-2)

    def test_ocr_result_empty_lines(self):
        """OCRResult 빈 라인 테스트"""
        result = OCRResult(
            text="",
            lines=[],
            language="en",
            confidence=0.0,
            engine="Test",
            processing_time=0.0,
        )
        assert result.line_count == 0
        assert result.average_line_confidence == 0.0

    def test_ocr_result_low_confidence_lines(self):
        """OCRResult 낮은 신뢰도 라인 테스트"""
        bbox = BoundingBox(x0=0, y0=0, x1=10, y1=10)
        lines = [
            OCRTextLine(text="Line 1", bbox=bbox, confidence=0.9),  # 높음
            OCRTextLine(text="Line 2", bbox=bbox, confidence=0.6),  # 낮음
            OCRTextLine(text="Line 3", bbox=bbox, confidence=0.5),  # 낮음
        ]
        result = OCRResult(
            text="Test",
            lines=lines,
            language="en",
            confidence=0.7,
            engine="Test",
            processing_time=1.0,
        )
        low_conf_lines = result.low_confidence_lines
        assert len(low_conf_lines) == 2
        assert low_conf_lines[0].text == "Line 2"
        assert low_conf_lines[1].text == "Line 3"

    def test_ocr_result_default_metadata(self):
        """OCRResult 기본 메타데이터 테스트"""
        result = OCRResult(
            text="Test",
            lines=[],
            language="en",
            confidence=0.9,
            engine="Test",
            processing_time=1.0,
        )
        assert result.metadata == {}

    def test_ocr_result_repr(self):
        """OCRResult 문자열 표현 테스트"""
        result = OCRResult(
            text="Test",
            lines=[],
            language="en",
            confidence=0.9,
            engine="PaddleOCR",
            processing_time=1.0,
        )
        repr_str = repr(result)
        assert "OCRResult" in repr_str
        assert "PaddleOCR" in repr_str
        assert "en" in repr_str


class TestOCRConfig:
    """OCRConfig 데이터 모델 테스트"""

    def test_ocr_config_defaults(self):
        """OCRConfig 기본값 테스트"""
        config = OCRConfig()

        assert config.engine == "paddleocr"
        assert config.language == "auto"
        assert config.use_gpu is True
        assert config.confidence_threshold == 0.5
        assert config.enable_preprocessing is True
        assert config.enable_llm_postprocessing is False

    def test_ocr_config_custom_values(self):
        """OCRConfig 커스텀 값 테스트"""
        config = OCRConfig(
            engine="easyocr",
            language="ko",
            use_gpu=False,
            confidence_threshold=0.7,
            enable_llm_postprocessing=True,
            llm_model="gpt-4o-mini",
        )

        assert config.engine == "easyocr"
        assert config.language == "ko"
        assert config.use_gpu is False
        assert config.confidence_threshold == 0.7
        assert config.enable_llm_postprocessing is True
        assert config.llm_model == "gpt-4o-mini"

    def test_ocr_config_invalid_engine(self):
        """OCRConfig 잘못된 엔진 테스트"""
        with pytest.raises(ValueError, match="Invalid engine"):
            OCRConfig(engine="invalid_engine")

    def test_ocr_config_invalid_confidence_threshold(self):
        """OCRConfig 잘못된 신뢰도 임계값 테스트"""
        with pytest.raises(ValueError, match="confidence_threshold must be between"):
            OCRConfig(confidence_threshold=1.5)

        with pytest.raises(ValueError, match="confidence_threshold must be between"):
            OCRConfig(confidence_threshold=-0.1)

    def test_ocr_config_llm_postprocessing_without_model(self):
        """OCRConfig LLM 후처리 활성화 시 모델 필수 테스트"""
        with pytest.raises(ValueError, match="llm_model must be specified"):
            OCRConfig(enable_llm_postprocessing=True)

    def test_ocr_config_unsupported_language_warning(self):
        """OCRConfig 지원하지 않는 언어 경고 테스트"""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = OCRConfig(language="xyz")
            assert len(w) == 1
            assert "may not be supported" in str(w[0].message)

    def test_ocr_config_preprocessing_options(self):
        """OCRConfig 전처리 옵션 테스트"""
        config = OCRConfig(
            denoise=False, contrast_adjustment=False, rotation_correction=False
        )

        assert config.denoise is False
        assert config.contrast_adjustment is False
        assert config.rotation_correction is False

    def test_ocr_config_postprocessing_options(self):
        """OCRConfig 후처리 옵션 테스트"""
        config = OCRConfig(
            enable_llm_postprocessing=True,
            llm_model="gpt-4o-mini",
            spell_check=True,
            grammar_check=True,
        )

        assert config.spell_check is True
        assert config.grammar_check is True

    def test_ocr_config_repr(self):
        """OCRConfig 문자열 표현 테스트"""
        config = OCRConfig(engine="paddleocr", language="ko")
        repr_str = repr(config)
        assert "OCRConfig" in repr_str
        assert "paddleocr" in repr_str
        assert "ko" in repr_str


class TestOCRConfigEngines:
    """OCRConfig 엔진별 설정 테스트"""

    def test_paddleocr_config(self):
        """PaddleOCR 설정 테스트"""
        config = OCRConfig(engine="paddleocr", language="ko", use_gpu=True)
        assert config.engine == "paddleocr"

    def test_easyocr_config(self):
        """EasyOCR 설정 테스트"""
        config = OCRConfig(engine="easyocr", language="en")
        assert config.engine == "easyocr"

    def test_trocr_config(self):
        """TrOCR 설정 테스트 (손글씨)"""
        config = OCRConfig(engine="trocr", language="en", use_gpu=True)
        assert config.engine == "trocr"

    def test_nougat_config(self):
        """Nougat 설정 테스트 (학술 논문)"""
        config = OCRConfig(engine="nougat", language="en")
        assert config.engine == "nougat"

    def test_surya_config(self):
        """Surya 설정 테스트 (복잡한 레이아웃)"""
        config = OCRConfig(engine="surya", language="auto")
        assert config.engine == "surya"

    def test_tesseract_config(self):
        """Tesseract 설정 테스트 (Fallback)"""
        config = OCRConfig(engine="tesseract", language="en")
        assert config.engine == "tesseract"

    def test_cloud_config(self):
        """Cloud API 설정 테스트"""
        config = OCRConfig(engine="cloud", language="auto")
        assert config.engine == "cloud"
