"""
Cloud OCR Engine 테스트

Note: google-cloud-vision 또는 boto3가 설치되지 않은 경우 대부분의 테스트는 skip됩니다.
      설치: pip install google-cloud-vision boto3
"""

import numpy as np
import pytest

from beanllm.domain.ocr.models import OCRConfig

# Google Vision 설치 여부 체크
try:
    from google.cloud import vision  # noqa: F401

    HAS_GOOGLE_VISION = True
except ImportError:
    HAS_GOOGLE_VISION = False

# AWS 설치 여부 체크
try:
    import boto3  # noqa: F401

    HAS_AWS = True
except ImportError:
    HAS_AWS = False

skip_without_google = pytest.mark.skipif(
    not HAS_GOOGLE_VISION, reason="google-cloud-vision not installed"
)
skip_without_aws = pytest.mark.skipif(not HAS_AWS, reason="boto3 not installed")


class TestCloudOCREngineImport:
    """CloudOCREngine import 테스트"""

    def test_cloud_engine_import_google_without_dependencies(self):
        """Google Vision 없이 import 시도"""
        if not HAS_GOOGLE_VISION:
            from beanllm.domain.ocr.engines.cloud_engine import CloudOCREngine

            with pytest.raises(ImportError, match="google-cloud-vision is required"):
                CloudOCREngine(provider="google")
        else:
            from beanllm.domain.ocr.engines.cloud_engine import CloudOCREngine

            engine = CloudOCREngine(provider="google")
            assert engine.name == "CloudOCR-GOOGLE"

    def test_cloud_engine_import_aws_without_dependencies(self):
        """AWS 없이 import 시도"""
        if not HAS_AWS:
            from beanllm.domain.ocr.engines.cloud_engine import CloudOCREngine

            with pytest.raises(ImportError, match="boto3 is required"):
                CloudOCREngine(provider="aws")
        else:
            from beanllm.domain.ocr.engines.cloud_engine import CloudOCREngine

            engine = CloudOCREngine(provider="aws")
            assert engine.name == "CloudOCR-AWS"

    def test_cloud_engine_invalid_provider(self):
        """잘못된 provider"""
        from beanllm.domain.ocr.engines.cloud_engine import CloudOCREngine

        with pytest.raises(ValueError, match="Unsupported provider"):
            CloudOCREngine(provider="invalid")


@skip_without_google
class TestCloudOCREngineGoogle:
    """Google Vision이 설치된 경우의 테스트"""

    def test_google_engine_initialization(self):
        """Google Vision 엔진 초기화 테스트"""
        from beanllm.domain.ocr.engines.cloud_engine import CloudOCREngine

        engine = CloudOCREngine(provider="google")
        assert engine.name == "CloudOCR-GOOGLE"
        assert engine.provider == "google"

    def test_google_repr(self):
        """__repr__ 테스트"""
        from beanllm.domain.ocr.engines.cloud_engine import CloudOCREngine

        engine = CloudOCREngine(provider="google")
        repr_str = repr(engine)
        assert "CloudOCREngine" in repr_str
        assert "google" in repr_str


@skip_without_aws
class TestCloudOCREngineAWS:
    """AWS가 설치된 경우의 테스트"""

    def test_aws_engine_initialization(self):
        """AWS Textract 엔진 초기화 테스트"""
        from beanllm.domain.ocr.engines.cloud_engine import CloudOCREngine

        engine = CloudOCREngine(provider="aws")
        assert engine.name == "CloudOCR-AWS"
        assert engine.provider == "aws"

    def test_aws_repr(self):
        """__repr__ 테스트"""
        from beanllm.domain.ocr.engines.cloud_engine import CloudOCREngine

        engine = CloudOCREngine(provider="aws")
        repr_str = repr(engine)
        assert "CloudOCREngine" in repr_str
        assert "aws" in repr_str


class TestCloudOCREngineIntegration:
    """beanOCR 통합 테스트"""

    @skip_without_google
    def test_google_in_bean_ocr(self):
        """beanOCR에서 Google Vision 사용 테스트"""
        from beanllm.domain.ocr import beanOCR

        ocr = beanOCR(engine="cloud-google", language="ko")
        assert ocr._engine is not None
        assert ocr._engine.name == "CloudOCR-GOOGLE"

    @skip_without_aws
    def test_aws_in_bean_ocr(self):
        """beanOCR에서 AWS Textract 사용 테스트"""
        from beanllm.domain.ocr import beanOCR

        ocr = beanOCR(engine="cloud-aws", language="ko")
        assert ocr._engine is not None
        assert ocr._engine.name == "CloudOCR-AWS"

    def test_google_import_error_handling(self):
        """Google Vision 미설치 시 에러 처리 테스트"""
        if not HAS_GOOGLE_VISION:
            from beanllm.domain.ocr import beanOCR

            with pytest.raises(ImportError, match="google-cloud-vision is required"):
                beanOCR(engine="cloud-google")

    def test_aws_import_error_handling(self):
        """AWS 미설치 시 에러 처리 테스트"""
        if not HAS_AWS:
            from beanllm.domain.ocr import beanOCR

            with pytest.raises(ImportError, match="boto3 is required"):
                beanOCR(engine="cloud-aws")


# 실제 OCR 테스트 (API 키 필요, 비용 발생)
@pytest.mark.slow
@pytest.mark.skipif(True, reason="Requires API credentials and incurs costs")
class TestCloudOCREngineRealAPI:
    """실제 API 테스트 (비용 발생, 기본적으로 skip)"""

    def test_google_vision_recognize(self):
        """Google Vision OCR 테스트"""
        pass  # API 키 필요

    def test_aws_textract_recognize(self):
        """AWS Textract OCR 테스트"""
        pass  # AWS 자격증명 필요
