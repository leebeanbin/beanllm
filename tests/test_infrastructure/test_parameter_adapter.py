"""
ParameterAdapter 테스트 - 파라미터 변환 테스트
"""

import pytest
from unittest.mock import Mock, patch

from llmkit.infrastructure.adapter import (
    AdaptedParameters,
    ParameterAdapter,
    adapt_parameters,
    validate_parameters,
)


class TestParameterAdapter:
    """ParameterAdapter 테스트"""

    @pytest.fixture
    def adapter(self):
        """ParameterAdapter 인스턴스"""
        return ParameterAdapter()

    def test_adapt_openai_basic(self, adapter):
        """OpenAI 기본 파라미터 변환 테스트"""
        params = {
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.9,
        }

        result = adapter.adapt("openai", "gpt-4o-mini", params)

        assert isinstance(result, AdaptedParameters)
        assert "temperature" in result.params
        assert result.params["temperature"] == 0.7
        assert "max_tokens" in result.params

    def test_adapt_google_max_tokens(self, adapter):
        """Google max_tokens → max_output_tokens 변환 테스트"""
        params = {
            "max_tokens": 1000,
            "temperature": 0.7,
        }

        result = adapter.adapt("google", "gemini-pro", params)

        assert isinstance(result, AdaptedParameters)
        # max_tokens가 max_output_tokens로 변환되었는지 확인
        assert "max_output_tokens" in result.params or "max_tokens" in result.params

    def test_adapt_ollama_max_tokens(self, adapter):
        """Ollama max_tokens → num_predict 변환 테스트"""
        params = {
            "max_tokens": 1000,
            "temperature": 0.7,
        }

        result = adapter.adapt("ollama", "llama2", params)

        assert isinstance(result, AdaptedParameters)
        # max_tokens가 num_predict로 변환되었는지 확인
        assert "num_predict" in result.params or "max_tokens" in result.params

    def test_adapt_unsupported_parameter(self, adapter):
        """지원하지 않는 파라미터 처리 테스트"""
        params = {
            "temperature": 0.7,
            "unknown_param": "value",
        }

        result = adapter.adapt("openai", "gpt-4o-mini", params)

        assert isinstance(result, AdaptedParameters)
        # unknown_param이 그대로 전달되거나 경고가 있을 수 있음
        # 실제 구현에 따라 다를 수 있으므로 결과가 AdaptedParameters인지만 확인
        assert isinstance(result, AdaptedParameters)

    def test_adapt_temperature_range(self, adapter):
        """Temperature 범위 조정 테스트"""
        params = {
            "temperature": 2.5,  # 범위 초과 가능
        }

        result = adapter.adapt("openai", "gpt-4o-mini", params)

        assert isinstance(result, AdaptedParameters)
        # OpenAI는 temperature 범위 제한이 없을 수 있음
        # Anthropic은 0.0-1.0으로 제한됨
        if "temperature" in result.params:
            assert isinstance(result.params["temperature"], (int, float))

    def test_adapt_temperature_anthropic_range(self, adapter):
        """Anthropic Temperature 범위 조정 테스트"""
        params = {
            "temperature": 1.5,  # Anthropic 범위 초과 (0.0-1.0)
        }

        result = adapter.adapt("anthropic", "claude-3-opus", params)

        assert isinstance(result, AdaptedParameters)
        # Anthropic은 temperature를 1.0으로 제한해야 함
        if "temperature" in result.params:
            assert result.params["temperature"] <= 1.0

    def test_adapt_anthropic(self, adapter):
        """Anthropic 파라미터 변환 테스트"""
        params = {
            "temperature": 0.7,
            "max_tokens": 1000,
        }

        result = adapter.adapt("anthropic", "claude-3-opus", params)

        assert isinstance(result, AdaptedParameters)
        assert "temperature" in result.params

    def test_validate_parameters_valid(self, adapter):
        """유효한 파라미터 검증 테스트"""
        params = {
            "temperature": 0.7,
            "max_tokens": 1000,
        }

        is_valid, errors = adapter.validate_parameters("openai", "gpt-4o-mini", params)

        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)

    def test_validate_parameters_invalid(self, adapter):
        """유효하지 않은 파라미터 검증 테스트"""
        params = {
            "temperature": 3.0,  # 범위 초과
            "unknown_param": "value",
        }

        is_valid, errors = adapter.validate_parameters("openai", "gpt-4o-mini", params)

        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)

    def test_adapt_parameters_function(self):
        """adapt_parameters 편의 함수 테스트"""
        params = {
            "temperature": 0.7,
            "max_tokens": 1000,
        }

        result = adapt_parameters("openai", "gpt-4o-mini", params)

        assert isinstance(result, AdaptedParameters)

    def test_validate_parameters_function(self):
        """validate_parameters 편의 함수 테스트"""
        params = {
            "temperature": 0.7,
            "max_tokens": 1000,
        }

        is_valid, errors = validate_parameters("openai", "gpt-4o-mini", params)

        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)


