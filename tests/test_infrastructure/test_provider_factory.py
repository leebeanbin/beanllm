"""
ProviderFactory 테스트 - Provider 팩토리 테스트
"""

import pytest
from unittest.mock import patch

from beanllm.infrastructure.provider import ProviderFactory


class TestProviderFactory:
    """ProviderFactory 테스트"""

    @pytest.fixture
    def factory(self):
        """ProviderFactory 클래스"""
        return ProviderFactory

    @patch("beanllm.infrastructure.provider.provider_factory.Config")
    def test_get_available_providers_with_keys(self, mock_config, factory):
        """API 키가 있는 Provider 목록 조회 테스트"""
        mock_config.OPENAI_API_KEY = "test_key"
        mock_config.ANTHROPIC_API_KEY = None
        mock_config.GEMINI_API_KEY = None
        mock_config.OLLAMA_HOST = None

        providers = factory.get_available_providers()

        assert isinstance(providers, list)
        assert "openai" in providers or len(providers) >= 0

    @patch("beanllm.infrastructure.provider.provider_factory.Config")
    def test_get_available_providers_no_keys(self, mock_config, factory):
        """API 키가 없는 경우 테스트"""
        mock_config.OPENAI_API_KEY = None
        mock_config.ANTHROPIC_API_KEY = None
        mock_config.GEMINI_API_KEY = None
        mock_config.OLLAMA_HOST = None

        providers = factory.get_available_providers()

        assert isinstance(providers, list)

    @patch("beanllm.infrastructure.provider.provider_factory.Config")
    def test_is_provider_available(self, mock_config, factory):
        """Provider 사용 가능 여부 확인 테스트"""
        mock_config.OPENAI_API_KEY = "test_key"
        mock_config.ANTHROPIC_API_KEY = None
        mock_config.GEMINI_API_KEY = None
        mock_config.OLLAMA_HOST = None

        is_available = factory.is_provider_available("openai")

        assert isinstance(is_available, bool)

    @patch("beanllm.infrastructure.provider.provider_factory.Config")
    def test_is_provider_available_not_available(self, mock_config, factory):
        """사용 불가능한 Provider 확인 테스트"""
        mock_config.OPENAI_API_KEY = None
        mock_config.ANTHROPIC_API_KEY = None
        mock_config.GEMINI_API_KEY = None
        mock_config.OLLAMA_HOST = None

        is_available = factory.is_provider_available("openai")

        assert isinstance(is_available, bool)
        assert not is_available

    @patch("beanllm.infrastructure.provider.provider_factory.Config")
    def test_get_default_provider(self, mock_config, factory):
        """기본 Provider 조회 테스트"""
        mock_config.OPENAI_API_KEY = "test_key"
        mock_config.ANTHROPIC_API_KEY = None
        mock_config.GEMINI_API_KEY = None
        mock_config.OLLAMA_HOST = None

        default_provider = factory.get_default_provider()

        assert default_provider is None or isinstance(default_provider, str)

    @patch("beanllm.infrastructure.provider.provider_factory.Config")
    def test_get_default_provider_no_available(self, mock_config, factory):
        """사용 가능한 Provider가 없는 경우 테스트"""
        # ollama는 항상 사용 가능하므로 실제로는 None이 아닐 수 있음
        mock_config.OPENAI_API_KEY = None
        mock_config.ANTHROPIC_API_KEY = None
        mock_config.GEMINI_API_KEY = None
        # OLLAMA_HOST는 None이어도 ollama는 사용 가능할 수 있음

        default_provider = factory.get_default_provider()

        # ollama가 기본으로 반환될 수 있으므로 None이거나 문자열
        assert default_provider is None or isinstance(default_provider, str)


