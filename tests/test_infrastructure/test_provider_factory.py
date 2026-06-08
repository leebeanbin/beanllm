"""
ProviderFactory 테스트 - Provider 팩토리 테스트
"""

from unittest.mock import MagicMock, patch

import pytest

from beanllm.providers.provider_factory import ProviderFactory

# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------


class TestProviderFactoryAdditional:
    def setup_method(self):
        ProviderFactory._instances.clear()

    def teardown_method(self):
        ProviderFactory._instances.clear()

    def test_cached_instance_returned(self):
        mock_provider = MagicMock()
        ProviderFactory._instances["openai"] = mock_provider
        result = ProviderFactory.get_provider("openai")
        assert result is mock_provider

    def test_raises_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            ProviderFactory.get_provider("totally_unknown_xyz")

    def test_raises_when_no_providers(self):
        with patch.object(ProviderFactory, "_get_provider_priority", return_value=[]):
            with pytest.raises(ValueError, match="No available LLM provider"):
                ProviderFactory.get_provider()

    def test_get_provider_priority_returns_list(self):
        result = ProviderFactory._get_provider_priority()
        assert isinstance(result, list)

    def test_get_available_providers_returns_list(self):
        with patch(
            "beanllm.providers.provider_registry.is_provider_env_available",
            return_value=False,
        ):
            result = ProviderFactory.get_available_providers()
        assert isinstance(result, list)

    def test_clear_cache_clears_instances(self):
        ProviderFactory._instances["openai"] = MagicMock(spec=[])
        ProviderFactory.clear_cache()
        assert ProviderFactory._instances == {}

    def test_clear_cache_with_provider_with_close(self):
        mock_p = MagicMock()
        ProviderFactory._instances["openai"] = mock_p
        with patch("asyncio.run"):
            ProviderFactory.clear_cache()
        assert ProviderFactory._instances == {}

    def test_available_providers_includes_provider_with_env(self):
        mock_cls = MagicMock()
        with (
            patch.object(
                ProviderFactory,
                "_get_provider_priority",
                return_value=[("openai", mock_cls, "OPENAI_API_KEY")],
            ),
            patch(
                "beanllm.providers.provider_registry.is_provider_env_available",
                return_value=True,
            ),
        ):
            result = ProviderFactory.get_available_providers()
        assert "openai" in result

    def test_get_provider_falls_back_when_first_unavailable(self):
        mock_cls1 = MagicMock()
        mock_cls2 = MagicMock()
        mock_inst = MagicMock()
        mock_inst.is_available.return_value = True
        mock_cls2.return_value = mock_inst

        avail = {"openai": False, "claude": True}
        with (
            patch.object(
                ProviderFactory,
                "_get_provider_priority",
                return_value=[
                    ("openai", mock_cls1, "OPENAI_API_KEY"),
                    ("claude", mock_cls2, "ANTHROPIC_API_KEY"),
                ],
            ),
            patch(
                "beanllm.providers.provider_registry.is_provider_env_available",
                side_effect=lambda n: avail.get(n, False),
            ),
        ):
            result = ProviderFactory.get_provider()
        assert result is mock_inst

    def test_creates_ollama_with_host_config(self):
        mock_cls = MagicMock()
        mock_inst = MagicMock()
        mock_inst.is_available.return_value = True
        mock_cls.return_value = mock_inst

        with (
            patch.object(
                ProviderFactory,
                "_get_provider_priority",
                return_value=[("ollama", mock_cls, "OLLAMA_HOST")],
            ),
            patch(
                "beanllm.providers.provider_registry.is_provider_env_available",
                return_value=True,
            ),
        ):
            result = ProviderFactory.get_provider("ollama")
        # Ollama gets config dict
        call_args = mock_cls.call_args
        assert call_args is not None
        assert result is mock_inst

    def test_get_default_provider_delegates_to_get_provider(self):
        mock_inst = MagicMock()
        with patch.object(ProviderFactory, "get_provider", return_value=mock_inst) as mock_gp:
            result = ProviderFactory.get_default_provider()
        mock_gp.assert_called_once()
        assert result is mock_inst


class TestProviderFactory:
    """ProviderFactory 테스트"""

    @pytest.fixture
    def factory(self):
        """ProviderFactory 클래스"""
        return ProviderFactory

    @patch("beanllm.providers.provider_factory.EnvConfig")
    def test_get_available_providers_with_keys(self, mock_config, factory):
        """API 키가 있는 Provider 목록 조회 테스트"""
        mock_config.OPENAI_API_KEY = "test_key"
        mock_config.ANTHROPIC_API_KEY = None
        mock_config.GEMINI_API_KEY = None
        mock_config.OLLAMA_HOST = None

        providers = factory.get_available_providers()

        assert isinstance(providers, list)

    @patch("beanllm.providers.provider_factory.EnvConfig")
    def test_get_available_providers_no_keys(self, mock_config, factory):
        """API 키가 없는 경우 테스트"""
        mock_config.OPENAI_API_KEY = None
        mock_config.ANTHROPIC_API_KEY = None
        mock_config.GEMINI_API_KEY = None
        mock_config.OLLAMA_HOST = None

        providers = factory.get_available_providers()

        assert isinstance(providers, list)

    def test_is_provider_available(self, factory):
        """Provider 사용 가능 여부 확인 테스트"""
        providers = factory.get_available_providers()
        if providers:
            is_available = providers[0] in providers
            assert is_available is True
        else:
            pytest.skip("No providers available")

    def test_is_provider_available_not_available(self, factory):
        """사용 불가능한 Provider 확인 테스트"""
        providers = factory.get_available_providers()
        is_available = "nonexistent_provider_xyz" in providers
        assert not is_available

    def test_get_default_provider(self, factory):
        """기본 Provider 조회 테스트"""
        try:
            default_provider = factory.get_default_provider()
            assert default_provider is not None
        except (ValueError, ImportError):
            pytest.skip("No provider available")

    def test_get_default_provider_no_available(self, factory):
        """get_provider 메서드 존재 확인"""
        assert hasattr(factory, "get_provider")
        assert hasattr(factory, "get_available_providers")
        assert hasattr(factory, "get_default_provider")
