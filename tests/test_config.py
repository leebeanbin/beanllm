"""
Test EnvConfig
"""

from beanllm.utils import EnvConfig


def test_env_config_exists():
    """Test EnvConfig class exists"""
    assert EnvConfig is not None


def test_get_active_providers():
    """Test get_active_providers returns list"""
    providers = EnvConfig.get_active_providers()
    assert isinstance(providers, list)
    # At least ollama should be in the list
    assert "ollama" in providers


def test_is_provider_available():
    """Test is_provider_available"""
    # Ollama is always available
    assert EnvConfig.is_provider_available("ollama") is True
