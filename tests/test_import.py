"""
Test basic imports
"""

def test_import_registry():
    """Test get_registry import"""
    from beanllm import get_registry
    assert get_registry is not None


def test_import_provider_factory():
    """Test ProviderFactory import"""
    from beanllm import ProviderFactory
    assert ProviderFactory is not None


def test_import_data_classes():
    """Test data class imports"""
    from beanllm import ModelCapabilityInfo, ProviderInfo
    assert ModelCapabilityInfo is not None
    assert ProviderInfo is not None


def test_import_utils():
    """Test utils imports"""
    from beanllm.utils import EnvConfig, ProviderError, retry, get_logger
    assert EnvConfig is not None
    assert ProviderError is not None
    assert retry is not None
    assert get_logger is not None
