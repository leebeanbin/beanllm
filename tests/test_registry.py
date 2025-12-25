"""
Test ModelRegistry
"""

import pytest
from beanllm import get_registry


def test_get_registry():
    """Test get_registry returns Registry instance"""
    registry = get_registry()
    assert registry is not None


def test_get_registry_singleton():
    """Test get_registry returns same instance"""
    registry1 = get_registry()
    registry2 = get_registry()
    assert registry1 is registry2


def test_get_available_models():
    """Test get_available_models returns list"""
    registry = get_registry()
    models = registry.get_available_models()
    assert isinstance(models, list)
    assert len(models) > 0


def test_get_model_info():
    """Test get_model_info for known model"""
    registry = get_registry()
    model = registry.get_model_info("gpt-4o-mini")
    assert model is not None
    assert model.model_name == "gpt-4o-mini"
    assert model.provider == "openai"


def test_get_active_providers():
    """Test get_active_providers returns list"""
    registry = get_registry()
    providers = registry.get_active_providers()
    assert isinstance(providers, list)
    # At least ollama should be available
    assert len(providers) >= 1


def test_get_summary():
    """Test get_summary returns dict"""
    registry = get_registry()
    summary = registry.get_summary()
    assert isinstance(summary, dict)
    assert "total_providers" in summary
    assert "active_providers" in summary
    assert "total_models" in summary
