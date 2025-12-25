"""
Infrastructure Layer 테스트 - 외부 시스템 인터페이스 테스트
"""

import pytest

try:
    from beanllm.infrastructure import (
        ModelRegistry,
        get_model_registry,
        ParameterAdapter,
        adapt_parameters,
    )
except ImportError:
    from src.beanllm.infrastructure import (
        ModelRegistry,
        get_model_registry,
        ParameterAdapter,
        adapt_parameters,
    )


class TestModelRegistry:
    """ModelRegistry 테스트"""

    def test_get_model_registry(self):
        """get_model_registry 테스트"""
        registry = get_model_registry()
        assert isinstance(registry, ModelRegistry)

    def test_get_model_registry_singleton(self):
        """get_model_registry 싱글톤 테스트"""
        registry1 = get_model_registry()
        registry2 = get_model_registry()
        assert registry1 is registry2

    def test_get_available_models(self):
        """사용 가능한 모델 목록 테스트"""
        registry = get_model_registry()
        models = registry.get_available_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_get_model_info(self):
        """모델 정보 조회 테스트"""
        registry = get_model_registry()
        model = registry.get_model_info("gpt-4o-mini")
        assert model is not None
        assert model.model_name == "gpt-4o-mini"
        assert model.provider == "openai"

    def test_get_model_info_not_found(self):
        """존재하지 않는 모델 조회 테스트"""
        registry = get_model_registry()
        model = registry.get_model_info("nonexistent-model-xyz")
        assert model is None

    def test_get_active_providers(self):
        """활성 Provider 목록 테스트"""
        registry = get_model_registry()
        providers = registry.get_active_providers()
        assert isinstance(providers, list)
        assert len(providers) >= 0  # Provider가 없을 수 있음

    def test_get_summary(self):
        """요약 정보 테스트"""
        registry = get_model_registry()
        summary = registry.get_summary()
        assert isinstance(summary, dict)
        assert "total_providers" in summary
        assert "active_providers" in summary
        assert "total_models" in summary


class TestParameterAdapter:
    """ParameterAdapter 테스트"""

    def test_adapt_parameters_basic(self):
        """기본 파라미터 변환 테스트"""
        from beanllm.infrastructure.adapter import AdaptedParameters

        params = {"temperature": 0.7, "max_tokens": 1000}
        adapted = adapt_parameters("openai", "gpt-4o", params)
        assert isinstance(adapted, AdaptedParameters)
        assert "temperature" in adapted.params

    def test_adapt_parameters_max_tokens(self):
        """max_tokens 파라미터 변환 테스트"""
        params = {"max_tokens": 1000}

        # OpenAI
        adapted = adapt_parameters("openai", "gpt-4o", params)
        assert "max_tokens" in adapted.params or "max_completion_tokens" in adapted.params

        # Gemini (max_output_tokens로 변환)
        adapted = adapt_parameters("gemini", "gemini-2.0-flash-exp", params)
        assert "max_output_tokens" in adapted.params or "max_tokens" in adapted.params

    def test_adapt_parameters_temperature(self):
        """temperature 파라미터 테스트"""
        params = {"temperature": 0.7}
        adapted = adapt_parameters("openai", "gpt-4o", params)
        assert adapted.params.get("temperature") == 0.7

    def test_validate_parameters(self):
        """파라미터 검증 테스트"""
        try:
            from beanllm.infrastructure import validate_parameters
        except ImportError:
            from src.beanllm.infrastructure import validate_parameters

        params = {"temperature": 0.7, "max_tokens": 1000}
        # 에러 없이 실행되어야 함
        try:
            validate_parameters("openai", "gpt-4o", params)
        except Exception as e:
            pytest.fail(f"validate_parameters failed: {e}")


class TestProviderFactory:
    """ProviderFactory 테스트"""

    def test_provider_factory_get_available_providers(self):
        """사용 가능한 Provider 목록 테스트"""
        try:
            from beanllm.infrastructure.provider import ProviderFactory
        except ImportError:
            from src.beanllm.infrastructure.provider import ProviderFactory

        providers = ProviderFactory.get_available_providers()
        assert isinstance(providers, list)

    def test_provider_factory_get_provider(self):
        """Provider 생성 테스트"""
        try:
            from beanllm._source_providers.provider_factory import ProviderFactory
        except ImportError:
            from src.beanllm._source_providers.provider_factory import ProviderFactory

        # Provider가 없을 수 있으므로 try-except
        try:
            provider = ProviderFactory.get_provider("openai")
            assert provider is not None
        except (ValueError, ImportError, AttributeError):
            pytest.skip("OpenAI provider not available")

    def test_provider_factory_get_default_provider(self):
        """기본 Provider 조회 테스트"""
        try:
            from beanllm.infrastructure.provider import ProviderFactory
        except ImportError:
            from src.beanllm.infrastructure.provider import ProviderFactory

        try:
            provider = ProviderFactory.get_default_provider()
            assert provider is not None
        except (ValueError, ImportError):
            pytest.skip("No provider available")


class TestInfrastructureIntegration:
    """Infrastructure 레이어 통합 테스트"""

    def test_registry_and_adapter_integration(self):
        """Registry와 Adapter 통합 테스트"""
        from beanllm.infrastructure.adapter import AdaptedParameters

        registry = get_model_registry()
        model = registry.get_model_info("gpt-4o-mini")

        if model:
            params = {"temperature": 0.7, "max_tokens": 1000}
            adapted = adapt_parameters(model.provider, model.model_name, params)
            assert isinstance(adapted, AdaptedParameters)
            assert "temperature" in adapted.params

