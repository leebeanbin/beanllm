"""
Model Definitions - Backward Compatibility
ModelConfigManager를 사용하되 기존 dict 형태 API 유지
"""

from typing import Dict, Optional

from .model_config import ModelConfig, ModelConfigManager
from .llm_provider import LLMProvider


def _model_config_to_dict(config: ModelConfig) -> Dict:
    """ModelConfig를 dict로 변환"""
    return {
        "name": config.name,
        "display_name": config.display_name,
        "provider": config.provider.value,
        "type": config.type,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "description": config.description,
        "use_case": config.use_case,
        "supports_temperature": config.supports_temperature,
        "supports_max_tokens": config.supports_max_tokens,
        "uses_max_completion_tokens": config.uses_max_completion_tokens,
    }


# Backward compatibility: ModelConfigManager.MODELS를 dict 형태로 변환
MODELS = {
    name: _model_config_to_dict(config)
    for name, config in ModelConfigManager.MODELS.items()
}


def get_all_models() -> Dict[str, Dict]:
    """모든 모델 정보 조회"""
    return {
        name: _model_config_to_dict(config)
        for name, config in ModelConfigManager.MODELS.items()
    }


def get_models_by_provider(provider: str) -> Dict[str, Dict]:
    """제공자별 모델 조회"""
    provider_map = {
        "openai": LLMProvider.OPENAI,
        "anthropic": LLMProvider.ANTHROPIC,
        "claude": LLMProvider.ANTHROPIC,
        "google": LLMProvider.GOOGLE,
        "gemini": LLMProvider.GOOGLE,
        "ollama": LLMProvider.OLLAMA,
        "deepseek": LLMProvider.DEEPSEEK,
        "perplexity": LLMProvider.PERPLEXITY,
    }
    
    normalized_provider = provider_map.get(provider.lower(), provider.lower())
    
    # Enum인 경우
    if isinstance(normalized_provider, LLMProvider):
        models = ModelConfigManager.get_models_by_provider(normalized_provider)
    else:
        # 문자열인 경우 직접 필터링
        models = {
            name: config
            for name, config in ModelConfigManager.MODELS.items()
            if config.provider.value.lower() == normalized_provider.lower()
        }
    
    return {name: _model_config_to_dict(config) for name, config in models.items()}


def get_models_by_type(model_type: str) -> Dict[str, Dict]:
    """타입별 모델 조회"""
    models = ModelConfigManager.get_models_by_type(model_type)
    return {name: _model_config_to_dict(config) for name, config in models.items()}


def get_default_model(provider: Optional[str] = None, model_type: str = "llm") -> Optional[str]:
    """기본 모델 조회"""
    from beanllm.utils.config import EnvConfig

    if provider:
        provider_map = {
            "openai": LLMProvider.OPENAI,
            "anthropic": LLMProvider.ANTHROPIC,
            "claude": LLMProvider.ANTHROPIC,
            "google": LLMProvider.GOOGLE,
            "gemini": LLMProvider.GOOGLE,
            "ollama": LLMProvider.OLLAMA,
            "deepseek": LLMProvider.DEEPSEEK,
            "perplexity": LLMProvider.PERPLEXITY,
        }
        normalized_provider = provider_map.get(provider.lower())
        if normalized_provider:
            models = ModelConfigManager.get_models_by_provider(normalized_provider)
            for name, config in models.items():
                if config.type == model_type:
                    return name
    else:
        if model_type == "slm":
            return "phi3.5"
        elif model_type == "llm":
            if EnvConfig.ANTHROPIC_API_KEY:
                return "claude-3-5-sonnet-20241022"
            elif EnvConfig.OPENAI_API_KEY:
                return "gpt-4o-mini"
            elif EnvConfig.GEMINI_API_KEY:
                return "gemini-1.5-flash"
            elif EnvConfig.DEEPSEEK_API_KEY:
                return "deepseek-chat"
            elif EnvConfig.PERPLEXITY_API_KEY:
                return "sonar"
            else:
                return "qwen2.5:7b"
    return None
